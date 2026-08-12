use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use std::thread;

use aec3::api::config::EchoCanceller3Config;
use aec3::api::control::EchoControl;
use aec3::audio_processing::stream_config::StreamConfig;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use aec3::audio_processing::aec3::echo_canceller3::EchoCanceller3;
use aec3::audio_processing::audio_buffer::AudioBuffer;
use aec3::audio_processing::high_pass_filter::HighPassFilter;

fn interleaved_to_channels(interleaved: &[f32], channels: usize, frames: usize) -> Vec<Vec<f32>> {
    let avail_frames = interleaved.len() / channels;
    let mut out = vec![vec![0f32; frames]; channels];
    let copy_frames = std::cmp::min(avail_frames, frames);
    for frame in 0..copy_frames {
        for ch in 0..channels {
            out[ch][frame] = interleaved[frame * channels + ch];
        }
    }
    out
}

fn channels_to_interleaved(channels_data: &mut [&[f32]], out: &mut [f32]) {
    let channels = channels_data.len();
    let frames = channels_data[0].len();
    for frame in 0..frames {
        for ch in 0..channels {
            out[frame * channels + ch] = channels_data[ch][frame];
        }
    }
}

fn processing_thread(
    rx_in: Receiver<Vec<f32>>,
    rx_render: Receiver<Vec<f32>>,
    tx_out: Sender<Vec<f32>>,
    sample_rate: usize,
    channels: usize,
) {
    // Create AEC instance for the given channels.
    let cfg = EchoCanceller3Config::default();
    //cfg.delay.use_external_delay_estimator = false;
    let mut aec3 = EchoCanceller3::with_multichannel_config(
        cfg,
        Some(EchoCanceller3Config::create_default_multichannel_config()),
        sample_rate as i32,
        channels,
        channels,
    );

    // AudioBuffer used for conversion
    let mut audio_buf =
        AudioBuffer::from_sample_rates(sample_rate, channels, sample_rate, channels, sample_rate);
    let stream_config = StreamConfig::new(sample_rate, channels, false);

    let mut last_metrics = std::time::Instant::now();
    let metrics_interval = std::time::Duration::from_secs(5);

    // Separate buffer for render frames (to call analyze_render)
    let mut render_buf =
        AudioBuffer::from_sample_rates(sample_rate, channels, sample_rate, channels, sample_rate);

    while let Ok(frame) = rx_in.recv() {
        // If there's a pending render frame, feed it to the AEC first.
        if let Ok(render_frame) = rx_render.try_recv() {
            let per_channel_render =
                interleaved_to_channels(&render_frame, channels, stream_config.num_frames());
            let refs_render: Vec<&[f32]> =
                per_channel_render.iter().map(|v| v.as_slice()).collect();
            render_buf.copy_from(&refs_render, &stream_config);
            // Mirror reference demo: split, analyze_render, then merge
            render_buf.split_into_frequency_bands();
            aec3.analyze_render(&mut render_buf);
            render_buf.merge_frequency_bands();
        }
        // Convert interleaved input into per-channel slices
        let per_channel = interleaved_to_channels(&frame, channels, stream_config.num_frames());
        let refs: Vec<&[f32]> = per_channel.iter().map(|v| v.as_slice()).collect();

        // Copy into audio buffer
        audio_buf.copy_from(&refs, &stream_config);

        // Analyze capture and run AEC processing. This mirrors the reference demo:
        //  - analyze_capture on time-domain buffer
        //  - split into frequency bands
        //  - apply high-pass filter to the lowest band (band 0)
        //  - set audio buffer delay (0 for this example)
        //  - process_capture to remove echo
        aec3.analyze_capture(&mut audio_buf);
        // Split into frequency bands if needed by the AEC internals
        audio_buf.split_into_frequency_bands();

        // Apply the high-pass filter to the capture (low band) as in the demo
        let mut hp_filter_channels: Vec<Vec<f32>> = (0..channels)
            .map(|ch| audio_buf.split_band(ch, 0).to_vec())
            .collect();
        let mut hp_filter = HighPassFilter::new(sample_rate as i32, channels);
        hp_filter.process(&mut hp_filter_channels);
        for ch in 0..channels {
            let dst = audio_buf.split_band_mut(ch, 0);
            dst.copy_from_slice(&hp_filter_channels[ch]);
        }

        aec3.set_audio_buffer_delay(116);
        aec3.process_capture(&mut audio_buf, false);

        // Merge bands back to time-domain buffer
        audio_buf.merge_frequency_bands();

        // Periodically print metrics every `metrics_interval`
        if last_metrics.elapsed() >= metrics_interval {
            let metrics = aec3.metrics();
            println!("AEC metrics: {:?}", metrics);
            last_metrics = std::time::Instant::now();
        }

        // After processing, copy processed data back to interleaved vector
        let mut output = vec![0f32; frame.len()];
        // Prepare mutable slices for copy_to_stream
        let mut out_mut: Vec<Vec<f32>> = vec![vec![0f32; audio_buf.num_frames()]; channels];
        let mut out_refs: Vec<&mut [f32]> = out_mut.iter_mut().map(|v| v.as_mut_slice()).collect();
        audio_buf.copy_to_stream(&stream_config, &mut out_refs);

        // interleave
        let mut out_refs_immut: Vec<&[f32]> = out_refs.iter().map(|r| &**r).collect();
        channels_to_interleaved(&mut out_refs_immut, &mut output);

        // Send to output; if channel is full/drop, skip
        let _ = tx_out.try_send(output.clone());
    }
}

fn main() -> Result<()> {
    let host = cpal::default_host();

    let input_device = host
        .default_input_device()
        .expect("No input device available");
    let output_device = host
        .default_output_device()
        .expect("No output device available");

    // Use 48 kHz (preferred) or fallback to the device default.
    let sample_rate_hz = 48_000;
    let channels = 2usize;
    let frames_per_buffer = (sample_rate_hz / 100) as usize; // 10 ms

    let (tx_in, rx_in) = bounded::<Vec<f32>>(16);
    let (tx_out, rx_out) = bounded::<Vec<f32>>(16);
    let (tx_render, rx_render) = bounded::<Vec<f32>>(16);

    // Spawn processing thread that owns the AEC instance.
    thread::spawn(move || processing_thread(rx_in, rx_render, tx_out, sample_rate_hz, channels));

    // Build input stream
    let in_config = cpal::StreamConfig {
        channels: channels as u16,
        sample_rate: sample_rate_hz as u32,
        buffer_size: cpal::BufferSize::Fixed(frames_per_buffer as u32),
    };

    let tx_in_clone = tx_in.clone();
    let input_stream = input_device.build_input_stream(
        in_config,
        move |data: &[f32], _| {
            // Copy data and send to processing thread
            let vec = data.to_vec();
            let _ = tx_in_clone.try_send(vec);
        },
        move |err| eprintln!("input stream error: {:?}", err),
        None,
    )?;

    // Build output stream
    let out_config = cpal::StreamConfig {
        channels: channels as u16,
        sample_rate: sample_rate_hz as u32,
        buffer_size: cpal::BufferSize::Fixed(frames_per_buffer as u32),
    };

    let tx_render_clone = tx_render.clone();
    let output_stream = output_device.build_output_stream(
        out_config,
        move |out: &mut [f32], _| {
            // Try to get processed frame; if none, output silence
            let frame_to_play = if let Ok(frame) = rx_out.try_recv() {
                let len = out.len().min(frame.len());
                out[..len].copy_from_slice(&frame[..len]);
                if len < out.len() {
                    for s in out[len..].iter_mut() {
                        *s = 0.0;
                    }
                }
                frame
            } else {
                for s in out.iter_mut() {
                    *s = 0.0;
                }
                vec![0.0f32; out.len()]
            };

            // Also forward the played frame to the AEC render path (non-blocking).
            let _ = tx_render_clone.try_send(frame_to_play);
        },
        move |err| eprintln!("output stream error: {:?}", err),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    println!("Running karaoke loop — press Ctrl+C to exit");

    // Wait until user terminates the example.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
