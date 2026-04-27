use ::aec3::nodes::audio::AudioFormat;
use ::aec3::pipelines::linear;
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, bounded};
use std::collections::VecDeque;
use std::thread;
use std::time::{Duration, Instant};

fn processing_thread(
    rx_capture: Receiver<Vec<f32>>,
    rx_render: Receiver<Vec<f32>>,
    tx_output: Sender<Vec<f32>>,
    sample_rate_hz: u32,
    channels: u16,
) {
    let format = AudioFormat::ten_ms(sample_rate_hz, channels);
    let mut pipeline = linear::builder(format, format)
        .initial_delay_ms(116)
        .export_metrics(true)
        .build()
        .expect("failed to build linear voice pipeline");

    let mut last_metrics = Instant::now();
    let metrics_interval = Duration::from_secs(5);

    while let Ok(frame) = rx_capture.recv() {
        while let Ok(render_frame) = rx_render.try_recv() {
            if let Err(err) = pipeline.handle_render_frame(&render_frame) {
                eprintln!("render push error: {err}");
            }
        }

        let mut output = vec![0.0f32; frame.len()];
        match pipeline.process_capture_frame(&frame, &mut output) {
            Ok(true) => {
                while let Ok(Some(packet)) = pipeline.try_pull_metrics() {
                    if last_metrics.elapsed() >= metrics_interval {
                        println!("AEC metrics: {:?}", packet.payload());
                        last_metrics = Instant::now();
                    }
                }
                let _ = tx_output.try_send(output);
            }
            Ok(false) => {}
            Err(err) => eprintln!("capture processing error: {err}"),
        }
    }
}

fn main() -> Result<()> {
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("no input device available");
    let output_device = host
        .default_output_device()
        .expect("no output device available");

    let loopback_supported = output_device.default_output_config()?;
    let loopback_stream_config: cpal::StreamConfig = loopback_supported.clone().into();
    let sample_rate_hz = loopback_stream_config.sample_rate.0;
    let channels = loopback_stream_config.channels;
    let frames_per_buffer = (sample_rate_hz / 100) as usize;

    let (tx_capture, rx_capture) = bounded::<Vec<f32>>(16);
    let (tx_render, rx_render) = bounded::<Vec<f32>>(16);
    let (tx_output, rx_output) = bounded::<Vec<f32>>(16);

    thread::spawn(move || {
        processing_thread(rx_capture, rx_render, tx_output, sample_rate_hz, channels)
    });

    let in_config = cpal::StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate_hz),
        buffer_size: cpal::BufferSize::Fixed(frames_per_buffer as u32),
    };
    let out_config = cpal::StreamConfig {
        channels,
        sample_rate: cpal::SampleRate(sample_rate_hz),
        buffer_size: cpal::BufferSize::Fixed(frames_per_buffer as u32),
    };

    let input_stream = input_device.build_input_stream(
        &in_config,
        move |data: &[f32], _| {
            let _ = tx_capture.try_send(data.to_vec());
        },
        move |err| eprintln!("input stream error: {err:?}"),
        None,
    )?;

    let loopback_stream = output_device.build_input_stream(
        &loopback_stream_config,
        move |data: &[f32], _| {
            let _ = tx_render.try_send(data.to_vec());
        },
        move |err| eprintln!("loopback stream error: {err:?}"),
        None,
    )?;

    let mut delay_queue: VecDeque<Vec<f32>> = VecDeque::new();
    let delay_buffers = std::cmp::max(1usize, (sample_rate_hz as usize * 3) / frames_per_buffer);
    let output_stream = output_device.build_output_stream(
        &out_config,
        move |out: &mut [f32], _| {
            while let Ok(frame) = rx_output.try_recv() {
                delay_queue.push_back(frame);
            }

            if delay_queue.len() > delay_buffers {
                if let Some(frame_to_play) = delay_queue.pop_front() {
                    let len = out.len().min(frame_to_play.len());
                    out[..len].copy_from_slice(&frame_to_play[..len]);
                    for sample in &mut out[len..] {
                        *sample = 0.0;
                    }
                } else {
                    out.fill(0.0);
                }
            } else {
                out.fill(0.0);
            }
        },
        move |err| eprintln!("output stream error: {err:?}"),
        None,
    )?;

    input_stream.play()?;
    loopback_stream.play()?;
    output_stream.play()?;

    println!("Running karaoke loopback with the linear voice pipeline. Press Ctrl+C to exit.");

    loop {
        thread::sleep(Duration::from_millis(100));
    }
}
