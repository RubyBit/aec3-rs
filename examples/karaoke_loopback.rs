use aec3::voip::VoipAec3;
use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, bounded};
use std::collections::VecDeque;
use std::thread;
use std::time::{Duration, Instant};

fn processing_thread(
    rx_in: Receiver<Vec<f32>>,
    rx_render: Receiver<Vec<f32>>,
    tx_out: Sender<Vec<f32>>,
    sample_rate: usize,
    channels: usize,
) {
    let mut pipeline = VoipAec3::builder(sample_rate, channels, channels)
        .initial_delay_ms(116)
        .build()
        .expect("failed to create AEC pipeline");

    let mut last_metrics = Instant::now();
    let metrics_interval = Duration::from_secs(5);

    while let Ok(frame) = rx_in.recv() {
        let render_frame = rx_render.try_recv().ok();
        let mut output = vec![0f32; frame.len()];
        match pipeline.process(&frame, render_frame.as_deref(), false, &mut output) {
            Ok(metrics) => {
                if last_metrics.elapsed() >= metrics_interval {
                    println!("AEC metrics: {:?}", metrics);
                    last_metrics = Instant::now();
                }
                let _ = tx_out.try_send(output);
            }
            Err(err) => {
                eprintln!("AEC processing error: {err}");
            }
        }
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

    // We'll prefer the system default output sample rate for consistent loopback capture.
    let loopback_supported = output_device.default_output_config()?;
    let loopback_stream_config: cpal::StreamConfig = loopback_supported.clone().into();
    let sample_rate_hz = loopback_stream_config.sample_rate.0 as usize;
    let channels = loopback_stream_config.channels as usize;
    let frames_per_buffer = (sample_rate_hz / 100) as usize; // 10 ms

    let (tx_in, rx_in) = bounded::<Vec<f32>>(16);
    let (tx_out, rx_out) = bounded::<Vec<f32>>(16);
    let (tx_render, rx_render) = bounded::<Vec<f32>>(16);

    // Spawn processing thread that owns the AEC instance.
    thread::spawn(move || processing_thread(rx_in, rx_render, tx_out, sample_rate_hz, channels));

    // Build microphone input stream using the same sample rate and channels as loopback.
    let in_config = cpal::StreamConfig {
        channels: channels as u16,
        sample_rate: cpal::SampleRate(sample_rate_hz as u32),
        buffer_size: cpal::BufferSize::Fixed(frames_per_buffer as u32),
    };

    let tx_in_clone = tx_in.clone();
    let input_stream = input_device.build_input_stream(
        &in_config,
        move |data: &[f32], _| {
            // Copy data and send to processing thread
            let vec = data.to_vec();
            let _ = tx_in_clone.try_send(vec);
        },
        move |err| eprintln!("input stream error: {:?}", err),
        None,
    )?;

    // Build a loopback input stream by opening an input stream on the output device using its default_output_config.
    // This captures what is being played on the system output (loopback).
    let tx_render_clone = tx_render.clone();
    let loopback_input_stream = output_device.build_input_stream(
        &loopback_stream_config,
        move |data: &[f32], _| {
            // forward the loopback (render) frames into the AEC render path
            let vec = data.to_vec();
            let _ = tx_render_clone.try_send(vec);
        },
        move |err| eprintln!("loopback input stream error: {:?}", err),
        None,
    )?;

    // Build output stream (play processed audio)
    let out_config = cpal::StreamConfig {
        channels: channels as u16,
        sample_rate: cpal::SampleRate(sample_rate_hz as u32),
        buffer_size: cpal::BufferSize::Fixed(frames_per_buffer as u32),
    };

    // Implement a playback delay queue to delay playback by ~3 seconds.
    let mut delay_queue: VecDeque<Vec<f32>> = VecDeque::new();
    let delay_buffers = std::cmp::max(1usize, (sample_rate_hz * 3) / frames_per_buffer);

    let output_stream = output_device.build_output_stream(
        &out_config,
        move |out: &mut [f32], _| {
            // Drain any available processed frames into the delay queue (non-blocking)
            while let Ok(frame) = rx_out.try_recv() {
                delay_queue.push_back(frame);
            }

            // If we've accumulated more than the target delay buffers, play the oldest one.
            if delay_queue.len() > delay_buffers {
                if let Some(frame_to_play) = delay_queue.pop_front() {
                    let len = out.len().min(frame_to_play.len());
                    out[..len].copy_from_slice(&frame_to_play[..len]);
                    if len < out.len() {
                        for s in out[len..].iter_mut() {
                            *s = 0.0;
                        }
                    }
                } else {
                    for s in out.iter_mut() {
                        *s = 0.0;
                    }
                }
            } else {
                // Not enough buffered yet, output silence to preserve the delay
                for s in out.iter_mut() {
                    *s = 0.0;
                }
            }
        },
        move |err| eprintln!("output stream error: {:?}", err),
        None,
    )?;

    input_stream.play()?;
    loopback_input_stream.play()?;
    output_stream.play()?;

    println!("Running karaoke loopback — press Ctrl+C to exit");

    // Wait until user terminates the example.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
