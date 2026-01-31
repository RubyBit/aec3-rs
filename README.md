[![crates.io](https://img.shields.io/crates/v/aec3.svg)](https://crates.io/crates/aec3)

aec3 — Rust port of WebRTC AEC3
================================

A small, pragmatic Rust port of WebRTC's **AEC3 acoustic echo canceller**.

This crate is for **real-time echo cancellation** in VoIP-style pipelines: it
uses far-end audio (the "render" / speaker signal) as a reference and removes
its echo from the near-end microphone capture.

It exposes:
- the full low-level AEC3 implementation in `crate::audio_processing::aec3`, and
- an ergonomic VoIP wrapper `crate::voip::VoipAec3` for typical "render + capture"
  streaming.

At a glance
-----------
- **Audio format:** interleaved `f32` frames
- **Frame size:** fixed **10 ms** frames (`frame_samples_per_channel * channels`)
- **Sample rates (wrapper input):** 16–48 kHz inclusive, including **44.1 kHz**
  (internally resampled to **48 kHz** for the AEC core)
- **AEC core full-band rates:** 16/32/48 kHz
- **Render/capture can be synchronous or asynchronous:** you can feed render and
  capture independently when they don’t arrive at the same time.

Key features
------------
- Implements the **AEC3 algorithm** aligned with the WebRTC reference pipeline.
- **Delay estimation + alignment** between render and capture.
- **Multi-band processing** (split/merge filter banks) + FFT-based analysis.
- Optional **capture high-pass filter** (enabled by default).
- Built-in **echo suppression** / residual echo control and comfort-noise logic
  (as part of the AEC3 pipeline).
- Small, dependency-light API intended for embedding in real-time apps.

Quick start (development)
-------------------------
1. Build and run the karaoke example (loopback + microphone). On Windows
   PowerShell:

```powershell
cargo run --example karaoke_loopback
```

2. Run the test-suite (unit + integration):

```powershell
cargo test
```

Using the VOIP wrapper
----------------------
The `VoipAec3` wrapper is the recommended way to integrate AEC3 into a
real-time pipeline. It handles conversion between interleaved frame buffers
and the internal multi-band audio buffers, applies an optional high-pass
filter, and exposes a small set of methods mirroring the reference demo.

Example (synchronous caller — render + capture available together):

```rust
use aec3::voip::VoipAec3;

let mut pipeline = VoipAec3::builder(48_000, 2, 2)
    .initial_delay_ms(116)
    .enable_high_pass(true)
    .build()
    .expect("failed to create pipeline");

// Per 10 ms captured frame (interleaved f32 samples):
let capture_frame: Vec<f32> = /* filled by your capture callback */;
let render_frame: Vec<f32> = /* optional far-end data */;
let mut out = vec![0.0f32; capture_frame.len()];

let metrics = pipeline.process(&capture_frame, Some(&render_frame), false, &mut out)?;
println!("AEC metrics: {:?}", metrics);
```

Example (asynchronous caller — render/capture arrive at different times)
-----------------------------------------------------------------------
In many real systems (especially on desktop), the far-end reference (loopback)
and microphone capture don’t arrive in lockstep. The wrapper supports this by
exposing separate methods:

- `handle_render_frame(&mut self, render_frame: &[f32])`
- `process_capture_frame(&mut self, capture_frame: &[f32], level_change: bool, out: &mut [f32])`

The key rule is:

- Feed **render** frames as soon as you get them.
- If you need to simulate or compensate device buffering latency, **delay the
  capture path**, not the render reference.

Minimal pattern (single processing thread with queues):

```rust
use aec3::voip::VoipAec3;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

let mut pipeline = VoipAec3::builder(44_100, 2, 1)
    .enable_high_pass(true)
    .build()
    .expect("failed to create pipeline");

// Optional: if you have an external estimate of device buffering delay.
pipeline.set_audio_buffer_delay(120);

// In real systems, this is unecessary or added to compensate for
// non causal buffering on the capture path <- this could be handled with ring buffers.
let target_capture_delay = Duration::from_millis(20);

// These queues are typically filled by your audio callbacks.
let mut pending_render: VecDeque<Vec<f32>> = VecDeque::new();
let mut pending_capture: VecDeque<(Instant, Vec<f32>)> = VecDeque::new();

loop {
    // 1) Drain render frames immediately to keep the reference current.
    while let Some(render_frame) = pending_render.pop_front() {
        pipeline.handle_render_frame(&render_frame).unwrap();
    }

    // 2) Process capture frames once they’ve aged past the target delay.
    while let Some((ts, _)) = pending_capture.front() {
        if ts.elapsed() < target_capture_delay {
            break;
        }

        let (_ts, capture_frame) = pending_capture.pop_front().unwrap();
        let mut out = vec![0.0f32; capture_frame.len()];

        let metrics = pipeline
            .process_capture_frame(&capture_frame, false, &mut out)
            .unwrap();

        // Send `out` to your encoder / stream, and optionally log metrics.
        let _ = metrics;
    }

    // In real code: block on your audio/event sources instead of busy looping.
}
```

Notes:
- The example uses **44.1 kHz** input: internally the canceller runs at **48 kHz**
  and resamples as needed.
- If you also have a render frame available at the same time as capture, prefer
  the combined `process(capture, Some(render), ...)` convenience method since it
  enforces the recommended "render first" ordering.


API summary
-----------
- VoipAec3::builder(sample_rate_hz, render_channels, capture_channels)
  - .with_config(EchoCanceller3Config) — supply custom config
  - .enable_high_pass(bool) — default true
  - .initial_delay_ms(i32) — optional external delay hint (ms)
  - .render_sample_rate_hz(usize) / .capture_sample_rate_hz(usize) — override rates
  - .build() -> Result<VoipAec3, VoipAec3Error>

- VoipAec3 methods
  - `capture_frame_samples()` — samples per channel per 10 ms capture frame
  - `render_frame_samples()` — samples per channel per 10 ms render frame
  - `sample_rate_hz()` — capture sample rate configured for the pipeline
  - `handle_render_frame(&mut self, render_frame: &[f32])` — feed far-end
  - `process_capture_frame(&mut self, capture_frame: &[f32], level_change: bool, out: &mut [f32]) -> Result<Metrics, Error>`
  - `process(&mut, capture_frame, Option<render_frame>, level_change, out)` — convenience
  - `set_audio_buffer_delay(&mut self, delay_ms: i32)` — update delay hint
  - `metrics(&self)` — get current metrics

Feature status / roadmap
------------------------

| Feature | Status | Notes |
|---|:---:|---|
| AEC3 core pipeline (render analysis + capture processing) | ✅ | `crate::audio_processing::aec3` |
| VoIP wrapper (`VoipAec3`) | ✅ | `crate::voip::VoipAec3` |
| 10 ms frame contract + validation | ✅ | checks render/capture frame sizes |
| Input sample rates 16–48 kHz | ✅ | inclusive, with special handling for 44.1 kHz |
| 44.1 kHz input support | ✅ | internally resampled to 48 kHz for the AEC core |
| Mixed render/capture sample rates | ✅ | render and capture can differ |
| Delay estimation + render/capture alignment | ✅ | built into the pipeline |
| Multi-band split/merge filterbanks + FFT analysis | ✅ | part of the AEC3 pipeline |
| Optional capture high-pass filter | ✅ | enabled by default |
| Metrics (ERL / ERLE / estimated delay) | ✅ | available via `metrics()` / return value |
| Diagnostics dumping | ✅ | available through the optional `diagnostics` feature |
| Expose “linear output” helpers in wrapper | 🚧 | core supports it; wrapper may expose later |
| Noise suppression (standalone NS) | Planned | out of scope today; may be added later |
| Automatic gain control (AGC) | Planned | out of scope today; may be added later |

Notes and integration tips
--------------------------
- Frame shape: the wrapper expects interleaved f32 frames sized as
-  `frame_samples_per_channel * channels`. `capture_frame_samples()` and
  `render_frame_samples()` return the per-channel length for 10 ms frames.
- Supported input sample rates are gated to 16–48 kHz. If you need other rates,
  resample before feeding frames to the wrapper.
- When you have both render and capture frames available at the same time,
  prefer calling `process(capture, Some(render), ...)` so the pipeline sees
  render first (consistent with the reference usage order).

Examples
--------
- `examples/karaoke_loopback.rs` — captures system loopback (render reference) +
  microphone and runs AEC in a processing thread.
- `examples/karaoke_loopback_delayed.rs` — simulates speaker-path latency by
  delaying **capture** frames and draining render frames separately. This is the
  recommended reference for integrating AEC when render/capture are asynchronous.

Contributing
------------
PRs welcome. Follow standard Rust contribution practices: ensure `cargo test`
passes and run `cargo fmt` before submitting.

Community projects
------------------
There are a few community-maintained projects that integrate with or wrap this
crate. For example:

- aec3-py — a community Python wrapper for this crate: https://github.com/RustedBytes/aec3-py

If you maintain a project that uses or wraps `aec3`, please open a PR to add it
here so others can find it easily.

License
-------
This repository is a port of code aligned with WebRTC reference algorithms.
Adopt and/or license in accordance with your needs and the original project
policy.
