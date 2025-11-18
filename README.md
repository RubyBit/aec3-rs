aec3 — Rust port of WebRTC AEC3
================================

A small, pragmatic Rust port of WebRTC's AEC3 acoustic-echo-canceller. This
crate exposes the full audio-processing implementation (in
`crate::audio_processing::aec3`) and provides a small, ergonomic VOIP-style
wrapper in `crate::voip::VoipAec3` for easy integration into real-time
pipelines.

What you'll find in this repository
-----------------------------------
- `src/audio_processing` — low-level audio primitives (buffers, filters, FFT,
  and the AEC3 pipeline implementation).
- `src/voip/mod.rs` — a small wrapper that hides the plumbing and provides a
  convenient builder-based API for VOIP use-cases.
- `examples/karaoke_loopback.rs` — example that captures system loopback + mic
  and runs AEC in a processing thread.
- `tests/voip.rs` — unit tests that exercise the wrapper API.

Key features
------------
- Implements the AEC3 algorithm compatible with WebRTC reference behavior.
- Supported full-band sample rates: 16 kHz, 32 kHz, 48 kHz.
- Builder-style wrapper for simple integration: optional HPF, initial delay
  hint, custom `EchoCanceller3Config`.
- Small, dependency-light API intended for embedding in VoIP apps.

Quick start (development)
-------------------------
1. Build and run the karaoke example (loopback + microphone). On Windows
   PowerShell:

```powershell
cd C:/Users/nikos/Desktop/Projects/aec3
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

Example (synchronous caller):

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

API summary
-----------
- VoipAec3::builder(sample_rate_hz, render_channels, capture_channels)
  - .with_config(EchoCanceller3Config) — supply custom config
  - .enable_high_pass(bool) — default true
  - .initial_delay_ms(i32) — optional external delay hint (ms)
  - .build() -> Result<VoipAec3, VoipAec3Error>

- VoipAec3 methods
  - `frame_samples()` — samples per channel per 10 ms frame
  - `sample_rate_hz()` — configured sample rate
  - `handle_render_frame(&mut self, render_frame: &[f32])` — feed far-end
  - `process_capture_frame(&mut self, capture_frame: &[f32], level_change: bool, out: &mut [f32]) -> Result<Metrics, Error>`
  - `process(&mut, capture_frame, Option<render_frame>, level_change, out)` — convenience
  - `set_audio_buffer_delay(&mut self, delay_ms: i32)` — update delay hint
  - `metrics(&self)` — get current metrics

Notes and integration tips
--------------------------
- Frame shape: the wrapper expects interleaved f32 frames sized as
  `frame_samples * channels`. `frame_samples()` returns the per-channel
  length for 10 ms frames (e.g. 480 for 48 kHz, 160 for 16 kHz).
- Supported sample rates are intentionally gated to the full-band rates used
  by the reference (16/32/48 kHz). If you need other rates, resample before
  feeding frames to the wrapper.
- When you have both render and capture frames available at the same time,
  prefer calling `process(capture, Some(render), ...)` so the pipeline sees
  render first (consistent with the reference usage order).

Contributing
------------
PRs welcome. Follow standard Rust contribution practices: ensure `cargo test`
passes and run `cargo fmt` before submitting.

License
-------
This repository is a port of code aligned with WebRTC reference algorithms.
Adopt and/or license in accordance with your needs and the original project
policy.
