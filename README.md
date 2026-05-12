[![crates.io](https://img.shields.io/crates/v/aec3.svg)](https://crates.io/crates/aec3)

aec3 - Rust port of WebRTC AEC3
===============================

`aec3` is a Rust port of WebRTC's AEC3 acoustic echo canceller plus a growing
set of reusable DSP building blocks. In `0.2`, the crate moves away from the
old special-purpose `voip` wrapper and exposes a generic event-driven DAG:

- `aec3::graph` is the runtime, scheduling, ports, packets, and validation layer
- `aec3::nodes` contains built-in audio/DSP nodes such as AEC3, NS, AGC2, HPF,
  resampling, and taps
- `aec3::pipelines` adds ergonomic builders on top of the graph for common layouts

This lets you model capture-only paths, duplex AEC paths, side-channel analysis
links, multi-output graphs, and custom user nodes with one execution model. This will also allow more flexible scheduling and optimization opportunities in the future such as running nodes at different rates, dynamic reconfiguration, and more efficient fan-out patterns + easier to integrate custom processing on your pipelines. 

NOTE: This is a work in progress and the API is expected to evolve. Feedback and contributions are very welcome, specially in terms of ergonomics and use cases but also performance (i.e I am still validating internally if this design is useful). You still utilize the processing modules by themselves in `aec3::audio_processing` if you want to avoid the graph API for now or have no use for it.

Highlights
----------

- Generic typed graph builder with `Source<T>`, `Sink<T>`, `InPort<T>`, and `OutPort<T>`
- Asynchronous stream arrival: render, capture, and control packets can arrive independently
- AEC3 as an ordinary node with optional linear-output, metrics, diagnostics, and delay-control ports
- Side inputs for nodes like noise suppression without hardcoding one pipeline shape
- Shared packet handles and copy-on-write audio buffers to minimize copying on fan-out paths
- Runtime node control states and resets for bypass/freeze/reinitialize workflows
- `aec3::pipelines::linear` for the common `render + capture -> HFP -> AEC3 -> NS -> AGC2` path
- Strong typing for ordinary wiring, plus runtime validation for graph invariants and format mismatches

Quick start
-----------

Run the examples:

```powershell
cargo run --example file_to_file -- render.wav capture.wav output.wav
cargo run --example karaoke_loopback
cargo run --example karaoke_loopback_delayed
```

Run the test suite:

```powershell
cargo test
```

Core model
----------

The crate is organized around three top-level modules:

- `aec3::graph`
  - graph builder
  - typed ports and packets
  - queueing, scheduling, and backpressure
  - packet timestamps and alignment rules
- `aec3::nodes`
  - `audio`: `AudioFormat`, `AudioChunk`, pooled audio storage
  - `aec3`: echo cancellation node
  - `agc2`: gain control node
  - `ns`: noise suppression node
  - `hpf`: high-pass filter node
  - `resample`: explicit sample-rate / channel adaptation
  - `tap`: packet fan-out without forcing eager copies
- `aec3::pipelines`
  - `linear`: convenience builder/runtime wrapper for the most common voice chain
- `aec3::audio_processing`
  - low-level processing modules ported from WebRTC (e.g. `aec3::audio_processing::aec3::echo_canceller3`, `aec3::audio_processing::gain_controller2`, `aec3::audio_processing::ns::noise_suppressor`)

All built-in DSP nodes operate on 10 ms audio frames carried as `AudioChunk`.

Common voice pipeline
---------------------

If you just want the standard voice path, start with `aec3::pipelines::linear`:

```rust
use aec3::nodes::audio::AudioFormat;
use aec3::pipelines::linear;

let format = AudioFormat::ten_ms(48_000, 1);
let mut pipeline = linear::builder(format, format)
    .initial_delay_ms(116)
    .export_metrics(true)
    .build()?;

let render = vec![0.0f32; format.sample_count()];
let capture = vec![0.0f32; format.sample_count()];
let mut output = vec![0.0f32; format.sample_count()];

pipeline.handle_render_frame(&render)?;
let produced = pipeline.process_capture_frame(&capture, &mut output)?;
assert!(produced);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`linear::builder(...).add_to(&mut GraphBuilder)` is also available when you
want the convenience layout but still plan to attach extra outputs manually.

Building a graph
----------------

```rust
use ::aec3::graph::{GraphBuilder, Packet, PacketMeta, QueueConfig, Runtime, RuntimeOptions};
use ::aec3::nodes::{
    aec3 as aec3_node,
    agc2,
    audio::{AudioChunk, AudioFormat},
    ns,
};

let format = AudioFormat::ten_ms(48_000, 1);

let mut graph = GraphBuilder::new();
let mic = graph.source::<AudioChunk>("mic", QueueConfig::audio_default());
let render = graph.source::<AudioChunk>("render", QueueConfig::audio_default());
let output = graph.sink::<AudioChunk>("output", QueueConfig::audio_default());

let agc_pre = agc2::builder(format).add_to(&mut graph)?;
let echo = aec3_node::builder(format, format)
    .export_linear_output(true)
    .export_metrics(true)
    .add_to(&mut graph)?;
let suppressor = ns::builder(format)
    .with_analysis_input(true)
    .add_to(&mut graph)?;

graph.connect(mic, agc_pre.audio_in)?;
graph.connect(agc_pre.audio_out, echo.capture_in)?;
graph.connect(render, echo.render_in)?;
graph.connect(echo.capture_out, suppressor.audio_in)?;
graph.connect(
    echo.linear_out.unwrap(),
    suppressor.analysis_in.unwrap(),
)?;
graph.connect(suppressor.audio_out, output)?;

let spec = graph.build()?;
let mut runtime = Runtime::new(spec, RuntimeOptions)?;

runtime.push(
    render,
    Packet {
        meta: PacketMeta::default(),
        payload: AudioChunk::silence(format),
    },
)?;
runtime.run_until_stalled()?;

runtime.push(
    mic,
    Packet {
        meta: PacketMeta::default(),
        payload: AudioChunk::silence(format),
    },
)?;
runtime.run_until_stalled()?;

if let Some(packet) = runtime.try_pull(output)? {
    println!("processed {} samples", packet.payload().samples().len());
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Notes:

- Render and capture do not need to arrive in lockstep.
- Nodes run when their scheduling policy says their dependencies are satisfied.
- `Runtime::try_pull` returns a shared `PacketHandle<T>` so one upstream packet can fan out cheaply.

Scheduling and async arrival
----------------------------

Built-in nodes use two scheduling styles:

- `SchedulePlan::OnArrival`
  - run whenever a trigger input receives a packet
  - used by nodes like AEC3 where render updates internal state independently of capture output
- `SchedulePlan::AlignOn`
  - run only when a trigger packet can be matched with dependency packets under a `MatchPolicy`
  - used for side-input patterns such as optional analysis audio

`PacketMeta` carries optional timestamps:

```rust
use aec3::graph::{PacketMeta, Timestamp};

let meta = PacketMeta {
    timestamp: Some(Timestamp {
        clock: 0,
        start: 48_000,
        duration: 480,
    }),
    sequence: 7,
    discontinuity: false,
};
```

Alignment rules only compare packets on the same clock. Cross-clock joins need
an explicit adaptation node. When timestamps are absent, `AlignOn` falls back
to FIFO/latest queue behavior instead of erroring.

Custom nodes
------------

You can insert your own nodes anywhere in the graph by implementing:

- `NodeSpec` to register typed ports and return handles
- `NodeFactory` to build runtime state
- `NodeRunner` to consume inputs and emit outputs from `ProcessCtx`

That keeps the graph core generic while letting node implementations own their
own state, readiness rules, and processing logic.

Node lifecycle control
----------------------

The runtime exposes per-node control states:

- `Active`: process normally
- `Bypassed`: pass through the primary audio path when the node supports it
- `Suspended`: freeze/drop work without changing topology

Built-in nodes also implement `reset()` through `Runtime::reset_node(...)`, and
the linear pipeline wrapper exposes convenience helpers like `reset_aec3()`.

Built-in node patterns
----------------------

- Capture-only processing: `source -> agc2 / hpf / ns -> sink`
- Duplex echo cancellation: `capture + render -> aec3 -> sink`
- Side-channel analysis: `aec3.linear_out -> ns.analysis_in`
- Common voice chain: `pipelines::linear::builder(render, capture)`
- Explicit format adaptation: insert `nodes::resample`
- Fan-out: insert `nodes::tap` or connect one output to multiple downstream ports


Examples
--------

- `examples/karaoke_loopback.rs`: live loopback + microphone processing with `pipelines::linear`
- `examples/karaoke_loopback_delayed.rs`: same setup with an intentionally delayed capture path
- `examples/file_to_file.rs`: minimal offline WAV render + capture -> processed WAV example

Contributing
------------

PRs welcome. Run `cargo fmt` and `cargo test` before submitting changes.

License
-------

This repository is a port of code aligned with WebRTC reference algorithms.
Adopt and/or license in accordance with your needs and the original project
policy.
