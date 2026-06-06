use aec3::graph::{
    BuildCtx, Dependency, GraphBuilder, GraphError, GraphResult, InPort, InputOptions, MatchPolicy,
    NodeControlState, NodeFactory, NodeRunner, NodeSpec, OutPort, OutputOptions, Packet,
    PacketMeta, ProcessCtx, QueueConfig, Runtime, SchedulePlan, Timestamp,
};
use aec3::nodes::{
    aec3 as aec3_node, agc2, audio::AudioChunk, audio::AudioFormat, hpf, ns, resample, tap,
};
use aec3::pipelines::linear;

fn mono_format(sample_rate_hz: u32) -> AudioFormat {
    AudioFormat::ten_ms(sample_rate_hz, 1)
}

fn audio_packet(
    format: AudioFormat,
    sequence: u64,
    timestamp: Option<u64>,
    seed: f32,
) -> Packet<AudioChunk> {
    let mut samples = vec![0.0f32; format.sample_count()];
    for (index, sample) in samples.iter_mut().enumerate() {
        let t = index as f32 / 19.0;
        *sample = (t.sin() + seed.cos()) * 1000.0;
    }
    Packet {
        meta: PacketMeta {
            timestamp: timestamp.map(|start| Timestamp {
                clock: 0,
                start,
                duration: u32::from(format.frames_per_channel),
            }),
            sequence: Some(sequence),
            discontinuity: false,
        },
        payload: AudioChunk::from_interleaved(format, &samples),
    }
}

#[derive(Debug, Clone, Copy)]
struct OffsetNode {
    input: InPort<AudioChunk>,
    output: OutPort<AudioChunk>,
}

#[derive(Debug, Clone, Copy)]
struct OffsetNodeBuilder {
    format: AudioFormat,
    offset: f32,
}

struct OffsetFactory {
    input: InPort<AudioChunk>,
    output: OutPort<AudioChunk>,
    offset: f32,
}

struct OffsetRunner {
    input: InPort<AudioChunk>,
    output: OutPort<AudioChunk>,
    offset: f32,
}

impl NodeSpec for OffsetNodeBuilder {
    type Handles = OffsetNode;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        let node = graph.new_node("offset");
        let input = graph.register_input::<AudioChunk>(
            node,
            "audio_in",
            InputOptions {
                format_key: Some(self.format.schema_key()),
                ..InputOptions::default()
            },
        );
        let output = graph.register_output::<AudioChunk>(
            node,
            "audio_out",
            OutputOptions {
                format_key: Some(self.format.schema_key()),
            },
        );
        graph.finish_node(
            node,
            SchedulePlan::OnArrival {
                triggers: vec![input.raw()],
            },
            Box::new(OffsetFactory {
                input,
                output,
                offset: self.offset,
            }),
        )?;
        Ok(OffsetNode { input, output })
    }
}

impl NodeFactory for OffsetFactory {
    fn build(self: Box<Self>, _ctx: &mut BuildCtx) -> GraphResult<Box<dyn NodeRunner>> {
        Ok(Box::new(OffsetRunner {
            input: self.input,
            output: self.output,
            offset: self.offset,
        }))
    }
}

impl NodeRunner for OffsetRunner {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        let Some(mut packet) = ctx.take(self.input)? else {
            return Ok(());
        };
        packet.payload_mut().samples_mut()[0] += self.offset;
        ctx.emit_handle(self.output, packet)
    }
}

#[derive(Debug, Clone, Copy)]
struct GateNode {
    audio_in: InPort<AudioChunk>,
    gate_in: InPort<i32>,
    audio_out: OutPort<AudioChunk>,
}

#[derive(Debug, Clone, Copy)]
struct GateNodeBuilder {
    format: AudioFormat,
    match_policy: MatchPolicy,
    required: bool,
}

struct GateFactory {
    audio_in: InPort<AudioChunk>,
    gate_in: InPort<i32>,
    audio_out: OutPort<AudioChunk>,
}

struct GateRunner {
    audio_in: InPort<AudioChunk>,
    gate_in: InPort<i32>,
    audio_out: OutPort<AudioChunk>,
}

impl NodeSpec for GateNodeBuilder {
    type Handles = GateNode;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        let node = graph.new_node("gate");
        let audio_in = graph.register_input::<AudioChunk>(
            node,
            "audio_in",
            InputOptions {
                format_key: Some(self.format.schema_key()),
                ..InputOptions::default()
            },
        );
        let gate_in = graph.register_input::<i32>(node, "gate_in", InputOptions::default());
        let audio_out = graph.register_output::<AudioChunk>(
            node,
            "audio_out",
            OutputOptions {
                format_key: Some(self.format.schema_key()),
            },
        );
        graph.finish_node(
            node,
            SchedulePlan::AlignOn {
                trigger: audio_in.raw(),
                deps: vec![Dependency {
                    input: gate_in.raw(),
                    required: self.required,
                    match_policy: self.match_policy,
                }],
            },
            Box::new(GateFactory {
                audio_in,
                gate_in,
                audio_out,
            }),
        )?;
        Ok(GateNode {
            audio_in,
            gate_in,
            audio_out,
        })
    }
}

impl NodeFactory for GateFactory {
    fn build(self: Box<Self>, _ctx: &mut BuildCtx) -> GraphResult<Box<dyn NodeRunner>> {
        Ok(Box::new(GateRunner {
            audio_in: self.audio_in,
            gate_in: self.gate_in,
            audio_out: self.audio_out,
        }))
    }
}

impl NodeRunner for GateRunner {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        let Some(audio_packet) = ctx.take(self.audio_in)? else {
            return Ok(());
        };
        let Some(_gate_packet) = ctx.take(self.gate_in)? else {
            return Ok(());
        };
        ctx.emit_handle(self.audio_out, audio_packet)
    }
}

#[test]
fn graph_rejects_cycles() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let tap = tap::builder(format)
        .add_to(&mut graph)
        .expect("tap node should build");
    graph
        .connect(tap.audio_out, tap.audio_in)
        .expect("cycle edge should connect before validation");

    let err = graph.build().err().expect("cycle should be rejected");
    assert!(matches!(err, GraphError::GraphCycle));
}

#[test]
fn graph_rejects_audio_format_mismatch_without_adapter() {
    let render = mono_format(48_000);
    let capture = mono_format(48_000);
    let mut graph = GraphBuilder::new();
    let aec = aec3_node::builder(render, capture)
        .export_linear_output(true)
        .add_to(&mut graph)
        .expect("aec node should build");
    let hpf = hpf::builder(capture)
        .add_to(&mut graph)
        .expect("hpf node should build");

    let err = graph.connect(aec.linear_out.expect("linear output enabled"), hpf.audio_in);
    assert!(matches!(err, Err(GraphError::FormatMismatch { .. })));
}

#[test]
fn resample_node_adapts_between_formats() {
    let input_format = mono_format(48_000);
    let output_format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let source = graph.source::<AudioChunk>("source");
    let sink = graph.sink::<AudioChunk>("sink", QueueConfig::audio_default());
    let resample = resample::builder(input_format, output_format)
        .add_to(&mut graph)
        .expect("resample node should build");
    graph
        .connect(source, resample.audio_in)
        .expect("source should connect");
    graph
        .connect(resample.audio_out, sink)
        .expect("resample should connect to sink");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");
    runtime
        .push(source, audio_packet(input_format, 1, Some(1), 0.2))
        .expect("push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");

    let packet = runtime
        .try_pull(sink)
        .expect("pull should succeed")
        .expect("sink should receive one packet");
    assert_eq!(packet.payload().format, output_format);
    assert_eq!(
        packet.payload().samples().len(),
        output_format.sample_count()
    );
}

#[test]
fn align_on_waits_for_required_side_input() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let audio_source = graph.source::<AudioChunk>("audio");
    let gate_source = graph.source::<i32>("gate");
    let sink = graph.sink::<AudioChunk>("sink", QueueConfig::audio_default());
    let gate = graph
        .add_node(GateNodeBuilder {
            format,
            match_policy: MatchPolicy::BySequence,
            required: true,
        })
        .expect("gate node should register");
    graph
        .connect(audio_source, gate.audio_in)
        .expect("audio source should connect");
    graph
        .connect(gate_source, gate.gate_in)
        .expect("gate source should connect");
    graph
        .connect(gate.audio_out, sink)
        .expect("gate should connect to sink");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");
    runtime
        .push(audio_source, audio_packet(format, 7, Some(5), 0.3))
        .expect("audio push should succeed");
    runtime
        .run_until_stalled()
        .expect("graph should stall cleanly");
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_none(),
        "audio should wait until the required side input arrives"
    );

    // A stamped side packet with a *different* sequence must not match.
    runtime
        .push(
            gate_source,
            Packet {
                meta: PacketMeta {
                    timestamp: None,
                    sequence: Some(6),
                    discontinuity: false,
                },
                payload: 0,
            },
        )
        .expect("gate push should succeed");
    runtime
        .run_until_stalled()
        .expect("graph should stall cleanly");
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_none(),
        "a mismatched sequence must not align"
    );

    runtime
        .push(
            gate_source,
            Packet {
                meta: PacketMeta {
                    timestamp: None,
                    sequence: Some(7),
                    discontinuity: false,
                },
                payload: 1,
            },
        )
        .expect("gate push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_some()
    );
}

#[test]
fn by_sequence_alignment_rejects_unstamped_packets_on_required_inputs() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let audio_source = graph.source::<AudioChunk>("audio");
    let gate_source = graph.source::<i32>("gate");
    let sink = graph.sink::<AudioChunk>("sink", QueueConfig::audio_default());
    let gate = graph
        .add_node(GateNodeBuilder {
            format,
            match_policy: MatchPolicy::BySequence,
            required: true,
        })
        .expect("gate node should register");
    graph
        .connect(audio_source, gate.audio_in)
        .expect("audio source should connect");
    graph
        .connect(gate_source, gate.gate_in)
        .expect("gate source should connect");
    graph
        .connect(gate.audio_out, sink)
        .expect("gate should connect to sink");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");

    // Unstamped trigger on a required sequence-aligned dependency: hard error
    // instead of a silent FIFO fallback or an unbounded stall.
    let mut unstamped = audio_packet(format, 0, None, 0.3);
    unstamped.meta.sequence = None;
    runtime
        .push(audio_source, unstamped)
        .expect("audio push should succeed");
    let err = runtime
        .run_until_stalled()
        .expect_err("unstamped trigger should be rejected");
    assert!(matches!(
        err,
        GraphError::MissingSequenceForAlignment { .. }
    ));
}

#[test]
fn by_sequence_alignment_prunes_stale_dependency_packets() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let audio_source = graph.source::<AudioChunk>("audio");
    let gate_source = graph.source::<i32>("gate");
    let sink = graph.sink::<AudioChunk>("sink", QueueConfig::audio_default());
    let gate = graph
        .add_node(GateNodeBuilder {
            format,
            match_policy: MatchPolicy::BySequence,
            required: true,
        })
        .expect("gate node should register");
    graph
        .connect(audio_source, gate.audio_in)
        .expect("audio source should connect");
    graph
        .connect(gate_source, gate.gate_in)
        .expect("gate source should connect");
    graph
        .connect(gate.audio_out, sink)
        .expect("gate should connect to sink");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");

    runtime
        .push(audio_source, audio_packet(format, 100, None, 0.3))
        .expect("audio push should succeed");
    runtime
        .run_until_stalled()
        .expect("graph should stall cleanly");

    // Sequences are monotonic, so packets older than the pending trigger can
    // never match and must be pruned. Without pruning, the gate queue
    // (bounded(8), RejectPush) would reject pushes after eight mismatches.
    for stale_sequence in 1..=20u64 {
        runtime
            .push(
                gate_source,
                Packet {
                    meta: PacketMeta {
                        timestamp: None,
                        sequence: Some(stale_sequence),
                        discontinuity: false,
                    },
                    payload: 0,
                },
            )
            .expect("stale gate packets should be pruned, not accumulate");
        runtime
            .run_until_stalled()
            .expect("graph should stall cleanly");
    }
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_none(),
        "stale gate packets must not align"
    );

    runtime
        .push(
            gate_source,
            Packet {
                meta: PacketMeta {
                    timestamp: None,
                    sequence: Some(100),
                    discontinuity: false,
                },
                payload: 1,
            },
        )
        .expect("matching gate push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_some()
    );
}

#[test]
fn unmatched_optional_dependency_reads_nothing() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let audio_source = graph.source::<AudioChunk>("audio");
    let gate_source = graph.source::<i32>("gate");
    let sink = graph.sink::<AudioChunk>("sink", QueueConfig::audio_default());
    let gate = graph
        .add_node(GateNodeBuilder {
            format,
            match_policy: MatchPolicy::BySequence,
            required: false,
        })
        .expect("gate node should register");
    graph
        .connect(audio_source, gate.audio_in)
        .expect("audio source should connect");
    graph
        .connect(gate_source, gate.gate_in)
        .expect("gate source should connect");
    graph
        .connect(gate.audio_out, sink)
        .expect("gate should connect to sink");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");

    // Queue a gate packet for a *future* frame, then trigger with an earlier
    // frame. The optional dependency has no match, so the node must run but
    // `take(gate_in)` must return nothing instead of consuming the queued
    // future packet (GateRunner only emits when it reads a gate packet).
    runtime
        .push(
            gate_source,
            Packet {
                meta: PacketMeta {
                    timestamp: None,
                    sequence: Some(5),
                    discontinuity: false,
                },
                payload: 1,
            },
        )
        .expect("gate push should succeed");
    runtime
        .push(audio_source, audio_packet(format, 1, None, 0.3))
        .expect("audio push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_none(),
        "an unmatched optional dependency must not consume a wrong-frame packet"
    );

    // The frame-5 gate packet must still be available for the frame-5 trigger.
    runtime
        .push(audio_source, audio_packet(format, 5, None, 0.4))
        .expect("audio push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_some()
    );
}

#[test]
fn graph_rejects_zero_capacity_queues() {
    let mut graph = GraphBuilder::new();
    let source = graph.source::<AudioChunk>("source");
    let sink = graph.sink::<AudioChunk>("sink", QueueConfig::latest(0));
    graph.connect(source, sink).expect("edge should connect");

    let err = graph
        .build()
        .err()
        .expect("zero-capacity queue should be rejected");
    assert!(matches!(err, GraphError::InvalidQueueCapacity { .. }));
}

#[test]
fn aec3_pipeline_supports_async_render_capture_and_side_inputs() {
    let format = mono_format(48_000);
    let mut graph = GraphBuilder::new();
    let mic = graph.source::<AudioChunk>("mic");
    let render = graph.source::<AudioChunk>("render");
    let output = graph.sink::<AudioChunk>("output", QueueConfig::audio_default());
    let metrics_sink = graph.sink::<aec3_node::Aec3Metrics>("metrics", QueueConfig::latest(1));

    let custom = graph
        .add_node(OffsetNodeBuilder {
            format,
            offset: 0.5,
        })
        .expect("custom node should register");
    let aec = aec3_node::builder(format, format)
        .export_linear_output(true)
        .export_metrics(true)
        .add_to(&mut graph)
        .expect("aec node should build");
    let suppressor = ns::builder(format)
        .with_analysis_input(true)
        .add_to(&mut graph)
        .expect("ns node should build");

    graph
        .connect(mic, custom.input)
        .expect("mic should connect");
    graph
        .connect(custom.output, aec.capture_in)
        .expect("custom node should connect");
    graph
        .connect(render, aec.render_in)
        .expect("render should connect");
    graph
        .connect(aec.capture_out, suppressor.audio_in)
        .expect("capture path should connect");
    graph
        .connect(
            aec.linear_out.expect("linear output enabled"),
            suppressor.analysis_in.expect("analysis input enabled"),
        )
        .expect("linear output should connect");
    graph
        .connect(suppressor.audio_out, output)
        .expect("output should connect");
    graph
        .connect(aec.metrics_out.expect("metrics enabled"), metrics_sink)
        .expect("metrics should connect");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");

    runtime
        .push(render, audio_packet(format, 1, Some(10), 0.1))
        .expect("render push should succeed");
    runtime
        .run_until_stalled()
        .expect("render path should drain");
    assert!(
        runtime
            .try_pull(output)
            .expect("pull should succeed")
            .is_none(),
        "render analysis alone should not produce capture output"
    );

    runtime
        .push(mic, audio_packet(format, 2, Some(10), 0.4))
        .expect("capture push should succeed");
    runtime
        .run_until_stalled()
        .expect("capture path should drain");

    let audio = runtime
        .try_pull(output)
        .expect("pull should succeed")
        .expect("capture output should be emitted");
    let metrics = runtime
        .try_pull(metrics_sink)
        .expect("metrics pull should succeed")
        .expect("metrics should be emitted");

    assert_eq!(audio.payload().format, format);
    assert_eq!(audio.payload().samples().len(), format.sample_count());
    assert!(metrics.payload().delay_ms >= 0);
}

#[test]
fn capture_only_agc_pipeline_runs_without_render() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let audio = graph.source::<AudioChunk>("audio");
    let input_volume = graph.source::<i32>("input_volume");
    let audio_out = graph.sink::<AudioChunk>("audio_out", QueueConfig::audio_default());
    let recommended_out = graph.sink::<i32>("recommended", QueueConfig::latest(1));
    let agc = agc2::builder(format)
        .add_to(&mut graph)
        .expect("agc node should build");

    graph
        .connect(audio, agc.audio_in)
        .expect("audio should connect");
    graph
        .connect(
            input_volume,
            agc.applied_input_volume_in.expect("volume input enabled"),
        )
        .expect("control source should connect");
    graph
        .connect(agc.audio_out, audio_out)
        .expect("audio output should connect");
    graph
        .connect(
            agc.recommended_input_volume_out
                .expect("recommended output enabled"),
            recommended_out,
        )
        .expect("recommended sink should connect");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");
    runtime
        .push(
            input_volume,
            Packet {
                meta: PacketMeta::default(),
                payload: 128,
            },
        )
        .expect("input volume push should succeed");
    runtime
        .run_until_stalled()
        .expect("control path should drain");

    for frame in 0..12 {
        runtime
            .push(audio, audio_packet(format, frame, Some(frame), 0.7))
            .expect("audio push should succeed");
        runtime.run_until_stalled().expect("agc should drain");
        let _ = runtime
            .try_pull(audio_out)
            .expect("audio pull should succeed");
    }

    assert!(
        runtime
            .try_pull(recommended_out)
            .expect("pull should succeed")
            .is_some(),
        "AGC2 should eventually emit a recommended input volume"
    );
}

#[test]
fn tap_node_fans_out_to_multiple_outputs() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let source = graph.source::<AudioChunk>("source");
    let sink_a = graph.sink::<AudioChunk>("sink_a", QueueConfig::audio_default());
    let sink_b = graph.sink::<AudioChunk>("sink_b", QueueConfig::audio_default());
    let tap = tap::builder(format)
        .add_to(&mut graph)
        .expect("tap node should build");

    graph
        .connect(source, tap.audio_in)
        .expect("source should connect");
    graph
        .connect(tap.audio_out, sink_a)
        .expect("tap audio output should connect");
    graph
        .connect(tap.tap_out, sink_b)
        .expect("tap output should connect");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");
    runtime
        .push(source, audio_packet(format, 1, Some(1), 1.2))
        .expect("push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");

    let a = runtime
        .try_pull(sink_a)
        .expect("pull should succeed")
        .expect("sink a should receive one packet");
    let b = runtime
        .try_pull(sink_b)
        .expect("pull should succeed")
        .expect("sink b should receive one packet");

    assert_eq!(a.payload().samples(), b.payload().samples());
}

#[test]
fn align_on_fifo_policy_matches_queue_order_without_metadata() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let source = graph.source::<AudioChunk>("source");
    let gate_source = graph.source::<i32>("gate");
    let sink = graph.sink::<AudioChunk>("sink", QueueConfig::audio_default());
    let gate = graph
        .add_node(GateNodeBuilder {
            format,
            match_policy: MatchPolicy::Fifo,
            required: true,
        })
        .expect("gate node should register");
    graph
        .connect(source, gate.audio_in)
        .expect("source should connect");
    graph
        .connect(gate_source, gate.gate_in)
        .expect("gate source should connect");
    graph
        .connect(gate.audio_out, sink)
        .expect("sink should connect");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");
    runtime
        .push(
            gate_source,
            Packet {
                meta: PacketMeta::default(),
                payload: 1,
            },
        )
        .expect("gate control should succeed");
    runtime
        .push(source, audio_packet(format, 1, None, 0.9))
        .expect("push should succeed");
    runtime
        .run_until_stalled()
        .expect("fifo alignment should match queue order without metadata");
    assert!(
        runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .is_some()
    );
}

#[test]
fn built_in_nodes_support_bypass_and_suspend_states() {
    let format = mono_format(16_000);
    let mut graph = GraphBuilder::new();
    let source = graph.source::<AudioChunk>("source");
    let sink_a = graph.sink::<AudioChunk>("sink_a", QueueConfig::audio_default());
    let sink_b = graph.sink::<AudioChunk>("sink_b", QueueConfig::audio_default());
    let tap_node = tap::builder(format)
        .add_to(&mut graph)
        .expect("tap node should build");

    graph
        .connect(source, tap_node.audio_in)
        .expect("source should connect");
    graph
        .connect(tap_node.audio_out, sink_a)
        .expect("audio sink should connect");
    graph
        .connect(tap_node.tap_out, sink_b)
        .expect("tap sink should connect");

    let spec = graph.build().expect("graph should build");
    let mut runtime = Runtime::new(spec).expect("runtime should build");

    runtime
        .set_node_state(tap_node.node_id(), NodeControlState::Bypassed)
        .expect("state change should succeed");
    runtime
        .push(source, audio_packet(format, 1, Some(1), 0.5))
        .expect("push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");
    assert!(
        runtime
            .try_pull(sink_a)
            .expect("pull should succeed")
            .is_some()
    );
    assert!(
        runtime
            .try_pull(sink_b)
            .expect("pull should succeed")
            .is_none()
    );

    runtime
        .set_node_state(tap_node.node_id(), NodeControlState::Suspended)
        .expect("state change should succeed");
    runtime
        .push(source, audio_packet(format, 2, Some(2), 0.8))
        .expect("push should succeed");
    runtime.run_until_stalled().expect("runtime should drain");
    assert!(
        runtime
            .try_pull(sink_a)
            .expect("pull should succeed")
            .is_none()
    );
    assert!(
        runtime
            .try_pull(sink_b)
            .expect("pull should succeed")
            .is_none()
    );
}

#[test]
fn linear_pipeline_builder_is_ergonomic_and_resettable() {
    let format = mono_format(48_000);
    let mut pipeline = linear::builder(format, format)
        .export_metrics(true)
        .export_linear_output(true)
        .build()
        .expect("pipeline should build");

    let render = vec![0.0f32; format.sample_count()];
    let capture = vec![0.0f32; format.sample_count()];
    let mut output = vec![0.0f32; format.sample_count()];

    pipeline
        .handle_render_frame(&render)
        .expect("render should be accepted");
    let produced = pipeline
        .process_capture_frame(&capture, &mut output)
        .expect("capture should be processed");
    assert!(produced, "linear pipeline should emit capture output");
    assert!(
        pipeline
            .try_pull_metrics()
            .expect("metrics pull should succeed")
            .is_some()
    );
    assert!(
        pipeline
            .try_pull_linear_output()
            .expect("linear pull should succeed")
            .is_some()
    );

    pipeline.reset_aec3().expect("aec reset should succeed");
    pipeline
        .set_node_state(
            pipeline.handles().aec3.node_id(),
            NodeControlState::Suspended,
        )
        .expect("state change should succeed");
    let produced = pipeline
        .process_capture_frame(&capture, &mut output)
        .expect("suspended capture should not fail");
    assert!(!produced, "suspended pipeline should not emit output");
}
