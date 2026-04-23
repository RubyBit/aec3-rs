use crate::audio_processing::ns::{NoiseSuppressor, NsConfig};
use crate::graph::{
    Dependency, GraphBuilder, GraphResult, InputOptions, MatchPolicy, NodeControlState,
    NodeFactory, NodeId, NodeRunner, NodeSpec, OutPort, OutputOptions, ProcessCtx, SchedulePlan,
};

use super::audio::{AudioChunk, AudioFormat};
use super::util::{ChunkIo, internal_sample_rate, validate_audio_format};

#[derive(Debug, Clone, Copy)]
pub struct NoiseSuppressorNode {
    node_id: NodeId,
    pub audio_in: crate::graph::InPort<AudioChunk>,
    pub analysis_in: Option<crate::graph::InPort<AudioChunk>>,
    pub audio_out: OutPort<AudioChunk>,
}

#[derive(Debug, Clone, Copy)]
pub struct NoiseSuppressorNodeBuilder {
    format: AudioFormat,
    with_analysis_input: bool,
    config: NsConfig,
}

pub fn builder(format: AudioFormat) -> NoiseSuppressorNodeBuilder {
    NoiseSuppressorNodeBuilder {
        format,
        with_analysis_input: false,
        config: NsConfig::default(),
    }
}

impl NoiseSuppressorNodeBuilder {
    pub fn with_analysis_input(mut self, enable: bool) -> Self {
        self.with_analysis_input = enable;
        self
    }

    pub fn config(mut self, config: NsConfig) -> Self {
        self.config = config;
        self
    }

    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<NoiseSuppressorNode> {
        graph.add_node(self)
    }
}

impl NoiseSuppressorNode {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

impl NodeSpec for NoiseSuppressorNodeBuilder {
    type Handles = NoiseSuppressorNode;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        validate_audio_format(self.format, "NoiseSuppressorNode")?;

        let analysis_format = AudioFormat::ten_ms(16_000, self.format.channels);
        let node = graph.new_node("noise_suppressor");
        let audio_in = graph.register_input::<AudioChunk>(
            node,
            "audio_in",
            InputOptions {
                format_key: Some(self.format.schema_key()),
                ..InputOptions::default()
            },
        );
        let analysis_in = self.with_analysis_input.then(|| {
            graph.register_input::<AudioChunk>(
                node,
                "analysis_in",
                InputOptions {
                    format_key: Some(analysis_format.schema_key()),
                    ..InputOptions::default()
                },
            )
        });
        let audio_out = graph.register_output::<AudioChunk>(
            node,
            "audio_out",
            OutputOptions {
                format_key: Some(self.format.schema_key()),
            },
        );

        let schedule = if let Some(analysis_in) = analysis_in {
            SchedulePlan::AlignOn {
                trigger: audio_in.raw(),
                deps: vec![Dependency {
                    input: analysis_in.raw(),
                    required: false,
                    match_policy: MatchPolicy::ExactTimestamp,
                }],
            }
        } else {
            SchedulePlan::OnArrival {
                triggers: vec![audio_in.raw()],
            }
        };

        graph.finish_node(
            node,
            schedule,
            Box::new(NoiseSuppressorFactory {
                format: self.format,
                analysis_format,
                config: self.config,
                audio_in,
                analysis_in,
                audio_out,
            }),
        )?;

        Ok(NoiseSuppressorNode {
            node_id: node,
            audio_in,
            analysis_in,
            audio_out,
        })
    }
}

struct NoiseSuppressorFactory {
    format: AudioFormat,
    analysis_format: AudioFormat,
    config: NsConfig,
    audio_in: crate::graph::InPort<AudioChunk>,
    analysis_in: Option<crate::graph::InPort<AudioChunk>>,
    audio_out: OutPort<AudioChunk>,
}

impl NodeFactory for NoiseSuppressorFactory {
    fn describe(&self, _io: &mut crate::graph::NodeIoBuilder<'_>) -> GraphResult<()> {
        Ok(())
    }

    fn build(self: Box<Self>, _ctx: &mut crate::graph::BuildCtx) -> GraphResult<Box<dyn NodeRunner>> {
        let internal_rate = internal_sample_rate(self.format, None);
        Ok(Box::new(NoiseSuppressorRunner {
            config: self.config,
            sample_rate_hz: internal_rate,
            num_channels: self.format.channels as usize,
            audio_in: self.audio_in,
            analysis_in: self.analysis_in,
            audio_out: self.audio_out,
            io: ChunkIo::new(self.format, internal_rate),
            analysis_io: self
                .analysis_in
                .map(|_| ChunkIo::new(self.analysis_format, 16_000)),
            suppressor: NoiseSuppressor::new(
                self.config,
                internal_rate,
                self.format.channels as usize,
            ),
        }))
    }
}

struct NoiseSuppressorRunner {
    config: NsConfig,
    sample_rate_hz: usize,
    num_channels: usize,
    audio_in: crate::graph::InPort<AudioChunk>,
    analysis_in: Option<crate::graph::InPort<AudioChunk>>,
    audio_out: OutPort<AudioChunk>,
    io: ChunkIo,
    analysis_io: Option<ChunkIo>,
    suppressor: NoiseSuppressor,
}

impl NodeRunner for NoiseSuppressorRunner {
    fn reset(&mut self) -> GraphResult<()> {
        self.suppressor = NoiseSuppressor::new(self.config, self.sample_rate_hz, self.num_channels);
        Ok(())
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        let Some(mut packet) = ctx.take(self.audio_in)? else {
            return Ok(());
        };

        match ctx.control_state() {
            NodeControlState::Active => {}
            NodeControlState::Bypassed => {
                if let Some(analysis_port) = self.analysis_in {
                    let _ = ctx.take(analysis_port)?;
                }
                return ctx.emit_handle(self.audio_out, packet);
            }
            NodeControlState::Suspended => {
                if let Some(analysis_port) = self.analysis_in {
                    let _ = ctx.take(analysis_port)?;
                }
                return Ok(());
            }
        }

        self.io.load_chunk(packet.payload(), "NoiseSuppressorNode")?;
        self.io.audio_buffer_mut().split_into_frequency_bands();

        if let (Some(analysis_port), Some(analysis_io)) = (self.analysis_in, self.analysis_io.as_mut()) {
            if let Some(analysis_packet) = ctx.take(analysis_port)? {
                analysis_io.load_chunk(analysis_packet.payload(), "NoiseSuppressorNode")?;
                analysis_io.audio_buffer_mut().split_into_frequency_bands();
                self.suppressor.analyze(analysis_io.audio_buffer());
            } else {
                self.suppressor.analyze(self.io.audio_buffer());
            }
        } else {
            self.suppressor.analyze(self.io.audio_buffer());
        }

        self.suppressor.process(self.io.audio_buffer_mut());
        self.io.audio_buffer_mut().merge_frequency_bands();
        self.io.store_chunk(packet.payload_mut())?;
        ctx.emit_handle(self.audio_out, packet)
    }
}
