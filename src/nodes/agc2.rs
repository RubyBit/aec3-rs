use crate::audio_processing::agc2::input_volume_controller::Config as InputVolumeControllerConfig;
use crate::audio_processing::gain_controller2::{GainController2, GainController2Config};
use crate::graph::{
    AccessMode, GraphBuilder, GraphError, GraphResult, InputOptions, NodeControlState, NodeFactory,
    NodeId, NodeRunner, NodeSpec, OutPort, OutputOptions, Packet, ProcessCtx, QueueConfig,
    SchedulePlan,
};

use super::audio::{AudioChunk, AudioFormat};
use super::util::{ChunkIo, internal_sample_rate, validate_audio_format};

#[derive(Debug, Clone, Copy)]
pub struct Agc2Node {
    node_id: NodeId,
    pub audio_in: crate::graph::InPort<AudioChunk>,
    pub applied_input_volume_in: Option<crate::graph::InPort<i32>>,
    pub capture_output_used_in: Option<crate::graph::InPort<bool>>,
    pub audio_out: OutPort<AudioChunk>,
    pub recommended_input_volume_out: Option<OutPort<i32>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Agc2NodeBuilder {
    format: AudioFormat,
    config: GainController2Config,
    input_volume_controller_config: InputVolumeControllerConfig,
    with_applied_input_volume: bool,
    with_capture_output_used: bool,
    with_recommended_input_volume: bool,
}

pub fn builder(format: AudioFormat) -> Agc2NodeBuilder {
    Agc2NodeBuilder {
        format,
        config: GainController2Config::default(),
        input_volume_controller_config: InputVolumeControllerConfig::default(),
        with_applied_input_volume: true,
        with_capture_output_used: true,
        with_recommended_input_volume: true,
    }
}

impl Agc2NodeBuilder {
    pub fn config(mut self, config: GainController2Config) -> Self {
        self.config = config;
        self
    }

    pub fn input_volume_controller_config(mut self, config: InputVolumeControllerConfig) -> Self {
        self.input_volume_controller_config = config;
        self
    }

    pub fn with_applied_input_volume(mut self, enable: bool) -> Self {
        self.with_applied_input_volume = enable;
        self
    }

    pub fn with_capture_output_used(mut self, enable: bool) -> Self {
        self.with_capture_output_used = enable;
        self
    }

    pub fn with_recommended_input_volume(mut self, enable: bool) -> Self {
        self.with_recommended_input_volume = enable;
        self
    }

    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<Agc2Node> {
        graph.add_node(self)
    }
}

impl Agc2Node {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

impl NodeSpec for Agc2NodeBuilder {
    type Handles = Agc2Node;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        validate_audio_format(self.format, "Agc2Node")?;
        if !GainController2::validate(&self.config) {
            return Err(GraphError::NodeError(
                "Agc2Node received an invalid GainController2Config".to_string(),
            ));
        }

        let node = graph.new_node("agc2");
        let audio_in = graph.register_input::<AudioChunk>(
            node,
            "audio_in",
            InputOptions {
                format_key: Some(self.format.schema_key()),
                ..InputOptions::default()
            },
        );
        let applied_input_volume_in = self.with_applied_input_volume.then(|| {
            graph.register_input::<i32>(
                node,
                "applied_input_volume_in",
                InputOptions {
                    queue: QueueConfig::latest(1),
                    access: AccessMode::PeekLatest,
                    ..InputOptions::default()
                },
            )
        });
        let capture_output_used_in = self.with_capture_output_used.then(|| {
            graph.register_input::<bool>(
                node,
                "capture_output_used_in",
                InputOptions {
                    queue: QueueConfig::latest(1),
                    access: AccessMode::PeekLatest,
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
        let recommended_input_volume_out = self.with_recommended_input_volume.then(|| {
            graph.register_output::<i32>(
                node,
                "recommended_input_volume_out",
                OutputOptions::default(),
            )
        });

        graph.finish_node(
            node,
            SchedulePlan::OnArrival {
                triggers: vec![audio_in.raw()],
            },
            Box::new(Agc2Factory {
                format: self.format,
                config: self.config,
                input_volume_controller_config: self.input_volume_controller_config,
                audio_in,
                applied_input_volume_in,
                capture_output_used_in,
                audio_out,
                recommended_input_volume_out,
            }),
        )?;

        Ok(Agc2Node {
            node_id: node,
            audio_in,
            applied_input_volume_in,
            capture_output_used_in,
            audio_out,
            recommended_input_volume_out,
        })
    }
}

struct Agc2Factory {
    format: AudioFormat,
    config: GainController2Config,
    input_volume_controller_config: InputVolumeControllerConfig,
    audio_in: crate::graph::InPort<AudioChunk>,
    applied_input_volume_in: Option<crate::graph::InPort<i32>>,
    capture_output_used_in: Option<crate::graph::InPort<bool>>,
    audio_out: OutPort<AudioChunk>,
    recommended_input_volume_out: Option<OutPort<i32>>,
}

impl NodeFactory for Agc2Factory {
    fn build(
        self: Box<Self>,
        _ctx: &mut crate::graph::BuildCtx,
    ) -> GraphResult<Box<dyn NodeRunner>> {
        let internal_rate = internal_sample_rate(self.format, None);
        Ok(Box::new(Agc2Runner {
            config: self.config,
            input_volume_controller_config: self.input_volume_controller_config,
            sample_rate_hz: internal_rate,
            num_channels: self.format.channels as usize,
            audio_in: self.audio_in,
            applied_input_volume_in: self.applied_input_volume_in,
            capture_output_used_in: self.capture_output_used_in,
            audio_out: self.audio_out,
            recommended_input_volume_out: self.recommended_input_volume_out,
            io: ChunkIo::new(self.format, internal_rate),
            controller: GainController2::new(
                self.config,
                self.input_volume_controller_config,
                internal_rate,
                self.format.channels as usize,
                true,
            ),
            last_applied_input_volume: None,
        }))
    }
}

struct Agc2Runner {
    config: GainController2Config,
    input_volume_controller_config: InputVolumeControllerConfig,
    sample_rate_hz: usize,
    num_channels: usize,
    audio_in: crate::graph::InPort<AudioChunk>,
    applied_input_volume_in: Option<crate::graph::InPort<i32>>,
    capture_output_used_in: Option<crate::graph::InPort<bool>>,
    audio_out: OutPort<AudioChunk>,
    recommended_input_volume_out: Option<OutPort<i32>>,
    io: ChunkIo,
    controller: GainController2,
    last_applied_input_volume: Option<i32>,
}

impl NodeRunner for Agc2Runner {
    fn reset(&mut self) -> GraphResult<()> {
        self.controller = GainController2::new(
            self.config,
            self.input_volume_controller_config,
            self.sample_rate_hz,
            self.num_channels,
            true,
        );
        self.last_applied_input_volume = None;
        Ok(())
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        let Some(mut packet) = ctx.take(self.audio_in)? else {
            return Ok(());
        };

        match ctx.control_state() {
            NodeControlState::Active => {}
            NodeControlState::Bypassed => return ctx.emit_handle(self.audio_out, packet),
            NodeControlState::Suspended => return Ok(()),
        }

        if let Some(port) = self.capture_output_used_in
            && let Some(capture_output_used) = ctx.peek(port)?
        {
            self.controller
                .set_capture_output_used(*capture_output_used.payload());
        }

        self.io.load_chunk(packet.payload(), "Agc2Node")?;

        let mut input_volume_changed = false;
        if let Some(port) = self.applied_input_volume_in
            && let Some(applied_input_volume) = ctx.peek(port)?
        {
            let applied_input_volume = *applied_input_volume.payload();
            if !(0..=255).contains(&applied_input_volume) {
                return Err(GraphError::NodeError(format!(
                    "Agc2Node expected applied input volume in [0, 255], got {}",
                    applied_input_volume
                )));
            }
            input_volume_changed = self.last_applied_input_volume != Some(applied_input_volume);
            self.last_applied_input_volume = Some(applied_input_volume);
            self.controller
                .analyze(applied_input_volume, self.io.audio_buffer());
        }

        self.controller
            .process(input_volume_changed, self.io.audio_buffer_mut());
        self.io.store_chunk(packet.payload_mut())?;
        ctx.emit_handle(self.audio_out, packet)?;

        if let (Some(port), Some(recommended_input_volume)) = (
            self.recommended_input_volume_out,
            self.controller.recommended_input_volume(),
        ) {
            ctx.emit(
                port,
                Packet {
                    meta: Default::default(),
                    payload: recommended_input_volume,
                },
            )?;
        }

        Ok(())
    }
}
