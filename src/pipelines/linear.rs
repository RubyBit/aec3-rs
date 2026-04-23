use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::agc2::input_volume_controller::Config as InputVolumeControllerConfig;
use crate::audio_processing::gain_controller2::GainController2Config;
use crate::audio_processing::ns::NsConfig;
use crate::graph::{
    GraphBuilder, GraphResult, NodeControlState, NodeId, OutPort, Packet, PacketHandle,
    PacketMeta, QueueConfig, Runtime, RuntimeOptions, Sink, Source,
};
use crate::nodes::{
    aec3,
    agc2,
    audio::{AudioChunk, AudioFormat},
    hpf,
    ns,
};

#[derive(Debug, Clone)]
pub struct LinearPipelineBuilder {
    render_format: AudioFormat,
    capture_format: AudioFormat,
    aec3_config: Option<EchoCanceller3Config>,
    ns_config: NsConfig,
    agc2_config: GainController2Config,
    input_volume_controller_config: InputVolumeControllerConfig,
    enable_high_pass_filter: bool,
    enable_noise_suppression: bool,
    enable_gain_controller2: bool,
    export_linear_output: bool,
    export_metrics: bool,
    initial_delay_ms: Option<i32>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearPipelineHandles {
    pub capture: Source<AudioChunk>,
    pub render: Source<AudioChunk>,
    pub delay_ms: Source<i32>,
    pub output: Sink<AudioChunk>,
    pub linear_output: Option<Sink<AudioChunk>>,
    pub metrics: Option<Sink<aec3::Aec3Metrics>>,
    pub output_port: OutPort<AudioChunk>,
    pub high_pass: Option<hpf::HighPassNode>,
    pub aec3: aec3::Aec3Node,
    pub noise_suppression: Option<ns::NoiseSuppressorNode>,
    pub gain_controller2: Option<agc2::Agc2Node>,
}

pub struct LinearPipeline {
    runtime: Runtime,
    handles: LinearPipelineHandles,
    render_format: AudioFormat,
    capture_format: AudioFormat,
    next_sequence: u64,
}

pub fn builder(render_format: AudioFormat, capture_format: AudioFormat) -> LinearPipelineBuilder {
    LinearPipelineBuilder {
        render_format,
        capture_format,
        aec3_config: None,
        ns_config: NsConfig::default(),
        agc2_config: GainController2Config::default(),
        input_volume_controller_config: InputVolumeControllerConfig::default(),
        enable_high_pass_filter: true,
        enable_noise_suppression: true,
        enable_gain_controller2: true,
        export_linear_output: false,
        export_metrics: false,
        initial_delay_ms: None,
    }
}

impl LinearPipelineBuilder {
    pub fn aec3_config(mut self, config: EchoCanceller3Config) -> Self {
        self.aec3_config = Some(config);
        self
    }

    pub fn noise_suppression_config(mut self, config: NsConfig) -> Self {
        self.ns_config = config;
        self
    }

    pub fn gain_controller2_config(mut self, config: GainController2Config) -> Self {
        self.agc2_config = config;
        self
    }

    pub fn input_volume_controller_config(
        mut self,
        config: InputVolumeControllerConfig,
    ) -> Self {
        self.input_volume_controller_config = config;
        self
    }

    pub fn enable_noise_suppression(mut self, enable: bool) -> Self {
        self.enable_noise_suppression = enable;
        self
    }

    pub fn enable_high_pass_filter(mut self, enable: bool) -> Self {
        self.enable_high_pass_filter = enable;
        self
    }

    pub fn enable_gain_controller2(mut self, enable: bool) -> Self {
        self.enable_gain_controller2 = enable;
        self
    }

    pub fn export_linear_output(mut self, enable: bool) -> Self {
        self.export_linear_output = enable;
        self
    }

    pub fn export_metrics(mut self, enable: bool) -> Self {
        self.export_metrics = enable;
        self
    }

    pub fn initial_delay_ms(mut self, delay_ms: i32) -> Self {
        self.initial_delay_ms = Some(delay_ms);
        self
    }

    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<LinearPipelineHandles> {
        let capture = graph.source::<AudioChunk>("capture", QueueConfig::audio_default());
        let render = graph.source::<AudioChunk>("render", QueueConfig::audio_default());
        let delay_ms = graph.source::<i32>("delay_ms", QueueConfig::latest(1));
        let output = graph.sink::<AudioChunk>("output", QueueConfig::audio_default());

        let uses_linear_analysis = self.enable_noise_suppression
            && self.ns_config.analyze_linear_aec_output_when_available;
        let export_linear_output = self.export_linear_output || uses_linear_analysis;

        let mut aec3_builder = aec3::builder(self.render_format, self.capture_format)
            .export_linear_output(export_linear_output)
            .export_metrics(self.export_metrics);
        if let Some(config) = self.aec3_config {
            aec3_builder = aec3_builder.with_config(config);
        }
        if let Some(delay) = self.initial_delay_ms {
            aec3_builder = aec3_builder.initial_delay_ms(delay);
        }
        let aec3 = aec3_builder.add_to(graph)?;

        let high_pass = if self.enable_high_pass_filter {
            let filter = hpf::builder(self.capture_format).add_to(graph)?;
            graph.connect(capture, filter.audio_in)?;
            graph.connect(filter.audio_out, aec3.capture_in)?;
            Some(filter)
        } else {
            graph.connect(capture, aec3.capture_in)?;
            None
        };
        graph.connect(render, aec3.render_in)?;
        graph.connect(delay_ms, aec3.delay_in)?;

        let mut output_port = aec3.capture_out;
        let noise_suppression = if self.enable_noise_suppression {
            let suppressor = ns::builder(self.capture_format)
                .with_analysis_input(uses_linear_analysis)
                .config(self.ns_config)
                .add_to(graph)?;
            graph.connect(output_port, suppressor.audio_in)?;
            if uses_linear_analysis {
                graph.connect(
                    aec3.linear_out.expect("linear output enabled for analysis"),
                    suppressor
                        .analysis_in
                        .expect("analysis input enabled for linear output"),
                )?;
            }
            output_port = suppressor.audio_out;
            Some(suppressor)
        } else {
            None
        };

        let gain_controller2 = if self.enable_gain_controller2 {
            let agc = agc2::builder(self.capture_format)
                .config(self.agc2_config)
                .input_volume_controller_config(self.input_volume_controller_config)
                .add_to(graph)?;
            graph.connect(output_port, agc.audio_in)?;
            output_port = agc.audio_out;
            Some(agc)
        } else {
            None
        };

        graph.connect(output_port, output)?;

        let linear_output = if self.export_linear_output {
            let sink = graph.sink::<AudioChunk>("linear_output", QueueConfig::audio_default());
            graph.connect(
                aec3.linear_out
                    .expect("linear output sink requested but AEC3 linear output is disabled"),
                sink,
            )?;
            Some(sink)
        } else {
            None
        };

        let metrics = if self.export_metrics {
            let sink = graph.sink::<aec3::Aec3Metrics>("metrics", QueueConfig::latest(1));
            graph.connect(
                aec3.metrics_out
                    .expect("metrics sink requested but AEC3 metrics output is disabled"),
                sink,
            )?;
            Some(sink)
        } else {
            None
        };

        Ok(LinearPipelineHandles {
            capture,
            render,
            delay_ms,
            output,
            linear_output,
            metrics,
            output_port,
            high_pass,
            aec3,
            noise_suppression,
            gain_controller2,
        })
    }

    pub fn build(self) -> GraphResult<LinearPipeline> {
        let render_format = self.render_format;
        let capture_format = self.capture_format;
        let mut graph = GraphBuilder::new();
        let handles = self.add_to(&mut graph)?;
        let spec = graph.build()?;
        let runtime = Runtime::new(spec, RuntimeOptions)?;
        Ok(LinearPipeline {
            runtime,
            handles,
            render_format,
            capture_format,
            next_sequence: 1,
        })
    }
}

impl LinearPipeline {
    pub fn handles(&self) -> &LinearPipelineHandles {
        &self.handles
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    pub fn render_format(&self) -> AudioFormat {
        self.render_format
    }

    pub fn capture_format(&self) -> AudioFormat {
        self.capture_format
    }

    pub fn set_node_state(&mut self, node_id: NodeId, state: NodeControlState) -> GraphResult<()> {
        self.runtime.set_node_state(node_id, state)
    }

    pub fn reset_node(&mut self, node_id: NodeId) -> GraphResult<()> {
        self.runtime.reset_node(node_id)
    }

    pub fn reset_aec3(&mut self) -> GraphResult<()> {
        self.runtime.reset_node(self.handles.aec3.node_id())
    }

    pub fn reset_high_pass(&mut self) -> GraphResult<()> {
        match self.handles.high_pass {
            Some(node) => self.runtime.reset_node(node.node_id()),
            None => Ok(()),
        }
    }

    pub fn reset_noise_suppression(&mut self) -> GraphResult<()> {
        match self.handles.noise_suppression {
            Some(node) => self.runtime.reset_node(node.node_id()),
            None => Ok(()),
        }
    }

    pub fn reset_gain_controller2(&mut self) -> GraphResult<()> {
        match self.handles.gain_controller2 {
            Some(node) => self.runtime.reset_node(node.node_id()),
            None => Ok(()),
        }
    }

    pub fn set_delay_ms(&mut self, delay_ms: i32) -> GraphResult<()> {
        let meta = self.next_meta();
        self.runtime.push(
            self.handles.delay_ms,
            Packet {
                meta,
                payload: delay_ms,
            },
        )
    }

    pub fn handle_render_frame(&mut self, render: &[f32]) -> GraphResult<()> {
        let meta = self.next_meta();
        self.handle_render_frame_with_meta(render, meta)
    }

    pub fn handle_render_frame_with_meta(
        &mut self,
        render: &[f32],
        meta: PacketMeta,
    ) -> GraphResult<()> {
        self.validate_frame(render, self.render_format, "render")?;
        self.runtime.push(
            self.handles.render,
            Packet {
                meta,
                payload: AudioChunk::from_interleaved(self.render_format, render),
            },
        )?;
        self.runtime.run_until_stalled()?;
        Ok(())
    }

    pub fn process_capture_frame(
        &mut self,
        capture: &[f32],
        output: &mut [f32],
    ) -> GraphResult<bool> {
        let meta = self.next_meta();
        self.process_capture_frame_with_meta(capture, meta, output)
    }

    pub fn process_capture_frame_with_meta(
        &mut self,
        capture: &[f32],
        meta: PacketMeta,
        output: &mut [f32],
    ) -> GraphResult<bool> {
        self.validate_frame(capture, self.capture_format, "capture")?;
        self.validate_frame(output, self.capture_format, "output")?;
        self.runtime.push(
            self.handles.capture,
            Packet {
                meta,
                payload: AudioChunk::from_interleaved(self.capture_format, capture),
            },
        )?;
        self.runtime.run_until_stalled()?;

        match self.runtime.try_pull(self.handles.output)? {
            Some(packet) => {
                output.copy_from_slice(packet.payload().samples());
                Ok(true)
            }
            None => {
                output.fill(0.0);
                Ok(false)
            }
        }
    }

    pub fn try_pull_output(&mut self) -> GraphResult<Option<PacketHandle<AudioChunk>>> {
        self.runtime.try_pull(self.handles.output)
    }

    pub fn try_pull_linear_output(&mut self) -> GraphResult<Option<PacketHandle<AudioChunk>>> {
        match self.handles.linear_output {
            Some(sink) => self.runtime.try_pull(sink),
            None => Ok(None),
        }
    }

    pub fn try_pull_metrics(&mut self) -> GraphResult<Option<PacketHandle<aec3::Aec3Metrics>>> {
        match self.handles.metrics {
            Some(sink) => self.runtime.try_pull(sink),
            None => Ok(None),
        }
    }

    fn validate_frame(
        &self,
        samples: &[f32],
        format: AudioFormat,
        kind: &str,
    ) -> GraphResult<()> {
        if samples.len() != format.sample_count() {
            return Err(crate::graph::GraphError::NodeError(format!(
                "{} frame expected {} samples for {}, got {}",
                kind,
                format.sample_count(),
                format.schema_key(),
                samples.len()
            )));
        }
        Ok(())
    }

    fn next_meta(&mut self) -> PacketMeta {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        PacketMeta {
            sequence,
            ..PacketMeta::default()
        }
    }
}
