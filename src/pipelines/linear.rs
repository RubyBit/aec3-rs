//! A small wrapper for the common render-plus-capture voice processing chain.
//!
//! The graph API is useful when an application needs custom routing, side
//! outputs, or non-standard scheduling. Many applications only need the usual
//! full-duplex voice path:
//!
//! `render reference + microphone capture -> high-pass filter -> AEC3 -> noise suppression -> AGC2`
//!
//! This module builds that graph for you and exposes it as a frame-oriented API.
//! All audio frames are interleaved `f32` samples and must contain exactly 10 ms
//! of audio for their [`AudioFormat`].
//!
//! # Basic usage
//!
//! Feed render frames with [`LinearPipeline::handle_render_frame`] whenever the
//! far-end/render stream has data, then process microphone frames with
//! [`LinearPipeline::process_capture_frame`].
//!
//! ```no_run
//! use aec3::nodes::audio::AudioFormat;
//! use aec3::pipelines::linear;
//!
//! let format = AudioFormat::ten_ms(48_000, 1);
//! let mut pipeline = linear::builder(format, format)
//!     .initial_delay_ms(116)
//!     .build()?;
//!
//! let render = vec![0.0; format.sample_count()];
//! let capture = vec![0.0; format.sample_count()];
//! let mut output = vec![0.0; format.sample_count()];
//!
//! pipeline.handle_render_frame(&render)?;
//! if pipeline.process_capture_frame(&capture, &mut output)? {
//!     // `output` now contains the processed capture frame.
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The render and capture streams do not have to be pushed in the same call. For
//! offline processing, a simple loop can push one render frame and one capture
//! frame at a time. For real-time processing, push render frames as they arrive
//! from the playback or loopback path and process capture frames from the
//! microphone path.
//!
//! # Extending the generated graph
//!
//! Use [`LinearPipelineBuilder::build`] when the wrapper API is enough. Use
//! [`LinearPipelineBuilder::add_to`] when you want the standard chain inserted
//! into a larger [`GraphBuilder`] and need direct access to its ports.
//!
//! [`GraphBuilder`]: crate::graph::GraphBuilder
use crate::api::config::EchoCanceller3Config;
use crate::audio_processing::agc2::input_volume_controller::Config as InputVolumeControllerConfig;
use crate::audio_processing::gain_controller2::GainController2Config;
use crate::audio_processing::ns::NsConfig;
use crate::graph::{
    GraphBuilder, GraphResult, NodeControlState, NodeId, OutPort, Packet, PacketHandle, PacketMeta,
    QueueConfig, Runtime, Sink, Source,
};
use crate::nodes::{
    aec3, agc2,
    audio::{AudioChunk, AudioFormat},
    hpf, ns,
};

/// Builder for the standard linear voice pipeline.
///
/// The default pipeline enables high-pass filtering, AEC3, noise suppression,
/// and AGC2. Optional ports such as AEC3 metrics and linear output are disabled
/// until explicitly requested.
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

/// Graph handles created by [`LinearPipelineBuilder::add_to`].
///
/// These are useful when embedding the standard chain in a custom graph. Most
/// users can ignore this type and use [`LinearPipeline`] instead.
#[derive(Debug, Clone, Copy)]
pub struct LinearPipelineHandles {
    /// Source for interleaved 10 ms microphone/capture frames.
    pub capture: Source<AudioChunk>,
    /// Source for interleaved 10 ms far-end/render reference frames.
    pub render: Source<AudioChunk>,
    /// Source for external capture delay updates in milliseconds.
    pub delay_ms: Source<i32>,
    /// Sink that receives the processed capture stream.
    pub output: Sink<AudioChunk>,
    /// Optional sink for AEC3 linear output before residual echo suppression.
    pub linear_output: Option<Sink<AudioChunk>>,
    /// Optional sink for AEC3 metrics.
    pub metrics: Option<Sink<aec3::Aec3Metrics>>,
    /// Output port at the end of the enabled processing chain before the sink.
    pub output_port: OutPort<AudioChunk>,
    /// High-pass filter node, when enabled.
    pub high_pass: Option<hpf::HighPassNode>,
    /// AEC3 node handle.
    pub aec3: aec3::Aec3Node,
    /// Noise suppression node, when enabled.
    pub noise_suppression: Option<ns::NoiseSuppressorNode>,
    /// AGC2 node, when enabled.
    pub gain_controller2: Option<agc2::Agc2Node>,
}

/// Frame-oriented runtime wrapper for the standard linear voice pipeline.
///
/// This type owns the generated graph runtime and provides convenience methods
/// for the two streams applications normally have: render reference audio and
/// microphone capture audio.
pub struct LinearPipeline {
    runtime: Runtime,
    handles: LinearPipelineHandles,
    render_format: AudioFormat,
    capture_format: AudioFormat,
    next_sequence: u64,
}

/// Starts configuring a standard linear voice pipeline.
///
/// `render_format` describes the far-end/playback reference frames and
/// `capture_format` describes the microphone frames. Each format must represent
/// 10 ms chunks; [`AudioFormat::ten_ms`] is the easiest way to construct them.
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
    /// Replaces the default AEC3 configuration.
    ///
    /// Use this for tuning suppression, delay behavior, or other AEC3 internals
    /// while keeping the standard pipeline layout.
    pub fn aec3_config(mut self, config: EchoCanceller3Config) -> Self {
        self.aec3_config = Some(config);
        self
    }

    /// Replaces the default noise suppression configuration.
    pub fn noise_suppression_config(mut self, config: NsConfig) -> Self {
        self.ns_config = config;
        self
    }

    /// Replaces the default AGC2 configuration.
    pub fn gain_controller2_config(mut self, config: GainController2Config) -> Self {
        self.agc2_config = config;
        self
    }

    /// Replaces the default input volume controller configuration used by AGC2.
    pub fn input_volume_controller_config(mut self, config: InputVolumeControllerConfig) -> Self {
        self.input_volume_controller_config = config;
        self
    }

    /// Enables or disables the noise suppression stage.
    ///
    /// Enabled by default.
    pub fn enable_noise_suppression(mut self, enable: bool) -> Self {
        self.enable_noise_suppression = enable;
        self
    }

    /// Enables or disables the high-pass filter before AEC3.
    ///
    /// Enabled by default.
    pub fn enable_high_pass_filter(mut self, enable: bool) -> Self {
        self.enable_high_pass_filter = enable;
        self
    }

    /// Enables or disables the AGC2 stage after noise suppression.
    ///
    /// Enabled by default.
    pub fn enable_gain_controller2(mut self, enable: bool) -> Self {
        self.enable_gain_controller2 = enable;
        self
    }

    /// Exposes AEC3's linear output as an additional sink.
    ///
    /// The linear output is the AEC output before nonlinear residual echo
    /// suppression. It is mostly useful for analysis, diagnostics, or feeding a
    /// side-chain processor.
    pub fn export_linear_output(mut self, enable: bool) -> Self {
        self.export_linear_output = enable;
        self
    }

    /// Exposes AEC3 metrics as an additional sink.
    ///
    /// Metrics can be drained with [`LinearPipeline::try_pull_metrics`] when
    /// using [`build`](Self::build), or from [`LinearPipelineHandles::metrics`]
    /// when using [`add_to`](Self::add_to).
    pub fn export_metrics(mut self, enable: bool) -> Self {
        self.export_metrics = enable;
        self
    }

    /// Sets the initial render-to-capture delay estimate in milliseconds.
    ///
    /// You can update the delay later with [`LinearPipeline::set_delay_ms`].
    pub fn initial_delay_ms(mut self, delay_ms: i32) -> Self {
        self.initial_delay_ms = Some(delay_ms);
        self
    }

    /// Adds the standard chain to an existing graph and returns its handles.
    ///
    /// This is the escape hatch for applications that want the convenience of
    /// the standard layout but still need to attach extra nodes, taps, or custom
    /// sinks before building the graph.
    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<LinearPipelineHandles> {
        let capture = graph.source::<AudioChunk>("capture");
        let render = graph.source::<AudioChunk>("render");
        let delay_ms = graph.source::<i32>("delay_ms");
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

    /// Builds a self-contained [`LinearPipeline`].
    ///
    /// This is the simplest entry point when you only need to feed render and
    /// capture frames and receive processed capture frames.
    pub fn build(self) -> GraphResult<LinearPipeline> {
        let render_format = self.render_format;
        let capture_format = self.capture_format;
        let mut graph = GraphBuilder::new();
        let handles = self.add_to(&mut graph)?;
        let spec = graph.build()?;
        let runtime = Runtime::new(spec)?;
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
    /// Returns the graph handles owned by this wrapper.
    ///
    /// Most applications use the wrapper methods instead of these handles, but
    /// they are available for control operations and introspection.
    pub fn handles(&self) -> &LinearPipelineHandles {
        &self.handles
    }

    /// Returns the underlying graph runtime.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Returns the underlying graph runtime mutably.
    ///
    /// Use this when the wrapper methods are not enough and you need direct
    /// graph runtime access.
    pub fn runtime_mut(&mut self) -> &mut Runtime {
        &mut self.runtime
    }

    /// Returns the expected render frame format.
    pub fn render_format(&self) -> AudioFormat {
        self.render_format
    }

    /// Returns the expected capture and output frame format.
    pub fn capture_format(&self) -> AudioFormat {
        self.capture_format
    }

    /// Sets the runtime state of a node in the generated graph.
    ///
    /// Node IDs are available through [`handles`](Self::handles).
    pub fn set_node_state(&mut self, node_id: NodeId, state: NodeControlState) -> GraphResult<()> {
        self.runtime.set_node_state(node_id, state)
    }

    /// Resets a node in the generated graph.
    ///
    /// Node IDs are available through [`handles`](Self::handles).
    pub fn reset_node(&mut self, node_id: NodeId) -> GraphResult<()> {
        self.runtime.reset_node(node_id)
    }

    /// Resets the AEC3 node state.
    pub fn reset_aec3(&mut self) -> GraphResult<()> {
        self.runtime.reset_node(self.handles.aec3.node_id())
    }

    /// Resets the high-pass filter state when that stage is enabled.
    pub fn reset_high_pass(&mut self) -> GraphResult<()> {
        match self.handles.high_pass {
            Some(node) => self.runtime.reset_node(node.node_id()),
            None => Ok(()),
        }
    }

    /// Resets the noise suppression state when that stage is enabled.
    pub fn reset_noise_suppression(&mut self) -> GraphResult<()> {
        match self.handles.noise_suppression {
            Some(node) => self.runtime.reset_node(node.node_id()),
            None => Ok(()),
        }
    }

    /// Resets the AGC2 state when that stage is enabled.
    pub fn reset_gain_controller2(&mut self) -> GraphResult<()> {
        match self.handles.gain_controller2 {
            Some(node) => self.runtime.reset_node(node.node_id()),
            None => Ok(()),
        }
    }

    /// Updates the current render-to-capture delay estimate in milliseconds.
    ///
    /// Call this when your application has a better external delay estimate than
    /// the initial value configured by [`LinearPipelineBuilder::initial_delay_ms`].
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

    /// Feeds one interleaved 10 ms render/reference frame into the pipeline.
    ///
    /// The slice length must equal `render_format().sample_count()`. This method
    /// runs the graph until no more work is immediately available.
    pub fn handle_render_frame(&mut self, render: &[f32]) -> GraphResult<()> {
        let meta = self.next_meta();
        self.handle_render_frame_with_meta(render, meta)
    }

    /// Feeds one render/reference frame with explicit packet metadata.
    ///
    /// Use this when integrating with a system that already tracks timestamps or
    /// sequence numbers. For simple usage, prefer [`handle_render_frame`](Self::handle_render_frame).
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

    /// Processes one interleaved 10 ms capture frame.
    ///
    /// Both `capture` and `output` must have
    /// `capture_format().sample_count()` samples. Returns `true` when a
    /// processed frame was written to `output`. If no output is available yet,
    /// `output` is filled with silence and the method returns `false`.
    pub fn process_capture_frame(
        &mut self,
        capture: &[f32],
        output: &mut [f32],
    ) -> GraphResult<bool> {
        let meta = self.next_meta();
        self.process_capture_frame_with_meta(capture, meta, output)
    }

    /// Processes one capture frame with explicit packet metadata.
    ///
    /// Use this when integrating with a system that already tracks timestamps or
    /// sequence numbers. For simple usage, prefer
    /// [`process_capture_frame`](Self::process_capture_frame).
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

    /// Pulls the next processed output packet, if one is queued.
    ///
    /// [`process_capture_frame`](Self::process_capture_frame) already pulls into
    /// a caller-provided slice. This lower-level method is mainly useful when
    /// driving the underlying runtime directly.
    pub fn try_pull_output(&mut self) -> GraphResult<Option<PacketHandle<AudioChunk>>> {
        self.runtime.try_pull(self.handles.output)
    }

    /// Pulls the next linear-output packet, if that export is enabled.
    ///
    /// Enable it with [`LinearPipelineBuilder::export_linear_output`].
    pub fn try_pull_linear_output(&mut self) -> GraphResult<Option<PacketHandle<AudioChunk>>> {
        match self.handles.linear_output {
            Some(sink) => self.runtime.try_pull(sink),
            None => Ok(None),
        }
    }

    /// Pulls the latest AEC3 metrics packet, if metrics export is enabled.
    ///
    /// Enable it with [`LinearPipelineBuilder::export_metrics`].
    pub fn try_pull_metrics(&mut self) -> GraphResult<Option<PacketHandle<aec3::Aec3Metrics>>> {
        match self.handles.metrics {
            Some(sink) => self.runtime.try_pull(sink),
            None => Ok(None),
        }
    }

    fn validate_frame(&self, samples: &[f32], format: AudioFormat, kind: &str) -> GraphResult<()> {
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
            sequence: Some(sequence),
            ..PacketMeta::default()
        }
    }
}
