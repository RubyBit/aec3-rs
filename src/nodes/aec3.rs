use std::path::PathBuf;

use crate::api::config::EchoCanceller3Config;
use crate::api::control::{EchoControl, Metrics};
use crate::audio_processing::aec3::echo_canceller3::EchoCanceller3;
use crate::audio_processing::logging::apm_data_dumper::DiagnosticLevel;
use crate::graph::{
    AccessMode, GraphBuilder, GraphResult, InputOptions, NodeControlState, NodeFactory, NodeId,
    NodeRunner, NodeSpec, OutPort, OutputOptions, Packet, ProcessCtx, QueueConfig, SchedulePlan,
};

use super::audio::{AudioChunk, AudioFormat};
use super::util::{ChunkIo, internal_sample_rate, maybe_enable_diagnostics, validate_audio_format};

pub type Aec3Metrics = Metrics;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticEvent {
    pub kind: &'static str,
    pub sequence: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct Aec3Node {
    node_id: NodeId,
    pub render_in: crate::graph::InPort<AudioChunk>,
    pub capture_in: crate::graph::InPort<AudioChunk>,
    pub delay_in: crate::graph::InPort<i32>,
    pub capture_out: OutPort<AudioChunk>,
    pub linear_out: Option<OutPort<AudioChunk>>,
    pub metrics_out: Option<OutPort<Aec3Metrics>>,
    pub diagnostics_out: Option<OutPort<DiagnosticEvent>>,
}

#[derive(Debug, Clone)]
pub struct Aec3NodeBuilder {
    render_format: AudioFormat,
    capture_format: AudioFormat,
    config: Option<EchoCanceller3Config>,
    export_linear_output: bool,
    export_metrics: bool,
    export_diagnostics: bool,
    diagnostics_enabled: bool,
    diagnostics_output_dir: Option<PathBuf>,
    diagnostics_level: Option<DiagnosticLevel>,
    initial_delay_ms: Option<i32>,
}

pub fn builder(render_format: AudioFormat, capture_format: AudioFormat) -> Aec3NodeBuilder {
    Aec3NodeBuilder {
        render_format,
        capture_format,
        config: None,
        export_linear_output: false,
        export_metrics: false,
        export_diagnostics: false,
        diagnostics_enabled: false,
        diagnostics_output_dir: None,
        diagnostics_level: None,
        initial_delay_ms: None,
    }
}

impl Aec3NodeBuilder {
    pub fn with_config(mut self, config: EchoCanceller3Config) -> Self {
        self.config = Some(config);
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

    pub fn export_diagnostics(mut self, enable: bool) -> Self {
        self.export_diagnostics = enable;
        self
    }

    pub fn enable_diagnostics(mut self, enable: bool) -> Self {
        self.diagnostics_enabled = enable;
        self
    }

    pub fn diagnostics_output_directory<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.diagnostics_output_dir = Some(path.into());
        self
    }

    pub fn diagnostics_level(mut self, level: DiagnosticLevel) -> Self {
        self.diagnostics_level = Some(level);
        self
    }

    pub fn initial_delay_ms(mut self, delay_ms: i32) -> Self {
        self.initial_delay_ms = Some(delay_ms);
        self
    }

    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<Aec3Node> {
        graph.add_node(self)
    }
}

impl Aec3Node {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

impl NodeSpec for Aec3NodeBuilder {
    type Handles = Aec3Node;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        validate_audio_format(self.render_format, "Aec3Node")?;
        validate_audio_format(self.capture_format, "Aec3Node")?;

        let linear_format = AudioFormat::ten_ms(16_000, self.capture_format.channels);
        let node = graph.new_node("aec3");
        // `latest(1)` is safe on this trigger input because the runner reads
        // render audio only through `ctx.trigger_packet`, never `ctx.take`:
        // the trigger event carries the packet, so replacing queued packets
        // cannot desync trigger events from queue contents.
        let render_in = graph.register_input::<AudioChunk>(
            node,
            "render_in",
            InputOptions {
                queue: QueueConfig::latest(1),
                access: AccessMode::PeekLatest,
                format_key: Some(self.render_format.schema_key()),
            },
        );
        let capture_in = graph.register_input::<AudioChunk>(
            node,
            "capture_in",
            InputOptions {
                format_key: Some(self.capture_format.schema_key()),
                ..InputOptions::default()
            },
        );
        let delay_in = graph.register_input::<i32>(
            node,
            "delay_in",
            InputOptions {
                queue: QueueConfig::latest(1),
                access: AccessMode::PeekLatest,
                ..InputOptions::default()
            },
        );
        let capture_out = graph.register_output::<AudioChunk>(
            node,
            "capture_out",
            OutputOptions {
                format_key: Some(self.capture_format.schema_key()),
            },
        );
        let linear_out = self.export_linear_output.then(|| {
            graph.register_output::<AudioChunk>(
                node,
                "linear_out",
                OutputOptions {
                    format_key: Some(linear_format.schema_key()),
                },
            )
        });
        let metrics_out = self.export_metrics.then(|| {
            graph.register_output::<Aec3Metrics>(node, "metrics_out", OutputOptions::default())
        });
        let diagnostics_out = self.export_diagnostics.then(|| {
            graph.register_output::<DiagnosticEvent>(
                node,
                "diagnostics_out",
                OutputOptions::default(),
            )
        });

        graph.finish_node(
            node,
            SchedulePlan::OnArrival {
                triggers: vec![render_in.raw(), capture_in.raw()],
            },
            Box::new(Aec3Factory {
                render_format: self.render_format,
                capture_format: self.capture_format,
                linear_format,
                config: self.config,
                diagnostics_enabled: self.diagnostics_enabled,
                diagnostics_output_dir: self.diagnostics_output_dir,
                diagnostics_level: self.diagnostics_level,
                initial_delay_ms: self.initial_delay_ms,
                render_in,
                capture_in,
                delay_in,
                capture_out,
                linear_out,
                metrics_out,
                diagnostics_out,
            }),
        )?;

        Ok(Aec3Node {
            node_id: node,
            render_in,
            capture_in,
            delay_in,
            capture_out,
            linear_out,
            metrics_out,
            diagnostics_out,
        })
    }
}

struct Aec3Factory {
    render_format: AudioFormat,
    capture_format: AudioFormat,
    linear_format: AudioFormat,
    config: Option<EchoCanceller3Config>,
    diagnostics_enabled: bool,
    diagnostics_output_dir: Option<PathBuf>,
    diagnostics_level: Option<DiagnosticLevel>,
    initial_delay_ms: Option<i32>,
    render_in: crate::graph::InPort<AudioChunk>,
    capture_in: crate::graph::InPort<AudioChunk>,
    delay_in: crate::graph::InPort<i32>,
    capture_out: OutPort<AudioChunk>,
    linear_out: Option<OutPort<AudioChunk>>,
    metrics_out: Option<OutPort<Aec3Metrics>>,
    diagnostics_out: Option<OutPort<DiagnosticEvent>>,
}

impl NodeFactory for Aec3Factory {
    fn build(
        self: Box<Self>,
        _ctx: &mut crate::graph::BuildCtx,
    ) -> GraphResult<Box<dyn NodeRunner>> {
        maybe_enable_diagnostics(
            self.diagnostics_enabled,
            self.diagnostics_output_dir.as_deref(),
            self.diagnostics_level,
        );

        let internal_rate = internal_sample_rate(self.capture_format, Some(self.render_format));
        // A default multichannel config is supplied only when the caller set no
        // config: if the caller provides a non-multichannel config, that config
        // is used for both mono and multichannel echo cancellation.
        let (mut config, mut multichannel_config) = match self.config {
            Some(config) => (config, None),
            None => (
                EchoCanceller3Config::default(),
                Some(EchoCanceller3Config::create_default_multichannel_config()),
            ),
        };
        if self.linear_out.is_some() {
            // Both configs must agree on this field, since the linear output is
            // set up outside the switchable part of the pipeline.
            config.filter.export_linear_aec_output = true;
            if let Some(multichannel_config) = multichannel_config.as_mut() {
                multichannel_config.filter.export_linear_aec_output = true;
            }
        }

        let echo = Aec3Runner::build_echo(
            config.clone(),
            multichannel_config.clone(),
            internal_rate,
            self.render_format.channels as usize,
            self.capture_format.channels as usize,
            self.diagnostics_enabled,
            self.initial_delay_ms,
        );

        Ok(Box::new(Aec3Runner {
            config,
            multichannel_config,
            internal_rate,
            render_channels: self.render_format.channels as usize,
            capture_channels: self.capture_format.channels as usize,
            diagnostics_enabled: self.diagnostics_enabled,
            last_delay_ms: self.initial_delay_ms,
            render_in: self.render_in,
            capture_in: self.capture_in,
            delay_in: self.delay_in,
            capture_out: self.capture_out,
            linear_out: self.linear_out,
            metrics_out: self.metrics_out,
            diagnostics_out: self.diagnostics_out,
            render_io: ChunkIo::new(self.render_format, internal_rate),
            capture_io: ChunkIo::new(self.capture_format, internal_rate),
            linear_io: self
                .linear_out
                .map(|_| ChunkIo::new(self.linear_format, 16_000)),
            linear_format: self.linear_format,
            echo,
        }))
    }
}

struct Aec3Runner {
    config: EchoCanceller3Config,
    multichannel_config: Option<EchoCanceller3Config>,
    internal_rate: usize,
    render_channels: usize,
    capture_channels: usize,
    diagnostics_enabled: bool,
    last_delay_ms: Option<i32>,
    render_in: crate::graph::InPort<AudioChunk>,
    capture_in: crate::graph::InPort<AudioChunk>,
    delay_in: crate::graph::InPort<i32>,
    capture_out: OutPort<AudioChunk>,
    linear_out: Option<OutPort<AudioChunk>>,
    metrics_out: Option<OutPort<Aec3Metrics>>,
    diagnostics_out: Option<OutPort<DiagnosticEvent>>,
    render_io: ChunkIo,
    capture_io: ChunkIo,
    linear_io: Option<ChunkIo>,
    linear_format: AudioFormat,
    echo: EchoCanceller3,
}

impl Aec3Runner {
    #[allow(clippy::too_many_arguments)]
    fn build_echo(
        config: EchoCanceller3Config,
        multichannel_config: Option<EchoCanceller3Config>,
        internal_rate: usize,
        render_channels: usize,
        capture_channels: usize,
        diagnostics_enabled: bool,
        delay_ms: Option<i32>,
    ) -> EchoCanceller3 {
        let mut echo = EchoCanceller3::with_multichannel_config(
            config,
            multichannel_config,
            internal_rate as i32,
            render_channels,
            capture_channels,
        );
        if diagnostics_enabled {
            echo.initiate_new_set_of_recordings();
        }
        if let Some(delay_ms) = delay_ms {
            echo.set_audio_buffer_delay(delay_ms);
        }
        echo
    }

    fn emit_diagnostic(
        &self,
        ctx: &mut ProcessCtx<'_>,
        meta: crate::graph::PacketMeta,
        kind: &'static str,
    ) -> GraphResult<()> {
        if let Some(port) = self.diagnostics_out {
            let sequence = meta.sequence;
            ctx.emit(
                port,
                Packet {
                    meta,
                    payload: DiagnosticEvent { kind, sequence },
                },
            )?;
        }
        Ok(())
    }

    fn apply_delay_control(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        let Some(delay_packet) = ctx.peek(self.delay_in)? else {
            return Ok(());
        };
        let delay_ms = *delay_packet.payload();
        if self.last_delay_ms != Some(delay_ms) {
            self.echo.set_audio_buffer_delay(delay_ms);
            self.last_delay_ms = Some(delay_ms);
            self.emit_diagnostic(ctx, delay_packet.meta().clone(), "delay_update")?;
        }
        Ok(())
    }
}

impl NodeRunner for Aec3Runner {
    fn reset(&mut self) -> GraphResult<()> {
        self.echo = Self::build_echo(
            self.config.clone(),
            self.multichannel_config.clone(),
            self.internal_rate,
            self.render_channels,
            self.capture_channels,
            self.diagnostics_enabled,
            self.last_delay_ms,
        );
        Ok(())
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        self.apply_delay_control(ctx)?;

        match ctx.control_state() {
            NodeControlState::Active => {}
            NodeControlState::Bypassed => {
                if ctx.trigger_is(self.capture_in) {
                    if let Some(capture_packet) = ctx.take(self.capture_in)? {
                        ctx.emit_handle(self.capture_out, capture_packet)?;
                    }
                }
                return Ok(());
            }
            NodeControlState::Suspended => {
                if ctx.trigger_is(self.capture_in) {
                    let _ = ctx.take(self.capture_in)?;
                }
                return Ok(());
            }
        }

        if ctx.trigger_is(self.render_in) {
            if let Some(render_packet) = ctx.trigger_packet(self.render_in)? {
                self.render_io
                    .load_chunk(render_packet.payload(), "Aec3Node(render)")?;
                self.render_io
                    .audio_buffer_mut()
                    .split_into_frequency_bands();
                self.echo.analyze_render(self.render_io.audio_buffer_mut());
                self.emit_diagnostic(ctx, render_packet.meta().clone(), "render_analyzed")?;
            }
            return Ok(());
        }

        let Some(mut capture_packet) = ctx.take(self.capture_in)? else {
            return Ok(());
        };

        self.capture_io
            .load_chunk(capture_packet.payload(), "Aec3Node(capture)")?;
        self.echo
            .analyze_capture(self.capture_io.audio_buffer_mut());
        self.capture_io
            .audio_buffer_mut()
            .split_into_frequency_bands();

        if let (Some(linear_out), Some(linear_io)) = (self.linear_out, self.linear_io.as_mut()) {
            self.echo.process_capture_with_linear_output(
                self.capture_io.audio_buffer_mut(),
                linear_io.audio_buffer_mut(),
                capture_packet.meta().discontinuity,
            );
            linear_io.audio_buffer_mut().merge_frequency_bands();
            let mut linear_chunk = AudioChunk::silence(self.linear_format);
            linear_io.store_chunk(&mut linear_chunk)?;
            ctx.emit(
                linear_out,
                Packet {
                    meta: capture_packet.meta().clone(),
                    payload: linear_chunk,
                },
            )?;
        } else {
            self.echo.process_capture(
                self.capture_io.audio_buffer_mut(),
                capture_packet.meta().discontinuity,
            );
        }

        self.capture_io.audio_buffer_mut().merge_frequency_bands();
        self.capture_io.store_chunk(capture_packet.payload_mut())?;
        let metrics = self.echo.metrics();
        let meta = capture_packet.meta().clone();
        ctx.emit_handle(self.capture_out, capture_packet)?;

        if let Some(port) = self.metrics_out {
            ctx.emit(
                port,
                Packet {
                    meta: meta.clone(),
                    payload: metrics,
                },
            )?;
        }

        self.emit_diagnostic(ctx, meta, "capture_processed")
    }
}
