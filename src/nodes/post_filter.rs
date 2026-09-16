use crate::audio_processing::post_filter::PostFilter;
use crate::graph::{
    GraphBuilder, GraphError, GraphResult, InputOptions, NodeControlState, NodeFactory, NodeId,
    NodeRunner, NodeSpec, OutPort, OutputOptions, ProcessCtx, SchedulePlan,
};

use super::audio::{AudioChunk, AudioFormat};
use super::util::validate_audio_format;

/// Fullband post-processing filter node, applied at the end of the capture
/// chain. Passes audio through unchanged below 48 kHz, so the graph topology
/// does not depend on the sample rate.
#[derive(Debug, Clone, Copy)]
pub struct PostFilterNode {
    node_id: NodeId,
    pub audio_in: crate::graph::InPort<AudioChunk>,
    pub audio_out: OutPort<AudioChunk>,
}

#[derive(Debug, Clone, Copy)]
pub struct PostFilterNodeBuilder {
    format: AudioFormat,
}

pub fn builder(format: AudioFormat) -> PostFilterNodeBuilder {
    PostFilterNodeBuilder { format }
}

impl PostFilterNodeBuilder {
    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<PostFilterNode> {
        graph.add_node(self)
    }
}

impl PostFilterNode {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

impl NodeSpec for PostFilterNodeBuilder {
    type Handles = PostFilterNode;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        validate_audio_format(self.format, "PostFilterNode")?;

        let node = graph.new_node("post_filter");
        let audio_in = graph.register_input::<AudioChunk>(
            node,
            "audio_in",
            InputOptions {
                format_key: Some(self.format.schema_key()),
                ..InputOptions::default()
            },
        );
        let audio_out = graph.register_output::<AudioChunk>(
            node,
            "audio_out",
            OutputOptions {
                format_key: Some(self.format.schema_key()),
            },
        );
        graph.finish_node(
            node,
            SchedulePlan::OnArrival {
                triggers: vec![audio_in.raw()],
            },
            Box::new(PostFilterFactory {
                format: self.format,
                audio_in,
                audio_out,
            }),
        )?;

        Ok(PostFilterNode {
            node_id: node,
            audio_in,
            audio_out,
        })
    }
}

struct PostFilterFactory {
    format: AudioFormat,
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
}

impl NodeFactory for PostFilterFactory {
    fn build(
        self: Box<Self>,
        _ctx: &mut crate::graph::BuildCtx,
    ) -> GraphResult<Box<dyn NodeRunner>> {
        let channels = self.format.channels as usize;
        let filter = PostFilter::create_if_needed(self.format.sample_rate_hz as i32, channels);
        let scratch = if filter.is_some() {
            (0..channels)
                .map(|_| vec![0.0; self.format.frames_per_channel as usize])
                .collect()
        } else {
            Vec::new()
        };

        Ok(Box::new(PostFilterRunner {
            audio_in: self.audio_in,
            audio_out: self.audio_out,
            format: self.format,
            filter,
            scratch,
        }))
    }
}

struct PostFilterRunner {
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
    format: AudioFormat,
    /// `None` below 48 kHz.
    filter: Option<PostFilter>,
    scratch: Vec<Vec<f32>>,
}

impl NodeRunner for PostFilterRunner {
    fn reset(&mut self) -> GraphResult<()> {
        if let Some(filter) = self.filter.as_mut() {
            filter.reset();
        }
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

        let Some(filter) = self.filter.as_mut() else {
            return ctx.emit_handle(self.audio_out, packet);
        };

        if packet.payload().format != self.format {
            return Err(GraphError::NodeError(format!(
                "PostFilterNode expected {}, got {}",
                self.format.schema_key(),
                packet.payload().format.schema_key()
            )));
        }

        let channels = self.format.channels as usize;
        let samples = packet.payload_mut().samples_mut();

        for (channel_index, slot) in self.scratch.iter_mut().enumerate() {
            for (frame, value) in slot.iter_mut().enumerate() {
                *value = samples[frame * channels + channel_index];
            }
        }
        filter.process(&mut self.scratch);
        for (channel_index, slot) in self.scratch.iter().enumerate() {
            for (frame, value) in slot.iter().enumerate() {
                samples[frame * channels + channel_index] = *value;
            }
        }

        ctx.emit_handle(self.audio_out, packet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Packet, PacketMeta, QueueConfig, Runtime};

    fn run_frame(format: AudioFormat, input: &[f32]) -> Vec<f32> {
        let mut graph = GraphBuilder::new();
        let source = graph.source::<AudioChunk>("audio");
        let sink = graph.sink::<AudioChunk>("output", QueueConfig::audio_default());
        let node = builder(format).add_to(&mut graph).expect("node builds");
        graph.connect(source, node.audio_in).expect("connect in");
        graph.connect(node.audio_out, sink).expect("connect out");

        let mut runtime = Runtime::new(graph.build().expect("graph builds")).expect("runtime");
        runtime
            .push(
                source,
                Packet {
                    meta: PacketMeta::default(),
                    payload: AudioChunk::from_interleaved(format, input),
                },
            )
            .expect("push");
        runtime.run_until_stalled().expect("run");
        let packet = runtime
            .try_pull(sink)
            .expect("pull")
            .expect("a frame is produced");
        packet.payload().samples().to_vec()
    }

    fn tone_frame(format: AudioFormat, frequency_hz: f32) -> Vec<f32> {
        let channels = format.channels as usize;
        let frames = format.frames_per_channel as usize;
        let mut samples = vec![0.0f32; format.sample_count()];
        for frame in 0..frames {
            let value = (2.0 * std::f32::consts::PI * frequency_hz * frame as f32
                / format.sample_rate_hz as f32)
                .sin();
            for channel in 0..channels {
                samples[frame * channels + channel] = value;
            }
        }
        samples
    }

    #[test]
    fn passes_audio_through_below_48k() {
        let format = AudioFormat::ten_ms(16_000, 1);
        let input = tone_frame(format, 1_000.0);
        assert_eq!(input, run_frame(format, &input));
    }

    #[test]
    fn filters_at_48k() {
        let format = AudioFormat::ten_ms(48_000, 1);
        let input = tone_frame(format, 22_000.0);
        let output = run_frame(format, &input);
        assert_ne!(input, output);

        // The tail of the frame is past the filter transient.
        let peak = output[output.len() / 2..]
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!(peak < 0.1, "22 kHz only attenuated to {peak}");
    }

    #[test]
    fn preserves_channel_interleaving() {
        let format = AudioFormat::ten_ms(48_000, 2);
        let channels = format.channels as usize;
        let frames = format.frames_per_channel as usize;

        // Left carries a passband tone, right is silent.
        let mut input = vec![0.0f32; format.sample_count()];
        for frame in 0..frames {
            input[frame * channels] = (2.0 * std::f32::consts::PI * 1_000.0 * frame as f32
                / format.sample_rate_hz as f32)
                .sin();
        }

        let output = run_frame(format, &input);
        assert!(output.iter().step_by(channels).any(|sample| *sample != 0.0));
        assert!(
            output
                .iter()
                .skip(1)
                .step_by(channels)
                .all(|sample| *sample == 0.0),
            "filtering leaked across channels"
        );
    }

    #[test]
    fn bypassed_node_passes_audio_through() {
        let format = AudioFormat::ten_ms(48_000, 1);
        let input = tone_frame(format, 22_000.0);

        let mut graph = GraphBuilder::new();
        let source = graph.source::<AudioChunk>("audio");
        let sink = graph.sink::<AudioChunk>("output", QueueConfig::audio_default());
        let node = builder(format).add_to(&mut graph).expect("node builds");
        graph.connect(source, node.audio_in).expect("connect in");
        graph.connect(node.audio_out, sink).expect("connect out");

        let mut runtime = Runtime::new(graph.build().expect("graph builds")).expect("runtime");
        runtime
            .set_node_state(node.node_id(), NodeControlState::Bypassed)
            .expect("bypass");
        runtime
            .push(
                source,
                Packet {
                    meta: PacketMeta::default(),
                    payload: AudioChunk::from_interleaved(format, &input),
                },
            )
            .expect("push");
        runtime.run_until_stalled().expect("run");
        let packet = runtime
            .try_pull(sink)
            .expect("pull")
            .expect("a frame is produced");
        assert_eq!(input, packet.payload().samples().to_vec());
    }
}
