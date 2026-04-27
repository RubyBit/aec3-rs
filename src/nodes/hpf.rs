use crate::audio_processing::audio_buffer::AudioBuffer;
use crate::audio_processing::high_pass_filter::HighPassFilter;
use crate::graph::{
    GraphBuilder, GraphResult, InputOptions, NodeControlState, NodeFactory, NodeId, NodeRunner,
    NodeSpec, OutPort, OutputOptions, ProcessCtx, SchedulePlan,
};

use super::audio::{AudioChunk, AudioFormat};
use super::util::{ChunkIo, internal_sample_rate, validate_audio_format};

#[derive(Debug, Clone, Copy)]
pub struct HighPassNode {
    node_id: NodeId,
    pub audio_in: crate::graph::InPort<AudioChunk>,
    pub audio_out: OutPort<AudioChunk>,
}

#[derive(Debug, Clone, Copy)]
pub struct HighPassNodeBuilder {
    format: AudioFormat,
}

pub fn builder(format: AudioFormat) -> HighPassNodeBuilder {
    HighPassNodeBuilder { format }
}

impl HighPassNodeBuilder {
    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<HighPassNode> {
        graph.add_node(self)
    }
}

impl HighPassNode {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

impl NodeSpec for HighPassNodeBuilder {
    type Handles = HighPassNode;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        validate_audio_format(self.format, "HighPassNode")?;

        let node = graph.new_node("high_pass");
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
            Box::new(HighPassFactory {
                format: self.format,
                audio_in,
                audio_out,
            }),
        )?;

        Ok(HighPassNode {
            node_id: node,
            audio_in,
            audio_out,
        })
    }
}

struct HighPassFactory {
    format: AudioFormat,
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
}

impl NodeFactory for HighPassFactory {
    fn describe(&self, _io: &mut crate::graph::NodeIoBuilder<'_>) -> GraphResult<()> {
        Ok(())
    }

    fn build(
        self: Box<Self>,
        _ctx: &mut crate::graph::BuildCtx,
    ) -> GraphResult<Box<dyn NodeRunner>> {
        let internal_rate = internal_sample_rate(self.format, None);
        Ok(Box::new(HighPassRunner {
            audio_in: self.audio_in,
            audio_out: self.audio_out,
            io: ChunkIo::new(self.format, internal_rate),
            scratch: (0..self.format.channels as usize)
                .map(|_| vec![0.0; AudioBuffer::SPLIT_BAND_SIZE])
                .collect(),
            filter: HighPassFilter::new(internal_rate as i32, self.format.channels as usize),
        }))
    }
}

struct HighPassRunner {
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
    io: ChunkIo,
    scratch: Vec<Vec<f32>>,
    filter: HighPassFilter,
}

impl NodeRunner for HighPassRunner {
    fn reset(&mut self) -> GraphResult<()> {
        self.filter.reset();
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

        self.io.load_chunk(packet.payload(), "HighPassNode")?;
        self.io.audio_buffer_mut().split_into_frequency_bands();
        for (channel_index, slot) in self.scratch.iter_mut().enumerate() {
            let src = self.io.audio_buffer().split_band(channel_index, 0);
            slot[..src.len()].copy_from_slice(src);
        }
        self.filter.process(&mut self.scratch);
        for (channel_index, slot) in self.scratch.iter().enumerate() {
            let dst = self.io.audio_buffer_mut().split_band_mut(channel_index, 0);
            dst.copy_from_slice(slot);
        }
        self.io.audio_buffer_mut().merge_frequency_bands();
        self.io.store_chunk(packet.payload_mut())?;
        ctx.emit_handle(self.audio_out, packet)
    }
}
