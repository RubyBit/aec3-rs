use crate::graph::{
    GraphBuilder, GraphResult, InputOptions, NodeControlState, NodeFactory, NodeId, NodeRunner,
    NodeSpec, OutPort, OutputOptions, ProcessCtx, SchedulePlan,
};

use super::audio::{AudioChunk, AudioFormat};
use super::util::{AudioAdapter, validate_audio_format};

#[derive(Debug, Clone, Copy)]
pub struct ResampleNode {
    node_id: NodeId,
    pub audio_in: crate::graph::InPort<AudioChunk>,
    pub audio_out: OutPort<AudioChunk>,
}

#[derive(Debug, Clone, Copy)]
pub struct ResampleNodeBuilder {
    input_format: AudioFormat,
    output_format: AudioFormat,
}

pub fn builder(input_format: AudioFormat, output_format: AudioFormat) -> ResampleNodeBuilder {
    ResampleNodeBuilder {
        input_format,
        output_format,
    }
}

impl ResampleNodeBuilder {
    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<ResampleNode> {
        graph.add_node(self)
    }
}

impl ResampleNode {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

impl NodeSpec for ResampleNodeBuilder {
    type Handles = ResampleNode;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        validate_audio_format(self.input_format, "ResampleNode")?;
        validate_audio_format(self.output_format, "ResampleNode")?;

        let node = graph.new_node("resample");
        let audio_in = graph.register_input::<AudioChunk>(
            node,
            "audio_in",
            InputOptions {
                format_key: Some(self.input_format.schema_key()),
                ..InputOptions::default()
            },
        );
        let audio_out = graph.register_output::<AudioChunk>(
            node,
            "audio_out",
            OutputOptions {
                format_key: Some(self.output_format.schema_key()),
            },
        );
        graph.finish_node(
            node,
            SchedulePlan::OnArrival {
                triggers: vec![audio_in.raw()],
            },
            Box::new(ResampleFactory {
                input_format: self.input_format,
                output_format: self.output_format,
                audio_in,
                audio_out,
            }),
        )?;

        Ok(ResampleNode {
            node_id: node,
            audio_in,
            audio_out,
        })
    }
}

struct ResampleFactory {
    input_format: AudioFormat,
    output_format: AudioFormat,
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
}

impl NodeFactory for ResampleFactory {
    fn describe(&self, _io: &mut crate::graph::NodeIoBuilder<'_>) -> GraphResult<()> {
        Ok(())
    }

    fn build(self: Box<Self>, _ctx: &mut crate::graph::BuildCtx) -> GraphResult<Box<dyn NodeRunner>> {
        Ok(Box::new(ResampleRunner {
            input_format: self.input_format,
            audio_in: self.audio_in,
            audio_out: self.audio_out,
            adapter: AudioAdapter::new(self.input_format, self.output_format),
            output_format: self.output_format,
        }))
    }
}

struct ResampleRunner {
    input_format: AudioFormat,
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
    adapter: AudioAdapter,
    output_format: AudioFormat,
}

impl NodeRunner for ResampleRunner {
    fn reset(&mut self) -> GraphResult<()> {
        self.adapter = AudioAdapter::new(self.input_format, self.output_format);
        Ok(())
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        let Some(packet) = ctx.take(self.audio_in)? else {
            return Ok(());
        };

        match ctx.control_state() {
            NodeControlState::Active => {}
            NodeControlState::Bypassed if self.input_format == self.output_format => {
                return ctx.emit_handle(self.audio_out, packet);
            }
            NodeControlState::Suspended => return Ok(()),
            NodeControlState::Bypassed => {}
        }

        let mut output = AudioChunk::silence(self.output_format);
        self.adapter.process(packet.payload(), &mut output)?;
        ctx.emit(
            self.audio_out,
            crate::graph::Packet {
                meta: packet.meta().clone(),
                payload: output,
            },
        )
    }
}
