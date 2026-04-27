use crate::graph::{
    GraphBuilder, GraphResult, InputOptions, NodeControlState, NodeFactory, NodeId, NodeRunner,
    NodeSpec, OutPort, OutputOptions, ProcessCtx, SchedulePlan,
};

use super::audio::{AudioChunk, AudioFormat};
use super::util::validate_audio_format;

#[derive(Debug, Clone, Copy)]
pub struct TapNode {
    node_id: NodeId,
    pub audio_in: crate::graph::InPort<AudioChunk>,
    pub audio_out: OutPort<AudioChunk>,
    pub tap_out: OutPort<AudioChunk>,
}

#[derive(Debug, Clone, Copy)]
pub struct TapNodeBuilder {
    format: AudioFormat,
}

pub fn builder(format: AudioFormat) -> TapNodeBuilder {
    TapNodeBuilder { format }
}

impl TapNodeBuilder {
    pub fn add_to(self, graph: &mut GraphBuilder) -> GraphResult<TapNode> {
        graph.add_node(self)
    }
}

impl TapNode {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

impl NodeSpec for TapNodeBuilder {
    type Handles = TapNode;

    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
        validate_audio_format(self.format, "TapNode")?;

        let node = graph.new_node("tap");
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
        let tap_out = graph.register_output::<AudioChunk>(
            node,
            "tap_out",
            OutputOptions {
                format_key: Some(self.format.schema_key()),
            },
        );
        graph.finish_node(
            node,
            SchedulePlan::OnArrival {
                triggers: vec![audio_in.raw()],
            },
            Box::new(TapFactory {
                audio_in,
                audio_out,
                tap_out,
            }),
        )?;

        Ok(TapNode {
            node_id: node,
            audio_in,
            audio_out,
            tap_out,
        })
    }
}

struct TapFactory {
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
    tap_out: OutPort<AudioChunk>,
}

impl NodeFactory for TapFactory {
    fn describe(&self, _io: &mut crate::graph::NodeIoBuilder<'_>) -> GraphResult<()> {
        Ok(())
    }

    fn build(
        self: Box<Self>,
        _ctx: &mut crate::graph::BuildCtx,
    ) -> GraphResult<Box<dyn NodeRunner>> {
        Ok(Box::new(TapRunner {
            audio_in: self.audio_in,
            audio_out: self.audio_out,
            tap_out: self.tap_out,
        }))
    }
}

struct TapRunner {
    audio_in: crate::graph::InPort<AudioChunk>,
    audio_out: OutPort<AudioChunk>,
    tap_out: OutPort<AudioChunk>,
}

impl NodeRunner for TapRunner {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        let Some(packet) = ctx.take(self.audio_in)? else {
            return Ok(());
        };
        match ctx.control_state() {
            NodeControlState::Active => {}
            NodeControlState::Bypassed => return ctx.emit_handle(self.audio_out, packet),
            NodeControlState::Suspended => return Ok(()),
        }
        ctx.emit_handle(self.audio_out, packet.clone())?;
        ctx.emit_handle(self.tap_out, packet)
    }
}
