//! Node traits, scheduling plans, and side-input match policies.

use super::GraphResult;
use super::builder::GraphBuilder;
use super::port::{NodeId, RawInPort};
use super::runtime::{ProcessCtx, ReadyCtx};

/// How a dependency packet is matched against a trigger packet when a node is
/// scheduled with [`SchedulePlan::AlignOn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchPolicy {
    /// Match the next packet in queue order. Requires no metadata.
    ///
    /// Only correct when the dependency stream produces exactly one packet per
    /// trigger packet, in order, with nothing dropped along the way.
    Fifo,
    /// Match the dependency packet whose [`PacketMeta::sequence`] equals the
    /// trigger packet's sequence. Sequences must be monotonically increasing
    /// per stream.
    ///
    /// Matching is strict: both sides must be stamped with `Some(sequence)`.
    /// For a *required* dependency, an unstamped packet on either side is a
    /// [`GraphError::MissingSequenceForAlignment`] because it can never become
    /// matchable. A stamped trigger with no matching dependency packet yet
    /// simply waits. Optional dependencies proceed without a match, and reads
    /// on an unmatched dependency port return nothing rather than falling back
    /// to an arbitrary queued packet.
    ///
    /// Dependency packets with sequences *older* than the pending trigger are
    /// pruned during matching — monotonicity means they can never match the
    /// current or any future trigger — so persistent mismatches cannot fill
    /// the dependency queue.
    ///
    /// Triggers are processed in order: a required dependency that never
    /// arrives stalls that node's trigger queue (head-of-line blocking). A
    /// trigger backlog cap that turns a permanent stall into an error is
    /// planned.
    ///
    /// [`PacketMeta::sequence`]: crate::graph::PacketMeta::sequence
    /// [`GraphError::MissingSequenceForAlignment`]: crate::graph::GraphError::MissingSequenceForAlignment
    BySequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dependency {
    pub input: RawInPort,
    pub required: bool,
    pub match_policy: MatchPolicy,
}

#[derive(Debug, Clone)]
pub enum SchedulePlan {
    OnArrival {
        triggers: Vec<RawInPort>,
    },
    AlignOn {
        trigger: RawInPort,
        deps: Vec<Dependency>,
    },
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    Ready,
    NotReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeControlState {
    #[default]
    Active,
    Bypassed,
    Suspended,
}

/// Registers a node's ports and schedule on a [`GraphBuilder`] and returns the
/// typed handles callers use for wiring.
pub trait NodeSpec {
    type Handles;
    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles>;
}

/// Builds the runtime state ([`NodeRunner`]) for a registered node.
pub trait NodeFactory: 'static {
    fn build(self: Box<Self>, ctx: &mut BuildCtx) -> GraphResult<Box<dyn NodeRunner>>;
}

pub trait NodeRunner {
    fn ready(&self, _ctx: &ReadyCtx<'_>) -> GraphResult<ReadyState> {
        Ok(ReadyState::Ready)
    }

    fn reset(&mut self) -> GraphResult<()> {
        Ok(())
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()>;
}

pub struct BuildCtx {
    pub(crate) node_id: NodeId,
}

impl BuildCtx {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

pub(crate) struct NoopRunner;

impl NodeRunner for NoopRunner {
    fn process(&mut self, _ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        Ok(())
    }
}
