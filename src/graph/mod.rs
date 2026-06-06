//! Event-driven processing graph: typed ports, packets, queueing, scheduling,
//! and validation.
//!
//! # Threading
//!
//! The runtime is single-threaded by design: build the graph, then drive the
//! resulting [`Runtime`] from one thread via [`Runtime::push`] and
//! [`Runtime::run_until_stalled`]. Packets are reference-counted with
//! [`std::rc::Rc`], so the runtime itself is not `Send`. If audio arrives on
//! another thread (e.g. an audio device callback), transfer frames into the
//! runtime's thread with a channel or ring buffer and call `push` there.
//!
//! # Queues and triggers
//!
//! Every node input has a bounded queue with an [`OverflowPolicy`]. Inputs that
//! act as scheduling triggers additionally record pending trigger events.
//! Until trigger bookkeeping is unified with the queues, a non-`RejectPush`
//! overflow policy on a *trigger* input can desync pending triggers from queue
//! contents (a dropped packet's trigger still fires). The rule is therefore:
//!
//! - A trigger input the node consumes with [`ProcessCtx::take`] must use
//!   [`OverflowPolicy::RejectPush`] (the default), so the queue head always
//!   corresponds to the pending trigger.
//! - A trigger input the node reads *only* through
//!   [`ProcessCtx::trigger_packet`] may use any policy: the trigger event
//!   itself carries the packet, so replacing or dropping queued packets is
//!   harmless. The built-in AEC3 render input uses `latest(1)` this way.
//! - Non-reject policies are otherwise intended for side/control inputs such
//!   as delay hints.
//!
//! # Alignment
//!
//! Side-input alignment is sequence-based: stamp [`PacketMeta::sequence`] on
//! source packets and use [`MatchPolicy::BySequence`] for dependencies that
//! must be derived from the same upstream frame as the trigger. Timestamps
//! ([`PacketMeta::timestamp`]) are carried through the graph untouched but are
//! never interpreted by the runtime.

mod builder;
mod node;
mod packet;
mod port;
mod runtime;

pub use builder::{GraphBuilder, GraphSpec};
pub use node::{
    BuildCtx, Dependency, MatchPolicy, NodeControlState, NodeFactory, NodeRunner, NodeSpec,
    ReadyState, SchedulePlan,
};
pub use packet::{ClockId, Packet, PacketHandle, PacketMeta, Timestamp};
pub use port::{
    AccessMode, InPort, InputOptions, NodeId, OutPort, OutputOptions, OverflowPolicy, PortData,
    QueueConfig, RawInPort, RawOutPort, Sink, Source,
};
pub use runtime::{ProcessCtx, ReadyCtx, Runtime};

use std::fmt;

pub type GraphResult<T> = Result<T, GraphError>;

#[derive(Debug)]
pub enum GraphError {
    UnknownNode(usize),
    UnknownInputPort(usize),
    UnknownOutputPort(usize),
    TypeMismatch {
        from: String,
        to: String,
        from_type: &'static str,
        to_type: &'static str,
    },
    FormatMismatch {
        from: String,
        to: String,
        from_format: String,
        to_format: String,
    },
    MissingSchedulePlan {
        node: String,
    },
    UnfinishedNode {
        node: String,
    },
    QueueFull {
        port: String,
    },
    InvalidQueueCapacity {
        port: String,
    },
    MissingSequenceForAlignment {
        port: String,
    },
    GraphCycle,
    PayloadTypeMismatch {
        expected: &'static str,
    },
    NodeError(String),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::UnknownNode(node) => write!(f, "unknown node id {}", node),
            GraphError::UnknownInputPort(port) => write!(f, "unknown input port id {}", port),
            GraphError::UnknownOutputPort(port) => write!(f, "unknown output port id {}", port),
            GraphError::TypeMismatch {
                from,
                to,
                from_type,
                to_type,
            } => write!(
                f,
                "cannot connect {} ({}) to {} ({}): payload types differ",
                from, from_type, to, to_type
            ),
            GraphError::FormatMismatch {
                from,
                to,
                from_format,
                to_format,
            } => write!(
                f,
                "cannot connect {} ({}) to {} ({}): format keys differ",
                from, from_format, to, to_format
            ),
            GraphError::MissingSchedulePlan { node } => {
                write!(f, "node {} is missing a schedule plan", node)
            }
            GraphError::UnfinishedNode { node } => {
                write!(f, "node {} was registered without a runtime factory", node)
            }
            GraphError::QueueFull { port } => write!(f, "queue for {} is full", port),
            GraphError::InvalidQueueCapacity { port } => {
                write!(f, "queue for {} must have a capacity of at least 1", port)
            }
            GraphError::MissingSequenceForAlignment { port } => write!(
                f,
                "sequence-based alignment on {} requires PacketMeta::sequence to be stamped on \
                 the trigger and all required dependency packets",
                port
            ),
            GraphError::GraphCycle => write!(f, "graph contains a cycle"),
            GraphError::PayloadTypeMismatch { expected } => {
                write!(f, "packet payload type did not match {}", expected)
            }
            GraphError::NodeError(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for GraphError {}
