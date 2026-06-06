//! Packets, packet metadata, and shared (copy-on-write) packet handles.

use std::any::Any;
use std::rc::Rc;

use super::port::PortData;
use super::{GraphError, GraphResult};

pub type ClockId = u32;

/// Optional media timestamp carried on [`PacketMeta`].
///
/// The runtime treats timestamps as opaque pass-through metadata: they are
/// preserved across nodes but never interpreted for scheduling or alignment.
/// Alignment is sequence-based; see [`PacketMeta::sequence`] and
/// [`crate::graph::MatchPolicy::BySequence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timestamp {
    pub clock: ClockId,
    pub start: u64,
    pub duration: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PacketMeta {
    /// Optional media timestamp. Carried through the graph as opaque metadata;
    /// the runtime does not interpret it.
    pub timestamp: Option<Timestamp>,
    /// Monotonic packet counter used by
    /// [`MatchPolicy::BySequence`](crate::graph::MatchPolicy::BySequence)
    /// alignment.
    ///
    /// `None` means "not stamped" and never matches anything. Packets that fan
    /// out from one upstream packet share its sequence, which is exactly the
    /// invariant side-channel alignment needs ("derived from the same frame").
    pub sequence: Option<u64>,
    pub discontinuity: bool,
}

#[derive(Debug, Clone)]
pub struct Packet<T: PortData> {
    pub meta: PacketMeta,
    pub payload: T,
}

#[derive(Debug, Clone)]
pub struct PacketHandle<T: PortData> {
    pub(crate) meta: PacketMeta,
    pub(crate) payload: Rc<T>,
}

impl<T: PortData> PacketHandle<T> {
    pub fn meta(&self) -> &PacketMeta {
        &self.meta
    }

    pub fn payload(&self) -> &T {
        &self.payload
    }

    pub fn meta_mut(&mut self) -> &mut PacketMeta {
        &mut self.meta
    }

    pub fn into_payload(self) -> Rc<T> {
        self.payload
    }
}

impl<T: PortData + Clone> PacketHandle<T> {
    pub fn payload_mut(&mut self) -> &mut T {
        Rc::make_mut(&mut self.payload)
    }

    pub fn into_packet(self) -> Packet<T> {
        Packet {
            meta: self.meta,
            payload: match Rc::try_unwrap(self.payload) {
                Ok(payload) => payload,
                Err(payload) => (*payload).clone(),
            },
        }
    }

    pub fn to_packet(&self) -> Packet<T> {
        Packet {
            meta: self.meta.clone(),
            payload: (*self.payload).clone(),
        }
    }
}

/// Type-erased packet stored in port queues and trigger events.
pub(crate) struct ErasedPacket {
    pub(crate) meta: PacketMeta,
    pub(crate) payload: Rc<dyn Any>,
}

pub(crate) fn downcast_handle<T: PortData>(
    erased: Rc<ErasedPacket>,
    expected: &'static str,
) -> GraphResult<Option<PacketHandle<T>>> {
    match erased.payload.clone().downcast::<T>() {
        Ok(payload) => Ok(Some(PacketHandle {
            meta: erased.meta.clone(),
            payload,
        })),
        Err(_) => Err(GraphError::PayloadTypeMismatch { expected }),
    }
}
