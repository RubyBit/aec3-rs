//! Typed port handles and per-input queue configuration.

use std::any::TypeId;
use std::marker::PhantomData;

/// Payload types that can travel through the graph.
///
/// The `Send` bound is forward compatibility for a future multi-threaded
/// ingress; the runtime itself is currently single-threaded (see the module
/// docs on [`crate::graph`]).
pub trait PortData: Send + 'static {}
impl<T: Send + 'static> PortData for T {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawInPort(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawOutPort(pub(crate) usize);

#[derive(Debug, PartialEq, Eq)]
pub struct Source<T: PortData> {
    pub(crate) raw: RawOutPort,
    pub(crate) marker: PhantomData<fn() -> T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Sink<T: PortData> {
    pub(crate) raw: RawInPort,
    pub(crate) marker: PhantomData<fn() -> T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct InPort<T: PortData> {
    pub(crate) raw: RawInPort,
    pub(crate) marker: PhantomData<fn() -> T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct OutPort<T: PortData> {
    pub(crate) raw: RawOutPort,
    pub(crate) marker: PhantomData<fn() -> T>,
}

impl<T: PortData> Copy for Source<T> {}
impl<T: PortData> Clone for Source<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PortData> Copy for Sink<T> {}
impl<T: PortData> Clone for Sink<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PortData> Copy for InPort<T> {}
impl<T: PortData> Clone for InPort<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PortData> Copy for OutPort<T> {}
impl<T: PortData> Clone for OutPort<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PortData> Source<T> {
    pub fn raw(self) -> RawOutPort {
        self.raw
    }
}

impl<T: PortData> Sink<T> {
    pub fn raw(self) -> RawInPort {
        self.raw
    }
}

impl<T: PortData> InPort<T> {
    pub fn raw(self) -> RawInPort {
        self.raw
    }
}

impl<T: PortData> OutPort<T> {
    pub fn raw(self) -> RawOutPort {
        self.raw
    }
}

impl<T: PortData> From<Source<T>> for OutPort<T> {
    fn from(value: Source<T>) -> Self {
        Self {
            raw: value.raw,
            marker: PhantomData,
        }
    }
}

impl<T: PortData> From<Sink<T>> for InPort<T> {
    fn from(value: Sink<T>) -> Self {
        Self {
            raw: value.raw,
            marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    RejectPush,
    DropNewest,
    DropOldest,
    ReplaceLatest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Consume,
    PeekLatest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueConfig {
    pub capacity: usize,
    pub overflow: OverflowPolicy,
}

impl QueueConfig {
    pub const fn bounded(capacity: usize) -> Self {
        Self {
            capacity,
            overflow: OverflowPolicy::RejectPush,
        }
    }

    pub const fn latest(capacity: usize) -> Self {
        Self {
            capacity,
            overflow: OverflowPolicy::ReplaceLatest,
        }
    }

    pub const fn audio_default() -> Self {
        Self::bounded(8)
    }
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self::bounded(8)
    }
}

#[derive(Debug, Clone)]
pub struct InputOptions {
    pub queue: QueueConfig,
    pub access: AccessMode,
    pub format_key: Option<String>,
}

impl Default for InputOptions {
    fn default() -> Self {
        Self {
            queue: QueueConfig::default(),
            access: AccessMode::Consume,
            format_key: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OutputOptions {
    pub format_key: Option<String>,
}

#[derive(Clone)]
pub(crate) struct PortDescriptor<T> {
    pub(crate) raw: T,
    pub(crate) owner: PortOwner,
    pub(crate) name: String,
    pub(crate) type_id: TypeId,
    pub(crate) type_name: &'static str,
    pub(crate) format_key: Option<String>,
    pub(crate) queue: Option<QueueConfig>,
    pub(crate) access: Option<AccessMode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PortOwner {
    Source,
    Sink,
    Node(NodeId),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Edge {
    pub(crate) from: RawOutPort,
    pub(crate) to: RawInPort,
}
