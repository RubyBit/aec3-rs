use std::any::{Any, TypeId, type_name};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

pub type ClockId = u32;
pub type GraphResult<T> = Result<T, GraphError>;

pub trait PortData: Send + 'static {}
impl<T: Send + 'static> PortData for T {}

pub trait ReusablePortData: PortData {
    type PoolKey: Clone + Eq + std::hash::Hash + Send + 'static;
    fn pool_key(&self) -> Self::PoolKey;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawInPort(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawOutPort(usize);

#[derive(Debug, PartialEq, Eq)]
pub struct Source<T: PortData> {
    raw: RawOutPort,
    marker: PhantomData<fn() -> T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Sink<T: PortData> {
    raw: RawInPort,
    marker: PhantomData<fn() -> T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct InPort<T: PortData> {
    raw: RawInPort,
    marker: PhantomData<fn() -> T>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct OutPort<T: PortData> {
    raw: RawOutPort,
    marker: PhantomData<fn() -> T>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchPolicy {
    AnyAvailable,
    ExactTimestamp,
    WithinSkew { ticks: u32 },
    LatestBefore { max_age: Option<u32> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dependency {
    pub input: RawInPort,
    pub required: bool,
    pub match_policy: MatchPolicy,
}

#[derive(Debug, Clone)]
pub enum SchedulePlan {
    OnArrival { triggers: Vec<RawInPort> },
    AlignOn { trigger: RawInPort, deps: Vec<Dependency> },
    Custom,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timestamp {
    pub clock: ClockId,
    pub start: u64,
    pub duration: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PacketMeta {
    pub timestamp: Option<Timestamp>,
    pub sequence: u64,
    pub discontinuity: bool,
}

#[derive(Debug, Clone)]
pub struct Packet<T: PortData> {
    pub meta: PacketMeta,
    pub payload: T,
}

#[derive(Debug, Clone)]
pub struct PacketHandle<T: PortData> {
    meta: PacketMeta,
    payload: Rc<T>,
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

pub trait NodeSpec {
    type Handles;
    fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles>;
}

pub trait NodeFactory: 'static {
    fn describe(&self, io: &mut NodeIoBuilder<'_>) -> GraphResult<()>;
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
    node_id: NodeId,
}

impl BuildCtx {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }
}

pub struct ReadyCtx<'a> {
    runtime: &'a Runtime,
    node_id: NodeId,
}

impl<'a> ReadyCtx<'a> {
    pub fn queued(&self, port: RawInPort) -> usize {
        self.runtime.queue_len(port)
    }

    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    pub fn control_state(&self) -> NodeControlState {
        self.runtime.node_state(self.node_id).unwrap_or(NodeControlState::Active)
    }
}

pub struct ProcessCtx<'a> {
    runtime: &'a mut Runtime,
    node_id: NodeId,
    control_state: NodeControlState,
    current_trigger: Option<RawInPort>,
    trigger_packet: Option<Rc<ErasedPacket>>,
    matched: HashMap<RawInPort, usize>,
}

impl<'a> ProcessCtx<'a> {
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    pub fn control_state(&self) -> NodeControlState {
        self.control_state
    }

    pub fn trigger(&self) -> Option<RawInPort> {
        self.current_trigger
    }

    pub fn trigger_is<T: PortData>(&self, port: InPort<T>) -> bool {
        self.current_trigger == Some(port.raw())
    }

    pub fn trigger_packet<T: PortData>(
        &self,
        port: InPort<T>,
    ) -> GraphResult<Option<PacketHandle<T>>> {
        if self.current_trigger != Some(port.raw()) {
            return Ok(None);
        }
        match self.trigger_packet.clone() {
            Some(packet) => downcast_handle(packet, "trigger packet"),
            None => Ok(None),
        }
    }

    pub fn take<T: PortData>(&mut self, port: InPort<T>) -> GraphResult<Option<PacketHandle<T>>> {
        self.runtime.take_from_input(port.raw, self.matched.get(&port.raw).copied())
    }

    pub fn peek<T: PortData>(&self, port: InPort<T>) -> GraphResult<Option<PacketHandle<T>>> {
        self.runtime.peek_input(port.raw, self.matched.get(&port.raw).copied())
    }

    pub fn emit<T: PortData>(&mut self, port: OutPort<T>, packet: Packet<T>) -> GraphResult<()> {
        self.runtime.emit_packet(port.raw, packet)
    }

    pub fn emit_handle<T: PortData>(
        &mut self,
        port: OutPort<T>,
        packet: PacketHandle<T>,
    ) -> GraphResult<()> {
        self.runtime.emit_handle(port.raw, packet)
    }
}

pub struct NodeIoBuilder<'a> {
    builder: &'a mut GraphBuilder,
    node_id: NodeId,
    schedule: Option<SchedulePlan>,
}

impl<'a> NodeIoBuilder<'a> {
    pub fn input<T: PortData>(&mut self, name: &str, options: InputOptions) -> InPort<T> {
        self.builder.register_input::<T>(self.node_id, name, options)
    }

    pub fn output<T: PortData>(&mut self, name: &str, options: OutputOptions) -> OutPort<T> {
        self.builder.register_output::<T>(self.node_id, name, options)
    }

    pub fn schedule_plan(&mut self, plan: SchedulePlan) {
        self.schedule = Some(plan);
    }
}

#[derive(Default)]
pub struct GraphBuilder {
    next_node_id: usize,
    next_in_port_id: usize,
    next_out_port_id: usize,
    nodes: Vec<PendingNode>,
    input_ports: Vec<PortDescriptor<RawInPort>>,
    output_ports: Vec<PortDescriptor<RawOutPort>>,
    edges: Vec<Edge>,
}

pub struct GraphSpec {
    nodes: Vec<NodeRecord>,
    input_ports: Vec<PortDescriptor<RawInPort>>,
    output_edges: Vec<Vec<RawInPort>>,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeOptions;

pub struct Runtime {
    spec: GraphSpec,
    nodes: Vec<RuntimeNode>,
    input_queues: Vec<Option<PortQueue>>,
    dirty_nodes: VecDeque<NodeId>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn source<T: PortData>(&mut self, name: &str, _queue: QueueConfig) -> Source<T> {
        let raw = RawOutPort(self.next_out_port_id);
        self.next_out_port_id += 1;
        self.output_ports.push(PortDescriptor {
            raw,
            owner: PortOwner::Source,
            name: name.to_string(),
            type_id: TypeId::of::<T>(),
            type_name: type_name::<T>(),
            format_key: None,
            queue: None,
            access: None,
        });
        Source {
            raw,
            marker: PhantomData,
        }
    }

    pub fn sink<T: PortData>(&mut self, name: &str, queue: QueueConfig) -> Sink<T> {
        let raw = self.register_shadow_sink(name, queue, type_name::<T>(), TypeId::of::<T>());
        Sink {
            raw,
            marker: PhantomData,
        }
    }

    pub fn add_node<N: NodeSpec>(&mut self, spec: N) -> GraphResult<N::Handles> {
        spec.register(self)
    }

    pub fn add_factory<F: NodeFactory>(&mut self, name: &str, factory: F) -> GraphResult<NodeId> {
        let node_id = self.new_node(name);
        let mut io = NodeIoBuilder {
            builder: self,
            node_id,
            schedule: None,
        };
        factory.describe(&mut io)?;
        let schedule = io.schedule.take().ok_or_else(|| GraphError::MissingSchedulePlan {
            node: name.to_string(),
        })?;
        io.builder.finish_node(node_id, schedule, Box::new(factory))?;
        Ok(node_id)
    }

    pub fn connect<T: PortData>(
        &mut self,
        from: impl Into<OutPort<T>>,
        to: impl Into<InPort<T>>,
    ) -> GraphResult<()> {
        let from = from.into().raw;
        let to = to.into().raw;
        let from_desc = self
            .output_ports
            .get(from.0)
            .ok_or(GraphError::UnknownOutputPort(from.0))?;
        let to_desc = self
            .input_ports
            .get(to.0)
            .ok_or(GraphError::UnknownInputPort(to.0))?;

        if from_desc.type_id != to_desc.type_id {
            return Err(GraphError::TypeMismatch {
                from: from_desc.name.clone(),
                to: to_desc.name.clone(),
                from_type: from_desc.type_name,
                to_type: to_desc.type_name,
            });
        }

        if let (Some(from_key), Some(to_key)) = (&from_desc.format_key, &to_desc.format_key)
            && from_key != to_key
        {
            return Err(GraphError::FormatMismatch {
                from: from_desc.name.clone(),
                to: to_desc.name.clone(),
                from_format: from_key.clone(),
                to_format: to_key.clone(),
            });
        }

        self.edges.push(Edge { from, to });
        Ok(())
    }

    pub fn build(self) -> GraphResult<GraphSpec> {
        for node in &self.nodes {
            if node.factory.is_none() {
                return Err(GraphError::UnfinishedNode {
                    node: node.name.clone(),
                });
            }
            if node.schedule.is_none() {
                return Err(GraphError::MissingSchedulePlan {
                    node: node.name.clone(),
                });
            }
        }

        validate_cycles(&self.nodes, &self.edges, &self.input_ports, &self.output_ports)?;

        let mut output_edges = vec![Vec::new(); self.output_ports.len()];
        for edge in &self.edges {
            output_edges[edge.from.0].push(edge.to);
        }

        let nodes = self
            .nodes
            .into_iter()
            .map(|node| NodeRecord {
                id: node.id,
                schedule: node.schedule.expect("validated missing schedule"),
                factory: Some(node.factory.expect("validated missing factory")),
            })
            .collect();

        Ok(GraphSpec {
            nodes,
            input_ports: self.input_ports,
            output_edges,
        })
    }

    pub fn new_node(&mut self, name: &str) -> NodeId {
        let node_id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(PendingNode {
            id: node_id,
            name: name.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            schedule: None,
            factory: None,
        });
        node_id
    }

    pub fn register_input<T: PortData>(
        &mut self,
        node_id: NodeId,
        name: &str,
        options: InputOptions,
    ) -> InPort<T> {
        let raw = RawInPort(self.next_in_port_id);
        self.next_in_port_id += 1;
        self.input_ports.push(PortDescriptor {
            raw,
            owner: PortOwner::Node(node_id),
            name: format!("{}::{}", self.nodes[node_id.0].name, name),
            type_id: TypeId::of::<T>(),
            type_name: type_name::<T>(),
            format_key: options.format_key,
            queue: Some(options.queue),
            access: Some(options.access),
        });
        self.nodes[node_id.0].inputs.push(raw);
        InPort {
            raw,
            marker: PhantomData,
        }
    }

    pub fn register_output<T: PortData>(
        &mut self,
        node_id: NodeId,
        name: &str,
        options: OutputOptions,
    ) -> OutPort<T> {
        let raw = RawOutPort(self.next_out_port_id);
        self.next_out_port_id += 1;
        self.output_ports.push(PortDescriptor {
            raw,
            owner: PortOwner::Node(node_id),
            name: format!("{}::{}", self.nodes[node_id.0].name, name),
            type_id: TypeId::of::<T>(),
            type_name: type_name::<T>(),
            format_key: options.format_key,
            queue: None,
            access: None,
        });
        self.nodes[node_id.0].outputs.push(raw);
        OutPort {
            raw,
            marker: PhantomData,
        }
    }

    pub fn finish_node(
        &mut self,
        node_id: NodeId,
        schedule: SchedulePlan,
        factory: Box<dyn NodeFactory>,
    ) -> GraphResult<()> {
        let node = self
            .nodes
            .get_mut(node_id.0)
            .ok_or(GraphError::UnknownNode(node_id.0))?;
        node.schedule = Some(schedule);
        node.factory = Some(factory);
        Ok(())
    }

    fn register_shadow_sink(
        &mut self,
        name: &str,
        queue: QueueConfig,
        type_name: &'static str,
        type_id: TypeId,
    ) -> RawInPort {
        let raw = RawInPort(self.next_in_port_id);
        self.next_in_port_id += 1;
        self.input_ports.push(PortDescriptor {
            raw,
            owner: PortOwner::Sink,
            name: name.to_string(),
            type_id,
            type_name,
            format_key: None,
            queue: Some(queue),
            access: Some(AccessMode::Consume),
        });
        raw
    }
}

impl Runtime {
    pub fn new(mut spec: GraphSpec, _options: RuntimeOptions) -> GraphResult<Self> {
        let mut nodes = Vec::with_capacity(spec.nodes.len());
        for node in &mut spec.nodes {
            let mut ctx = BuildCtx { node_id: node.id };
            let runner = node
                .factory
                .take()
                .expect("runtime construction should only happen once")
                .build(&mut ctx)?;
            nodes.push(RuntimeNode {
                runner,
                pending_triggers: VecDeque::new(),
                control_state: NodeControlState::Active,
                dirty: false,
            });
        }

        let mut input_queues = (0..spec.input_ports.len()).map(|_| None).collect::<Vec<_>>();
        for port in &spec.input_ports {
            input_queues[port.raw.0] = Some(PortQueue {
                items: VecDeque::new(),
                config: port.queue.clone().expect("input ports always have queues"),
                access: port.access.expect("input ports always have access"),
            });
        }

        Ok(Self {
            spec,
            nodes,
            input_queues,
            dirty_nodes: VecDeque::new(),
        })
    }

    pub fn push<T: PortData>(&mut self, src: Source<T>, packet: Packet<T>) -> GraphResult<()> {
        self.emit_packet(src.raw, packet)
    }

    pub fn try_pull<T: PortData>(&mut self, sink: Sink<T>) -> GraphResult<Option<PacketHandle<T>>> {
        self.take_from_input(sink.raw, None)
    }

    pub fn set_node_state(
        &mut self,
        node_id: NodeId,
        state: NodeControlState,
    ) -> GraphResult<()> {
        let node = self
            .nodes
            .get_mut(node_id.0)
            .ok_or(GraphError::UnknownNode(node_id.0))?;
        node.control_state = state;
        Ok(())
    }

    pub fn node_state(&self, node_id: NodeId) -> GraphResult<NodeControlState> {
        self.nodes
            .get(node_id.0)
            .map(|node| node.control_state)
            .ok_or(GraphError::UnknownNode(node_id.0))
    }

    pub fn reset_node(&mut self, node_id: NodeId) -> GraphResult<()> {
        let node = self
            .nodes
            .get_mut(node_id.0)
            .ok_or(GraphError::UnknownNode(node_id.0))?;
        node.runner.reset()
    }

    pub fn run_until_stalled(&mut self) -> GraphResult<usize> {
        let mut processed = 0usize;
        while let Some(node_id) = self.dirty_nodes.pop_front() {
            let node_index = node_id.0;
            if !self.nodes[node_index].dirty {
                continue;
            }
            self.nodes[node_index].dirty = false;

            let execution = match self.execution_for(node_id)? {
                Some(execution) => execution,
                None => continue,
            };
            if execution.trigger.is_some() {
                let _ = self.nodes[node_index].pending_triggers.pop_front();
            }

            let control_state = self.nodes[node_index].control_state;
            let mut runner =
                std::mem::replace(&mut self.nodes[node_index].runner, Box::new(NoopRunner));
            let mut ctx = ProcessCtx {
                runtime: self,
                node_id,
                control_state,
                current_trigger: execution.trigger,
                trigger_packet: execution.trigger_packet,
                matched: execution.matched,
            };
            runner.process(&mut ctx)?;
            self.nodes[node_index].runner = runner;
            processed += 1;

            if self.execution_for(node_id)?.is_some() {
                self.mark_dirty(node_id);
            }
        }
        Ok(processed)
    }

    fn execution_for(&self, node_id: NodeId) -> GraphResult<Option<ExecutionContext>> {
        let node = &self.spec.nodes[node_id.0];
        match &node.schedule {
            SchedulePlan::OnArrival { .. } => {
                if let Some(trigger) = self.nodes[node_id.0].pending_triggers.front() {
                    Ok(Some(ExecutionContext {
                        trigger: Some(trigger.port),
                        trigger_packet: Some(trigger.packet.clone()),
                        matched: HashMap::new(),
                    }))
                } else {
                    Ok(None)
                }
            }
            SchedulePlan::AlignOn { trigger, deps } => {
                let Some(trigger_event) = self.nodes[node_id.0].pending_triggers.front() else {
                    return Ok(None);
                };
                if trigger_event.port != *trigger {
                    return Ok(None);
                }

                let mut matched = HashMap::new();
                for dep in deps {
                    let candidate = self.find_match(
                        dep.input,
                        trigger_event.packet.meta.timestamp.as_ref(),
                        dep.match_policy,
                    )?;
                    match candidate {
                        Some(index) => {
                            matched.insert(dep.input, index);
                        }
                        None if dep.required => return Ok(None),
                        None => {}
                    }
                }

                Ok(Some(ExecutionContext {
                    trigger: Some(*trigger),
                    trigger_packet: Some(trigger_event.packet.clone()),
                    matched,
                }))
            }
            SchedulePlan::Custom => {
                let ready = self.nodes[node_id.0]
                    .runner
                    .ready(&ReadyCtx {
                        runtime: self,
                        node_id,
                    })?;
                if ready == ReadyState::Ready {
                    Ok(Some(ExecutionContext {
                        trigger: None,
                        trigger_packet: None,
                        matched: HashMap::new(),
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn queue_len(&self, port: RawInPort) -> usize {
        self.input_queues
            .get(port.0)
            .and_then(|queue| queue.as_ref())
            .map_or(0, |queue| queue.items.len())
    }

    fn take_from_input<T: PortData>(
        &mut self,
        port: RawInPort,
        index: Option<usize>,
    ) -> GraphResult<Option<PacketHandle<T>>> {
        let queue = self
            .input_queues
            .get_mut(port.0)
            .and_then(|queue| queue.as_mut())
            .ok_or(GraphError::UnknownInputPort(port.0))?;
        let idx = index.unwrap_or_else(|| match queue.access {
            AccessMode::Consume => 0,
            AccessMode::PeekLatest => queue.items.len().saturating_sub(1),
        });
        let erased = match queue.items.remove(idx) {
            Some(packet) => packet,
            None => return Ok(None),
        };
        downcast_handle(erased, self.spec.input_ports[port.0].type_name)
    }

    fn peek_input<T: PortData>(
        &self,
        port: RawInPort,
        index: Option<usize>,
    ) -> GraphResult<Option<PacketHandle<T>>> {
        let queue = self
            .input_queues
            .get(port.0)
            .and_then(|queue| queue.as_ref())
            .ok_or(GraphError::UnknownInputPort(port.0))?;
        let idx = index.unwrap_or_else(|| match queue.access {
            AccessMode::Consume => 0,
            AccessMode::PeekLatest => queue.items.len().saturating_sub(1),
        });
        let erased = match queue.items.get(idx) {
            Some(packet) => packet.clone(),
            None => return Ok(None),
        };
        downcast_handle(erased, self.spec.input_ports[port.0].type_name)
    }

    fn emit_packet<T: PortData>(&mut self, port: RawOutPort, packet: Packet<T>) -> GraphResult<()> {
        let erased = Rc::new(ErasedPacket {
            meta: packet.meta,
            payload: Rc::new(packet.payload),
        });
        self.dispatch(port, erased)
    }

    fn emit_handle<T: PortData>(
        &mut self,
        port: RawOutPort,
        packet: PacketHandle<T>,
    ) -> GraphResult<()> {
        let erased = Rc::new(ErasedPacket {
            meta: packet.meta,
            payload: packet.payload,
        });
        self.dispatch(port, erased)
    }

    fn dispatch(&mut self, port: RawOutPort, packet: Rc<ErasedPacket>) -> GraphResult<()> {
        let targets = self
            .spec
            .output_edges
            .get(port.0)
            .ok_or(GraphError::UnknownOutputPort(port.0))?
            .clone();

        for target in targets {
            self.enqueue_input(target, packet.clone())?;
        }
        Ok(())
    }

    fn enqueue_input(&mut self, port: RawInPort, packet: Rc<ErasedPacket>) -> GraphResult<()> {
        let owner = self.spec.input_ports[port.0].owner;
        let queue = self
            .input_queues
            .get_mut(port.0)
            .and_then(|queue| queue.as_mut())
            .ok_or(GraphError::UnknownInputPort(port.0))?;
        apply_overflow_policy(queue, packet.clone())?;

        if let PortOwner::Node(node_id) = owner {
            let schedule = &self.spec.nodes[node_id.0].schedule;
            match schedule {
                SchedulePlan::OnArrival { triggers } if triggers.contains(&port) => {
                    self.nodes[node_id.0]
                        .pending_triggers
                        .push_back(TriggerEvent { port, packet });
                    self.mark_dirty(node_id);
                }
                SchedulePlan::AlignOn { trigger, .. } if *trigger == port => {
                    self.nodes[node_id.0]
                        .pending_triggers
                        .push_back(TriggerEvent { port, packet });
                    self.mark_dirty(node_id);
                }
                SchedulePlan::AlignOn { deps, .. }
                    if deps.iter().any(|dependency| dependency.input == port) =>
                {
                    self.mark_dirty(node_id);
                }
                SchedulePlan::Custom => self.mark_dirty(node_id),
                _ => {}
            }
        }

        Ok(())
    }

    fn mark_dirty(&mut self, node_id: NodeId) {
        if !self.nodes[node_id.0].dirty {
            self.nodes[node_id.0].dirty = true;
            self.dirty_nodes.push_back(node_id);
        }
    }

    fn find_match(
        &self,
        port: RawInPort,
        trigger_timestamp: Option<&Timestamp>,
        policy: MatchPolicy,
    ) -> GraphResult<Option<usize>> {
        let queue = self
            .input_queues
            .get(port.0)
            .and_then(|queue| queue.as_ref())
            .ok_or(GraphError::UnknownInputPort(port.0))?;

        if trigger_timestamp.is_none() {
            return Ok(default_queue_index(queue));
        }
        let trigger_timestamp = trigger_timestamp.expect("checked above");

        match policy {
            MatchPolicy::AnyAvailable => Ok(default_queue_index(queue)),
            MatchPolicy::ExactTimestamp => Ok(
                find_matching_timestamp(queue, trigger_timestamp, 0, true)
                    .or_else(|| fallback_if_missing_timestamps(queue)),
            ),
            MatchPolicy::WithinSkew { ticks } => Ok(
                find_matching_timestamp(queue, trigger_timestamp, ticks, false)
                    .or_else(|| fallback_if_missing_timestamps(queue)),
            ),
            MatchPolicy::LatestBefore { max_age } => {
                let mut best = None;
                let mut missing_timestamps = false;
                for (index, packet) in queue.items.iter().enumerate() {
                    let Some(timestamp) = packet.meta.timestamp.as_ref() else {
                        missing_timestamps = true;
                        continue;
                    };
                    if timestamp.clock != trigger_timestamp.clock || timestamp.start > trigger_timestamp.start {
                        continue;
                    }
                    let age = trigger_timestamp.start - timestamp.start;
                    if max_age.is_some_and(|limit| age > limit as u64) {
                        continue;
                    }
                    best = Some(index);
                }
                Ok(best.or_else(|| missing_timestamps.then(|| default_queue_index(queue)).flatten()))
            }
        }
    }

}

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
    MissingTimestampForAlignment {
        port: String,
    },
    CrossClockAlignment {
        trigger_clock: ClockId,
        dependency_clock: ClockId,
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
            GraphError::MissingTimestampForAlignment { port } => write!(
                f,
                "alignment requires timestamps, but {} received a packet without one",
                port
            ),
            GraphError::CrossClockAlignment {
                trigger_clock,
                dependency_clock,
            } => write!(
                f,
                "cannot align packets across clocks {} and {} without an explicit sync node",
                trigger_clock, dependency_clock
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

#[derive(Clone)]
struct PortDescriptor<T> {
    raw: T,
    owner: PortOwner,
    name: String,
    type_id: TypeId,
    type_name: &'static str,
    format_key: Option<String>,
    queue: Option<QueueConfig>,
    access: Option<AccessMode>,
}

struct PendingNode {
    id: NodeId,
    name: String,
    inputs: Vec<RawInPort>,
    outputs: Vec<RawOutPort>,
    schedule: Option<SchedulePlan>,
    factory: Option<Box<dyn NodeFactory>>,
}

struct NodeRecord {
    id: NodeId,
    schedule: SchedulePlan,
    factory: Option<Box<dyn NodeFactory>>,
}

struct RuntimeNode {
    runner: Box<dyn NodeRunner>,
    pending_triggers: VecDeque<TriggerEvent>,
    control_state: NodeControlState,
    dirty: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortOwner {
    Source,
    Sink,
    Node(NodeId),
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    from: RawOutPort,
    to: RawInPort,
}

struct PortQueue {
    items: VecDeque<Rc<ErasedPacket>>,
    config: QueueConfig,
    access: AccessMode,
}

#[derive(Clone)]
struct TriggerEvent {
    port: RawInPort,
    packet: Rc<ErasedPacket>,
}

struct ErasedPacket {
    meta: PacketMeta,
    payload: Rc<dyn Any>,
}

struct ExecutionContext {
    trigger: Option<RawInPort>,
    trigger_packet: Option<Rc<ErasedPacket>>,
    matched: HashMap<RawInPort, usize>,
}

struct NoopRunner;

impl NodeRunner for NoopRunner {
    fn process(&mut self, _ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
        Ok(())
    }
}

fn downcast_handle<T: PortData>(
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

fn apply_overflow_policy(queue: &mut PortQueue, packet: Rc<ErasedPacket>) -> GraphResult<()> {
    if queue.items.len() < queue.config.capacity {
        queue.items.push_back(packet);
        return Ok(());
    }

    match queue.config.overflow {
        OverflowPolicy::RejectPush => Err(GraphError::QueueFull {
            port: "port queue".to_string(),
        }),
        OverflowPolicy::DropNewest => Ok(()),
        OverflowPolicy::DropOldest => {
            let _ = queue.items.pop_front();
            queue.items.push_back(packet);
            Ok(())
        }
        OverflowPolicy::ReplaceLatest => {
            let _ = queue.items.pop_back();
            queue.items.push_back(packet);
            Ok(())
        }
    }
}

fn find_matching_timestamp(
    queue: &PortQueue,
    trigger_timestamp: &Timestamp,
    skew: u32,
    exact: bool,
) -> Option<usize> {
    for (index, packet) in queue.items.iter().enumerate() {
        let Some(timestamp) = packet.meta.timestamp.as_ref() else {
            continue;
        };
        if timestamp.clock != trigger_timestamp.clock {
            continue;
        }

        let delta = timestamp.start.abs_diff(trigger_timestamp.start);
        let matches = if exact {
            delta == 0
        } else {
            delta <= skew as u64
        };
        if matches {
            return Some(index);
        }
    }
    None
}

fn default_queue_index(queue: &PortQueue) -> Option<usize> {
    if queue.items.is_empty() {
        None
    } else if queue.access == AccessMode::PeekLatest {
        Some(queue.items.len() - 1)
    } else {
        Some(0)
    }
}

fn fallback_if_missing_timestamps(queue: &PortQueue) -> Option<usize> {
    queue
        .items
        .iter()
        .any(|packet| packet.meta.timestamp.is_none())
        .then(|| default_queue_index(queue))
        .flatten()
}

fn validate_cycles(
    nodes: &[PendingNode],
    edges: &[Edge],
    inputs: &[PortDescriptor<RawInPort>],
    outputs: &[PortDescriptor<RawOutPort>],
) -> GraphResult<()> {
    let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut indegree: HashMap<NodeId, usize> = HashMap::new();
    for node in nodes {
        adjacency.insert(node.id, Vec::new());
        indegree.insert(node.id, 0);
    }

    for edge in edges {
        let from_owner = outputs[edge.from.0].owner;
        let to_owner = inputs[edge.to.0].owner;
        if let (PortOwner::Node(from_node), PortOwner::Node(to_node)) = (from_owner, to_owner) {
            adjacency.entry(from_node).or_default().push(to_node);
            *indegree.entry(to_node).or_default() += 1;
        }
    }

    let mut queue: VecDeque<NodeId> = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(*node))
        .collect();
    let mut visited = 0usize;

    while let Some(node) = queue.pop_front() {
        visited += 1;
        if let Some(children) = adjacency.get(&node) {
            for child in children {
                let degree = indegree.get_mut(child).expect("child indegree must exist");
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(*child);
                }
            }
        }
    }

    if visited != nodes.len() {
        return Err(GraphError::GraphCycle);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Dummy(u32);

    struct PassthroughSpec;

    struct PassthroughHandles {
        input: InPort<Dummy>,
        output: OutPort<Dummy>,
    }

    struct PassthroughFactory {
        input: InPort<Dummy>,
        output: OutPort<Dummy>,
    }

    struct PassthroughRunner {
        input: InPort<Dummy>,
        output: OutPort<Dummy>,
    }

    impl NodeRunner for PassthroughRunner {
        fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> GraphResult<()> {
            if let Some(packet) = ctx.take(self.input)? {
                ctx.emit_handle(self.output, packet)?;
            }
            Ok(())
        }
    }

    impl NodeSpec for PassthroughSpec {
        type Handles = PassthroughHandles;

        fn register(self, graph: &mut GraphBuilder) -> GraphResult<Self::Handles> {
            let node = graph.new_node("passthrough");
            let input = graph.register_input::<Dummy>(node, "in", InputOptions::default());
            let output = graph.register_output::<Dummy>(node, "out", OutputOptions::default());
            graph.finish_node(
                node,
                SchedulePlan::OnArrival {
                    triggers: vec![input.raw()],
                },
                Box::new(PassthroughFactory { input, output }),
            )?;
            Ok(PassthroughHandles { input, output })
        }
    }

    impl NodeFactory for PassthroughFactory {
        fn describe(&self, _io: &mut NodeIoBuilder<'_>) -> GraphResult<()> {
            Ok(())
        }

        fn build(self: Box<Self>, _ctx: &mut BuildCtx) -> GraphResult<Box<dyn NodeRunner>> {
            Ok(Box::new(PassthroughRunner {
                input: self.input,
                output: self.output,
            }))
        }
    }

    #[test]
    fn runtime_routes_packets_between_source_and_sink() {
        let mut builder = GraphBuilder::new();
        let source = builder.source::<Dummy>("src", QueueConfig::bounded(4));
        let sink = builder.sink::<Dummy>("sink", QueueConfig::bounded(4));
        let node = builder.add_node(PassthroughSpec).expect("node should register");
        builder
            .connect(source, node.input)
            .expect("source should connect");
        builder
            .connect(node.output, sink)
            .expect("node should connect to sink");

        let spec = builder.build().expect("graph should build");
        let mut runtime = Runtime::new(spec, RuntimeOptions).expect("runtime should build");
        runtime
            .push(
                source,
                Packet {
                    meta: PacketMeta::default(),
                    payload: Dummy(7),
                },
            )
            .expect("push should succeed");
        runtime
            .run_until_stalled()
            .expect("runtime should drain the graph");

        let packet = runtime
            .try_pull(sink)
            .expect("pull should succeed")
            .expect("sink should receive one packet");
        assert_eq!(packet.payload(), &Dummy(7));
    }
}
