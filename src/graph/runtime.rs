//! Graph execution: input queues, trigger scheduling, and node contexts.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use super::builder::GraphSpec;
use super::node::{
    BuildCtx, MatchPolicy, NodeControlState, NodeRunner, NoopRunner, ReadyState, SchedulePlan,
};
use super::packet::{ErasedPacket, Packet, PacketHandle, PacketMeta, downcast_handle};
use super::port::{
    AccessMode, InPort, NodeId, OutPort, OverflowPolicy, PortData, PortOwner, QueueConfig,
    RawInPort, RawOutPort, Sink, Source,
};
use super::{GraphError, GraphResult};

pub struct Runtime {
    spec: GraphSpec,
    nodes: Vec<RuntimeNode>,
    input_queues: Vec<Option<PortQueue>>,
    dirty_nodes: VecDeque<NodeId>,
}

struct RuntimeNode {
    runner: Box<dyn NodeRunner>,
    pending_triggers: VecDeque<TriggerEvent>,
    control_state: NodeControlState,
    dirty: bool,
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

struct ExecutionContext {
    trigger: Option<RawInPort>,
    trigger_packet: Option<Rc<ErasedPacket>>,
    matched: HashMap<RawInPort, usize>,
    /// Inputs declared as `AlignOn` dependencies for this execution. Reads on
    /// these ports must go through the matched index (or return nothing) so an
    /// unmatched dependency can never fall back to an arbitrary queued packet.
    dep_ports: Vec<RawInPort>,
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
        self.runtime
            .node_state(self.node_id)
            .unwrap_or(NodeControlState::Active)
    }
}

pub struct ProcessCtx<'a> {
    runtime: &'a mut Runtime,
    node_id: NodeId,
    control_state: NodeControlState,
    current_trigger: Option<RawInPort>,
    trigger_packet: Option<Rc<ErasedPacket>>,
    matched: HashMap<RawInPort, usize>,
    dep_ports: Vec<RawInPort>,
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
        let Some(index) = self.dep_index(port.raw) else {
            return Ok(None);
        };
        self.runtime.take_from_input(port.raw, index)
    }

    pub fn peek<T: PortData>(&self, port: InPort<T>) -> GraphResult<Option<PacketHandle<T>>> {
        let Some(index) = self.dep_index(port.raw) else {
            return Ok(None);
        };
        self.runtime.peek_input(port.raw, index)
    }

    /// Resolves which queue index a read on `port` may use.
    ///
    /// Returns `None` (read nothing) for an aligned dependency that has no
    /// match this execution: falling back to the queue head would hand the
    /// node a packet from the wrong frame.
    fn dep_index(&self, port: RawInPort) -> Option<Option<usize>> {
        match self.matched.get(&port).copied() {
            Some(index) => Some(Some(index)),
            None if self.dep_ports.contains(&port) => None,
            None => Some(None),
        }
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

impl Runtime {
    pub fn new(mut spec: GraphSpec) -> GraphResult<Self> {
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

        let mut input_queues = (0..spec.input_ports.len())
            .map(|_| None)
            .collect::<Vec<_>>();
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

    /// Pushes a packet into the graph through a source port, returning an error if the port is
    /// unknown or a connected input queue is full. The graph automatically routes the packet to all
    /// connected input ports, where it may trigger node executions or be rejected due to overflow
    /// based on [`OverflowPolicy`].
    ///
    /// [`OverflowPolicy`]: crate::graph::OverflowPolicy
    pub fn push<T: PortData>(&mut self, src: Source<T>, packet: Packet<T>) -> GraphResult<()> {
        self.emit_packet(src.raw, packet)
    }

    /// Attempts to pull a packet from a sink port.
    ///
    /// Returns `Ok(None)` when the sink queue is empty, or an error if the port is
    /// unknown.
    pub fn try_pull<T: PortData>(&mut self, sink: Sink<T>) -> GraphResult<Option<PacketHandle<T>>> {
        self.take_from_input(sink.raw, None)
    }

    pub fn set_node_state(&mut self, node_id: NodeId, state: NodeControlState) -> GraphResult<()> {
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

    /// Runs the graph until no further node can make progress without new input.
    /// Returns the number of nodes executed.
    pub fn run_until_stalled(&mut self) -> GraphResult<usize> {
        let mut processed = 0usize;
        while let Some(node_id) = self.dirty_nodes.pop_front() {
            let node_index = node_id.0;
            if !self.nodes[node_index].dirty {
                continue;
            }
            self.nodes[node_index].dirty = false;

            self.prune_stale_deps(node_id);
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
                dep_ports: execution.dep_ports,
            };
            runner.process(&mut ctx)?;
            self.nodes[node_index].runner = runner;
            processed += 1;

            self.prune_stale_deps(node_id);
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
                        dep_ports: Vec::new(),
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
                        trigger_event.port,
                        &trigger_event.packet.meta,
                        dep.match_policy,
                        dep.required,
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
                    dep_ports: deps.iter().map(|dep| dep.input).collect(),
                }))
            }
            SchedulePlan::Custom => {
                let ready = self.nodes[node_id.0].runner.ready(&ReadyCtx {
                    runtime: self,
                    node_id,
                })?;
                if ready == ReadyState::Ready {
                    Ok(Some(ExecutionContext {
                        trigger: None,
                        trigger_packet: None,
                        matched: HashMap::new(),
                        dep_ports: Vec::new(),
                    }))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Drops `BySequence` dependency packets that are older than the pending
    /// trigger. Sequences are monotonic, so such packets can never match the
    /// current or any future trigger; without pruning, persistent mismatches
    /// would accumulate until the dependency queue rejects pushes.
    fn prune_stale_deps(&mut self, node_id: NodeId) {
        let SchedulePlan::AlignOn { trigger, deps } = &self.spec.nodes[node_id.0].schedule else {
            return;
        };
        let Some(trigger_event) = self.nodes[node_id.0].pending_triggers.front() else {
            return;
        };
        if trigger_event.port != *trigger {
            return;
        }
        let Some(trigger_sequence) = trigger_event.packet.meta.sequence else {
            return;
        };

        for dep in deps {
            if dep.match_policy != MatchPolicy::BySequence {
                continue;
            }
            if let Some(queue) = self
                .input_queues
                .get_mut(dep.input.0)
                .and_then(|queue| queue.as_mut())
            {
                queue.items.retain(
                    |packet| !matches!(packet.meta.sequence, Some(s) if s < trigger_sequence),
                );
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
        let descriptor = &self.spec.input_ports[port.0];
        let owner = descriptor.owner;
        let port_name = descriptor.name.as_str();
        let queue = self
            .input_queues
            .get_mut(port.0)
            .and_then(|queue| queue.as_mut())
            .ok_or(GraphError::UnknownInputPort(port.0))?;
        apply_overflow_policy(queue, packet.clone(), port_name)?;

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
        trigger_port: RawInPort,
        trigger_meta: &PacketMeta,
        policy: MatchPolicy,
        required: bool,
    ) -> GraphResult<Option<usize>> {
        let queue = self
            .input_queues
            .get(port.0)
            .and_then(|queue| queue.as_ref())
            .ok_or(GraphError::UnknownInputPort(port.0))?;

        match policy {
            MatchPolicy::Fifo => Ok(default_queue_index(queue)),
            MatchPolicy::BySequence => {
                let Some(trigger_sequence) = trigger_meta.sequence else {
                    if required {
                        // An unstamped trigger can never become matchable;
                        // failing loudly beats stalling forever. Attribute the
                        // error to the trigger port — that is where the stamp
                        // is missing.
                        return Err(GraphError::MissingSequenceForAlignment {
                            port: self.spec.input_ports[trigger_port.0].name.clone(),
                        });
                    }
                    return Ok(None);
                };
                let mut saw_missing_sequence = false;
                for (index, packet) in queue.items.iter().enumerate() {
                    match packet.meta.sequence {
                        Some(sequence) if sequence == trigger_sequence => {
                            return Ok(Some(index));
                        }
                        Some(_) => {}
                        None => saw_missing_sequence = true,
                    }
                }
                if saw_missing_sequence && required {
                    return Err(GraphError::MissingSequenceForAlignment {
                        port: self.spec.input_ports[port.0].name.clone(),
                    });
                }
                // The matching packet may simply not have arrived yet: wait.
                Ok(None)
            }
        }
    }
}

fn apply_overflow_policy(
    queue: &mut PortQueue,
    packet: Rc<ErasedPacket>,
    port_name: &str,
) -> GraphResult<()> {
    if queue.items.len() < queue.config.capacity {
        queue.items.push_back(packet);
        return Ok(());
    }

    match queue.config.overflow {
        OverflowPolicy::RejectPush => Err(GraphError::QueueFull {
            port: port_name.to_string(),
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

fn default_queue_index(queue: &PortQueue) -> Option<usize> {
    if queue.items.is_empty() {
        None
    } else if queue.access == AccessMode::PeekLatest {
        Some(queue.items.len() - 1)
    } else {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{
        GraphBuilder, GraphResult, InPort, InputOptions, NodeFactory, NodeRunner, NodeSpec,
        OutPort, OutputOptions, Packet, PacketMeta, ProcessCtx, QueueConfig, Runtime, SchedulePlan,
    };

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
        fn build(
            self: Box<Self>,
            _ctx: &mut crate::graph::BuildCtx,
        ) -> GraphResult<Box<dyn NodeRunner>> {
            Ok(Box::new(PassthroughRunner {
                input: self.input,
                output: self.output,
            }))
        }
    }

    #[test]
    fn runtime_routes_packets_between_source_and_sink() {
        let mut builder = GraphBuilder::new();
        let source = builder.source::<Dummy>("src");
        let sink = builder.sink::<Dummy>("sink", QueueConfig::bounded(4));
        let node = builder
            .add_node(PassthroughSpec)
            .expect("node should register");
        builder
            .connect(source, node.input)
            .expect("source should connect");
        builder
            .connect(node.output, sink)
            .expect("node should connect to sink");

        let spec = builder.build().expect("graph should build");
        let mut runtime = Runtime::new(spec).expect("runtime should build");
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
