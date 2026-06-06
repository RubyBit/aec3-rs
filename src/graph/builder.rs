//! Graph construction: node and port registration, wiring, and validation.

use std::any::{TypeId, type_name};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

use super::node::{NodeFactory, NodeSpec, SchedulePlan};
use super::port::{
    AccessMode, Edge, InPort, InputOptions, NodeId, OutPort, OutputOptions, PortData,
    PortDescriptor, PortOwner, QueueConfig, RawInPort, RawOutPort, Sink, Source,
};
use super::{GraphError, GraphResult};

/// Incrementally constructs a typed processing graph.
///
/// A builder registers external [`Source`] and [`Sink`] handles, adds nodes,
/// wires compatible ports with [`connect`](Self::connect), then freezes the
/// result with [`build`](Self::build).
///
/// ```
/// use aec3::graph::{GraphBuilder, Packet, PacketMeta, QueueConfig, Runtime};
///
/// let mut graph = GraphBuilder::new();
/// let source = graph.source::<i32>("control");
/// let sink = graph.sink::<i32>("observed", QueueConfig::latest(1));
///
/// graph.connect(source, sink)?;
///
/// let spec = graph.build()?;
/// let mut runtime = Runtime::new(spec)?;
/// runtime.push(
///     source,
///     Packet {
///         meta: PacketMeta::default(),
///         payload: 7,
///     },
/// )?;
///
/// let packet = runtime.try_pull(sink)?.expect("packet should be routed");
/// assert_eq!(*packet.payload(), 7);
/// # Ok::<(), aec3::graph::GraphError>(())
/// ```
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

/// Validated graph description consumed by [`Runtime`](crate::graph::Runtime).
///
/// `GraphSpec` is produced by [`GraphBuilder::build`]. It owns the registered
/// node factories and wiring plan, and is normally passed directly to
/// [`Runtime::new`](crate::graph::Runtime::new).
pub struct GraphSpec {
    pub(crate) nodes: Vec<NodeRecord>,
    pub(crate) input_ports: Vec<PortDescriptor<RawInPort>>,
    pub(crate) output_edges: Vec<Vec<RawInPort>>,
}

struct PendingNode {
    id: NodeId,
    name: String,
    schedule: Option<SchedulePlan>,
    factory: Option<Box<dyn NodeFactory>>,
}

pub(crate) struct NodeRecord {
    pub(crate) id: NodeId,
    pub(crate) schedule: SchedulePlan,
    pub(crate) factory: Option<Box<dyn NodeFactory>>,
}

impl GraphBuilder {
    /// Creates an empty graph builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an external input handle for packets pushed into the runtime.
    ///
    /// Sources do not own queues. Packets pushed through a source are routed to
    /// the connected input and sink queues.
    pub fn source<T: PortData>(&mut self, name: &str) -> Source<T> {
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

    /// Registers an external output handle that can be pulled from the runtime.
    ///
    /// The provided queue controls how many packets the sink can buffer and
    /// what happens when that buffer is full.
    pub fn sink<T: PortData>(&mut self, name: &str, queue: QueueConfig) -> Sink<T> {
        let raw = self.register_shadow_sink(name, queue, type_name::<T>(), TypeId::of::<T>());
        Sink {
            raw,
            marker: PhantomData,
        }
    }

    /// Adds a node described by a [`NodeSpec`] and returns its typed handles.
    ///
    /// Built-in nodes expose builder helpers that call this internally:
    ///
    /// ```no_run
    /// # use aec3::graph::{GraphBuilder, GraphError};
    /// # use aec3::nodes::{audio::AudioFormat, hpf};
    /// let mut graph = GraphBuilder::new();
    /// let format = AudioFormat::ten_ms(48_000, 1);
    /// let high_pass = hpf::builder(format).add_to(&mut graph)?;
    /// # Ok::<(), GraphError>(())
    /// ```
    pub fn add_node<N: NodeSpec>(&mut self, spec: N) -> GraphResult<N::Handles> {
        spec.register(self)
    }

    /// Connects an output port to an input port of the same payload type.
    ///
    /// If both ports declare format keys, those keys must match. This catches
    /// common audio wiring mistakes such as connecting 16 kHz audio to a node
    /// input registered for 48 kHz audio.
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

    /// Validates and freezes the graph so it can be run.
    ///
    /// Validation checks for unfinished nodes, missing schedules, zero-capacity
    /// queues, and cycles between runtime nodes.
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

        // Zero-capacity queues would otherwise behave inconsistently across
        // overflow policies (RejectPush rejects everything, DropOldest and
        // ReplaceLatest still store one packet).
        for port in &self.input_ports {
            if let Some(queue) = &port.queue
                && queue.capacity == 0
            {
                return Err(GraphError::InvalidQueueCapacity {
                    port: port.name.clone(),
                });
            }
        }

        validate_cycles(
            &self.nodes,
            &self.edges,
            &self.input_ports,
            &self.output_ports,
        )?;

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

    /// Starts manual registration of a custom node.
    ///
    /// Most users should prefer [`add_node`](Self::add_node). This lower-level
    /// method is mainly for [`NodeSpec`] implementations that need to register
    /// several ports before calling [`finish_node`](Self::finish_node).
    pub fn new_node(&mut self, name: &str) -> NodeId {
        let node_id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(PendingNode {
            id: node_id,
            name: name.to_string(),
            schedule: None,
            factory: None,
        });
        node_id
    }

    /// Registers an input port owned by a custom node.
    ///
    /// The input queue and access mode come from [`InputOptions`]. If a
    /// `format_key` is supplied, [`connect`](Self::connect) uses it for runtime
    /// format validation.
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
        InPort {
            raw,
            marker: PhantomData,
        }
    }

    /// Registers an output port owned by a custom node.
    ///
    /// If a `format_key` is supplied, [`connect`](Self::connect) checks it
    /// against connected inputs that also declare a format key.
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
        OutPort {
            raw,
            marker: PhantomData,
        }
    }

    /// Completes manual node registration with its schedule and runtime factory.
    ///
    /// A node created by [`new_node`](Self::new_node) must be finished exactly
    /// once before [`build`](Self::build), otherwise graph validation returns an
    /// error.
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
