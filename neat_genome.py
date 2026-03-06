"""
neat_genome.py - NEAT Genome: Nodes, Connections, Innovation Numbers
=====================================================================
A genome encodes a graph of nodes and directed connections.
Connections may be recurrent (loop back), forming an RNN.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  INNOVATION TRACKER  (global — shared across the entire population)
# ══════════════════════════════════════════════════════════════════════════════

class InnovationTracker:
    """
    Assigns a unique innovation number to every new structural gene.
    Within one generation, the same structural change always gets the
    same innovation number so crossover can align genomes correctly.
    """
    def __init__(self):
        self._counter: int = 0
        # (in_node, out_node) -> innovation number  (reset each generation)
        self._gen_cache: Dict[Tuple[int, int], int] = {}
        # node split: connection_innov -> new_node_id
        self._node_cache: Dict[int, int] = {}
        self._node_counter: int = 0

    def next_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def get_connection_innov(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self._gen_cache:
            self._counter += 1
            self._gen_cache[key] = self._counter
        return self._gen_cache[key]

    def get_split_node(self, connection_innov: int) -> int:
        if connection_innov not in self._node_cache:
            self._node_counter += 1
            self._node_cache[connection_innov] = self._node_counter
        return self._node_cache[connection_innov]

    def new_generation(self):
        """Call at the start of each generation to reset the per-gen cache."""
        self._gen_cache.clear()

    def init_node_counter(self, n: int):
        """Set node counter after building the initial population."""
        self._node_counter = max(self._node_counter, n)


# ══════════════════════════════════════════════════════════════════════════════
#  GENE TYPES
# ══════════════════════════════════════════════════════════════════════════════

NODE_INPUT   = "input"
NODE_OUTPUT  = "output"
NODE_HIDDEN  = "hidden"

ACTIVATIONS = {
    "sigmoid":  lambda x: 1 / (1 + np.exp(-np.clip(x, -60, 60))),
    "tanh":     np.tanh,
    "relu":     lambda x: np.maximum(0, x),
    "leaky":    lambda x: np.where(x >= 0, x, 0.01 * x),
    "identity": lambda x: x,
    "gauss":    lambda x: np.exp(-x**2),
    "sin":      np.sin,
}


@dataclass
class NodeGene:
    node_id:    int
    node_type:  str          # NODE_INPUT | NODE_OUTPUT | NODE_HIDDEN
    activation: str = "sigmoid"
    bias:       float = 0.0


@dataclass
class ConnectionGene:
    in_node:   int
    out_node:  int
    weight:    float
    enabled:   bool
    innov:     int           # innovation number


# ══════════════════════════════════════════════════════════════════════════════
#  GENOME
# ══════════════════════════════════════════════════════════════════════════════

class Genome:
    """
    A NEAT genome: a collection of node genes and connection genes
    that together describe a (potentially recurrent) neural network graph.
    """

    def __init__(self, genome_id: int, n_inputs: int, n_outputs: int):
        self.genome_id = genome_id
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.fitness:  float = 0.0
        self.species_id: Optional[int] = None

        # keyed by node_id and innov respectively for O(1) lookup
        self.nodes:       Dict[int, NodeGene]       = {}
        self.connections: Dict[int, ConnectionGene] = {}

    # ── construction helpers ──────────────────────────────────────────────────

    def add_node(self, node: NodeGene):
        self.nodes[node.node_id] = node

    def add_connection(self, conn: ConnectionGene):
        self.connections[conn.innov] = conn

    @property
    def input_ids(self)  -> List[int]: return [n.node_id for n in self.nodes.values() if n.node_type == NODE_INPUT]
    @property
    def output_ids(self) -> List[int]: return [n.node_id for n in self.nodes.values() if n.node_type == NODE_OUTPUT]
    @property
    def hidden_ids(self) -> List[int]: return [n.node_id for n in self.nodes.values() if n.node_type == NODE_HIDDEN]

    def copy(self, new_id: int) -> Genome:
        g = Genome(new_id, self.n_inputs, self.n_outputs)
        g.nodes       = {k: copy.deepcopy(v) for k, v in self.nodes.items()}
        g.connections = {k: copy.deepcopy(v) for k, v in self.connections.items()}
        return g

    def __repr__(self):
        en = sum(1 for c in self.connections.values() if c.enabled)
        return (f"Genome(id={self.genome_id}, "
                f"nodes={len(self.nodes)}, "
                f"conns={len(self.connections)} [{en} enabled], "
                f"fitness={self.fitness:.4f})")


# ══════════════════════════════════════════════════════════════════════════════
#  GENOME FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_genome(
    genome_id:   int,
    n_inputs:    int,
    n_outputs:   int,
    tracker:     InnovationTracker,
    rng:         np.random.Generator,
    output_activation: str = "sigmoid",
    initial_connection: str = "full",   # "full" | "partial" | "none"
    initial_weight_std: float = 1.0,
) -> Genome:
    """
    Build a minimal genome: input nodes + output nodes + optional connections.
    No hidden nodes are created initially (NEAT grows them via mutation).

    Node IDs: inputs 1..n_inputs, outputs n_inputs+1..n_inputs+n_outputs
    """
    g = Genome(genome_id, n_inputs, n_outputs)

    # input nodes (no activation / bias — pass-through)
    for i in range(1, n_inputs + 1):
        g.add_node(NodeGene(node_id=i, node_type=NODE_INPUT, activation="identity", bias=0.0))

    # output nodes
    for i in range(n_inputs + 1, n_inputs + n_outputs + 1):
        g.add_node(NodeGene(node_id=i, node_type=NODE_OUTPUT,
                            activation=output_activation,
                            bias=float(rng.normal(0, 0.1))))

    tracker.init_node_counter(n_inputs + n_outputs)

    # initial connections
    if initial_connection == "full":
        pairs = [(i, j)
                 for i in range(1, n_inputs + 1)
                 for j in range(n_inputs + 1, n_inputs + n_outputs + 1)]
    elif initial_connection == "partial":
        all_pairs = [(i, j)
                     for i in range(1, n_inputs + 1)
                     for j in range(n_inputs + 1, n_inputs + n_outputs + 1)]
        pairs = [p for p in all_pairs if rng.random() < 0.5]
    else:
        pairs = []

    for in_id, out_id in pairs:
        innov = tracker.get_connection_innov(in_id, out_id)
        g.add_connection(ConnectionGene(
            in_node=in_id, out_node=out_id,
            weight=float(rng.normal(0, initial_weight_std)),
            enabled=True, innov=innov,
        ))

    return g
