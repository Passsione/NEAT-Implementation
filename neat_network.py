"""
neat_network.py - Decode a Genome into a runnable RNN
======================================================
Supports recurrent (looping) connections.
Activation is computed iteratively over timesteps — each call to
activate() advances the network one step using the previous state.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from neat_genome import Genome, ACTIVATIONS, NODE_INPUT, NODE_OUTPUT, NODE_HIDDEN


class Network:
    """
    A compiled, runnable neural network decoded from a NEAT Genome.

    Supports:
    - Feed-forward connections
    - Recurrent / looping connections (value from previous timestep)
    - Per-node activation functions and biases
    - Multiple activation passes per call (for deeper feedforward paths)

    Usage
    -----
    net = Network.from_genome(genome)
    net.reset()                          # clear recurrent state
    output = net.activate([0.5, 1.0])   # one timestep
    """

    def __init__(
        self,
        input_ids:   List[int],
        output_ids:  List[int],
        node_order:  List[int],          # all non-input nodes in eval order
        activations: Dict[int, callable],
        biases:      Dict[int, float],
        connections: List[Tuple[int, int, float]],  # (in_node, out_node, weight)
    ):
        self.input_ids   = input_ids
        self.output_ids  = output_ids
        self.node_order  = node_order
        self.activations = activations
        self.biases      = biases
        self.connections = connections

        # state: node_id -> float (carries recurrent values between timesteps)
        self.state: Dict[int, float] = {}
        self.reset()

    def reset(self):
        """Clear all recurrent state (call between independent episodes)."""
        self.state = {nid: 0.0 for nid in self.node_order}
        self.state.update({nid: 0.0 for nid in self.input_ids})

    def activate(self, inputs: List[float], n_passes: int = 1) -> List[float]:
        """
        Feed inputs through the network for one timestep.

        Parameters
        ----------
        inputs   : list of floats, length == n_inputs
        n_passes : number of activation passes (use >1 for deep feedforward paths)

        Returns
        -------
        list of floats, length == n_outputs
        """
        if len(inputs) != len(self.input_ids):
            raise ValueError(f"Expected {len(self.input_ids)} inputs, got {len(inputs)}.")

        # load inputs into state
        for nid, val in zip(self.input_ids, inputs):
            self.state[nid] = float(val)

        for _ in range(n_passes):
            # accumulate weighted inputs for each node
            incoming: Dict[int, float] = {nid: self.biases.get(nid, 0.0)
                                          for nid in self.node_order}
            for in_id, out_id, weight in self.connections:
                if out_id in incoming:
                    incoming[out_id] += self.state.get(in_id, 0.0) * weight

            # apply activation functions and update state
            for nid in self.node_order:
                raw = incoming[nid]
                fn  = self.activations.get(nid, lambda x: x)
                self.state[nid] = float(fn(raw))

        return [self.state[nid] for nid in self.output_ids]

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_genome(cls, genome: Genome) -> Network:
        """Decode a Genome into a Network ready for activation."""
        act_fns = {
            nid: ACTIVATIONS.get(node.activation, ACTIVATIONS["sigmoid"])
            for nid, node in genome.nodes.items()
        }
        biases = {nid: node.bias for nid, node in genome.nodes.items()}

        # collect enabled connections
        conns = [
            (c.in_node, c.out_node, c.weight)
            for c in genome.connections.values()
            if c.enabled
        ]

        # eval order: outputs + hidden, inputs are pre-loaded separately
        # We don't topologically sort — recurrent nets evaluate all nodes each pass
        # non_input = [nid for nid, n in genome.nodes.items()
        #              if n.node_type != NODE_INPUT]

        # put outputs last for clarity (doesn't affect correctness)
        hidden  = [nid for nid, n in genome.nodes.items() if n.node_type == NODE_HIDDEN]
        outputs = [nid for nid, n in genome.nodes.items() if n.node_type == NODE_OUTPUT]
        node_order = hidden + outputs

        return cls(
            input_ids   = genome.input_ids,
            output_ids  = genome.output_ids,
            node_order  = node_order,
            activations = act_fns,
            biases      = biases,
            connections = conns,
        )
