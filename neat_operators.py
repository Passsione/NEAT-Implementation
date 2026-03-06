"""
neat_operators.py - NEAT Mutation and Crossover
================================================
Mutation:
  - Perturb / replace weights
  - Add connection (including recurrent)
  - Add node (split existing connection)
  - Toggle connection enabled/disabled
  - Mutate bias
  - Mutate activation function

Crossover:
  - Align genes by innovation number
  - Inherit matching genes randomly
  - Inherit disjoint/excess genes from fitter parent
"""

from __future__ import annotations
import copy
from typing import Optional
import numpy as np

from neat_genome import (
    Genome, NodeGene, ConnectionGene,
    InnovationTracker, ACTIVATIONS,
    NODE_INPUT, NODE_OUTPUT, NODE_HIDDEN,
)


# ══════════════════════════════════════════════════════════════════════════════
#  MUTATION
# ══════════════════════════════════════════════════════════════════════════════

def mutate(
    genome:  Genome,
    tracker: InnovationTracker,
    rng:     np.random.Generator,
    # --- rates (all independently rolled) ---
    weight_mutate_rate:    float = 0.8,   # prob of mutating each weight
    weight_replace_rate:   float = 0.1,   # prob of replacing vs perturbing
    weight_perturb_std:    float = 0.3,
    weight_init_std:       float = 1.0,
    bias_mutate_rate:      float = 0.3,
    bias_perturb_std:      float = 0.1,
    add_connection_rate:   float = 0.05,
    add_node_rate:         float = 0.03,
    toggle_rate:           float = 0.01,
    activation_mutate_rate: float = 0.01,
    allow_recurrent:       bool  = True,
    max_add_tries:         int   = 20,    # attempts to find a valid new connection
) -> Genome:
    """
    Apply all mutation operators to a copy of the genome.
    Returns the mutated child (original is unchanged).
    """
    g = genome.copy(genome.genome_id)

    _mutate_weights(g, rng, weight_mutate_rate, weight_replace_rate,
                    weight_perturb_std, weight_init_std)
    _mutate_biases(g, rng, bias_mutate_rate, bias_perturb_std)

    if rng.random() < activation_mutate_rate:
        _mutate_activation(g, rng)

    if rng.random() < toggle_rate:
        _toggle_connection(g, rng)

    if rng.random() < add_node_rate:
        _add_node(g, tracker, rng, weight_init_std)

    if rng.random() < add_connection_rate:
        _add_connection(g, tracker, rng, weight_init_std, allow_recurrent, max_add_tries)

    return g


# ── weight mutation ───────────────────────────────────────────────────────────

def _mutate_weights(g, rng, rate, replace_rate, perturb_std, init_std):
    for conn in g.connections.values():
        if rng.random() < rate:
            if rng.random() < replace_rate:
                conn.weight = float(rng.normal(0, init_std))
            else:
                conn.weight += float(rng.normal(0, perturb_std))
            conn.weight = float(np.clip(conn.weight, -8.0, 8.0))


def _mutate_biases(g, rng, rate, perturb_std):
    for node in g.nodes.values():
        if node.node_type == NODE_INPUT:
            continue
        if rng.random() < rate:
            node.bias += float(rng.normal(0, perturb_std))
            node.bias  = float(np.clip(node.bias, -4.0, 4.0))


def _mutate_activation(g, rng):
    hidden = [n for n in g.nodes.values() if n.node_type == NODE_HIDDEN]
    if hidden:
        node = rng.choice(hidden)
        node.activation = rng.choice(list(ACTIVATIONS.keys()))


def _toggle_connection(g, rng):
    if g.connections:
        conn = rng.choice(list(g.connections.values()))
        conn.enabled = not conn.enabled


# ── structural mutations ──────────────────────────────────────────────────────

def _add_node(g: Genome, tracker: InnovationTracker, rng, weight_init_std):
    """
    Split a random enabled connection A->B into A->NEW->B.
    Original connection is disabled; two new connections are created.
    """
    enabled = [c for c in g.connections.values() if c.enabled]
    if not enabled:
        return

    old_conn = rng.choice(enabled)
    old_conn.enabled = False

    new_node_id = tracker.get_split_node(old_conn.innov)

    # avoid adding the same node twice
    if new_node_id in g.nodes:
        return

    new_node = NodeGene(
        node_id=new_node_id,
        node_type=NODE_HIDDEN,
        activation=str(rng.choice(list(ACTIVATIONS.keys()))),
        bias=0.0,
    )
    g.add_node(new_node)

    # A -> NEW  (weight 1.0 preserves original signal)
    innov1 = tracker.get_connection_innov(old_conn.in_node, new_node_id)
    g.add_connection(ConnectionGene(
        in_node=old_conn.in_node, out_node=new_node_id,
        weight=1.0, enabled=True, innov=innov1,
    ))

    # NEW -> B  (weight = original weight preserves original signal)
    innov2 = tracker.get_connection_innov(new_node_id, old_conn.out_node)
    g.add_connection(ConnectionGene(
        in_node=new_node_id, out_node=old_conn.out_node,
        weight=old_conn.weight, enabled=True, innov=innov2,
    ))


def _add_connection(
    g: Genome, tracker: InnovationTracker,
    rng, weight_init_std, allow_recurrent, max_tries,
):
    """Add a new connection between two existing nodes (may be recurrent)."""
    existing = {(c.in_node, c.out_node) for c in g.connections.values()}
    all_ids  = list(g.nodes.keys())
    non_input = [nid for nid, n in g.nodes.items() if n.node_type != NODE_INPUT]

    for _ in range(max_tries):
        in_id  = int(rng.choice(all_ids))
        out_id = int(rng.choice(non_input))

        if (in_id, out_id) in existing:
            continue

        # disallow input -> input
        if g.nodes[in_id].node_type == NODE_INPUT and g.nodes[out_id].node_type == NODE_INPUT:
            continue

        # if recurrent not allowed, skip self-loops and back-connections
        if not allow_recurrent and in_id == out_id:
            continue

        innov = tracker.get_connection_innov(in_id, out_id)
        g.add_connection(ConnectionGene(
            in_node=in_id, out_node=out_id,
            weight=float(rng.normal(0, weight_init_std)),
            enabled=True, innov=innov,
        ))
        return


# ══════════════════════════════════════════════════════════════════════════════
#  CROSSOVER
# ══════════════════════════════════════════════════════════════════════════════

def crossover(
    parent1:   Genome,
    parent2:   Genome,
    child_id:  int,
    rng:       np.random.Generator,
    disable_prob: float = 0.75,  # prob to disable gene if disabled in either parent
) -> Genome:
    """
    Produce a child genome by crossing over two parents.

    - Matching genes (same innovation number): inherited randomly from either parent
    - Disjoint / excess genes: inherited from the fitter parent only
    - If equal fitness, inherit all disjoint/excess genes randomly

    Node genes are inherited to match the received connection genes.
    """
    # determine which parent is fitter
    if parent1.fitness > parent2.fitness:
        fit, unfit = parent1, parent2
    elif parent2.fitness > parent1.fitness:
        fit, unfit = parent2, parent1
    else:
        # equal fitness: treat both as "fit" — randomly pick for disjoint too
        fit, unfit = parent1, parent2

    child = Genome(child_id, parent1.n_inputs, parent1.n_outputs)

    fit_innovs   = set(fit.connections.keys())
    unfit_innovs = set(unfit.connections.keys())
    all_innovs   = fit_innovs | unfit_innovs

    equal_fitness = abs(parent1.fitness - parent2.fitness) < 1e-9

    for innov in all_innovs:
        in_fit   = innov in fit_innovs
        in_unfit = innov in unfit_innovs

        if in_fit and in_unfit:
            # matching gene: pick randomly
            src_conn = fit.connections[innov] if rng.random() < 0.5 \
                       else unfit.connections[innov]
            conn = copy.deepcopy(src_conn)
            # disable if either parent has it disabled
            if not fit.connections[innov].enabled or not unfit.connections[innov].enabled:
                conn.enabled = rng.random() > disable_prob
        elif in_fit:
            # disjoint/excess from fit parent: always inherit
            conn = copy.deepcopy(fit.connections[innov])
        else:
            # disjoint/excess from unfit parent: inherit only if equal fitness
            if not equal_fitness:
                continue
            if rng.random() < 0.5:
                continue
            conn = copy.deepcopy(unfit.connections[innov])

        child.add_connection(conn)

        # ensure both endpoint nodes exist in the child
        for nid in (conn.in_node, conn.out_node):
            if nid not in child.nodes:
                # prefer node gene from fitter parent
                src_nodes = fit.nodes if nid in fit.nodes else unfit.nodes
                if nid in src_nodes:
                    child.add_node(copy.deepcopy(src_nodes[nid]))

    # ensure all input/output nodes are present even if no connections reference them
    for nid, node in fit.nodes.items():
        if nid not in child.nodes:
            child.add_node(copy.deepcopy(node))

    return child
