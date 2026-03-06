"""
run.py - Configure and run your NEAT simulation
=================================================
This is the only file you need to edit.

Two examples are included:
  1. XOR  - the classic NEAT benchmark (run as-is to verify everything works)
  2. YOUR FITNESS FUNCTION - template for plugging in your own task

To use your own task, replace `my_fitness_fn` with whatever evaluation
logic you need. The function receives a Genome and must return a float
(higher = better).
"""

import numpy as np
from neat import NEATConfig, NEATSimulator
from neat_network import Network
from neat_genome import Genome


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 1: XOR (classic NEAT benchmark — delete when done)
# ══════════════════════════════════════════════════════════════════════════════

XOR_INPUTS  = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_TARGETS = [0,       1,      1,      0     ]

def xor_fitness(genome: Genome) -> float:
    net = Network.from_genome(genome)
    error = 0.0
    for inputs, target in zip(XOR_INPUTS, XOR_TARGETS):
        net.reset()
        output = net.activate(inputs, n_passes=2)[0]
        error += (output - target) ** 2
    return 4.0 - error   # max fitness = 4.0


# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE 2: YOUR CUSTOM FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
#
# Replace the body of this function with your own evaluation logic.
# The network is a recurrent RNN — call net.reset() between episodes,
# then net.activate([...]) for each timestep.
#
# Return a float. Higher = better.
#
# def my_fitness_fn(genome: Genome) -> float:
#     net = Network.from_genome(genome)
#
#     total_reward = 0.0
#
#     for episode in range(5):
#         net.reset()           # clear recurrent state between episodes
#         observation = [0.0, 0.0]   # replace with your initial state
#
#         for step in range(100):
#             action = net.activate(observation)
#             # ... apply action, get next observation and reward ...
#             # total_reward += reward
#             pass
#
#     return total_reward


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

cfg = NEATConfig(
    # ── network ───────────────────────────────────────────────────────────────
    n_inputs  = 2,              # must match your fitness function's input size
    n_outputs = 1,              # must match your fitness function's output size
    output_activation   = "sigmoid",   # sigmoid | tanh | relu | identity | ...
    initial_connection  = "partial",      # "full" | "partial" | "none"
    allow_recurrent     = True,        # allow looping / recurrent connections

    # ── population ────────────────────────────────────────────────────────────
    population_size     = 150,
    elitism_per_species = 1,           # champions skip mutation

    # ── mutation ──────────────────────────────────────────────────────────────
    weight_mutate_rate    = 0.8,       # prob each weight is mutated
    weight_replace_rate   = 0.1,       # prob of full replacement vs perturbation
    weight_perturb_std    = 0.3,
    bias_mutate_rate      = 0.3,
    add_connection_rate   = 0.05,      # structural: new connection
    add_node_rate         = 0.03,      # structural: split connection with new node
    toggle_rate           = 0.01,      # enable/disable a connection
    activation_mutate_rate = 0.01,     # change a hidden node's activation fn

    # ── crossover ─────────────────────────────────────────────────────────────
    crossover_rate    = 0.75,
    disable_gene_prob = 0.75,          # prob to disable if disabled in either parent

    # ── speciation ────────────────────────────────────────────────────────────
    compatibility_threshold = 3.0,     # lower = more species, higher = fewer
    c1 = 1.0,                          # excess gene coefficient
    c2 = 1.0,                          # disjoint gene coefficient
    c3 = 0.4,                          # weight difference coefficient
    stagnation_limit  = 15,            # species removed after N gens without improvement
    survival_ratio    = 0.5,           # fraction of each species kept each gen

    # ── stopping ──────────────────────────────────────────────────────────────
    max_generations   = 300,
    fitness_threshold = 3.9,           # stop when best fitness >= this (XOR: ~3.9)

    # ── logging ───────────────────────────────────────────────────────────────
    verbose     = True,
    print_every = 10,

    seed = 42,
)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # swap xor_fitness for my_fitness_fn once you have your own task
    sim    = NEATSimulator(cfg, fitness_fn=xor_fitness)
    result = sim.run()

    print("\n" + result.summary())

    # ── verify the best network on XOR ───────────────────────────────────────
    print("\nBest network XOR outputs:")
    net = result.best_network()
    for inputs, target in zip(XOR_INPUTS, XOR_TARGETS):
        net.reset()
        out = net.activate(inputs)[0]
        print(f"  {inputs} -> {out:.4f}  (target {target})  "
              f"{'OK' if abs(out - target) < 0.5 else 'FAIL'}")

    result.plot(save_path="neat_results.png", show=False)
