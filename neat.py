"""
neat.py - Full NEAT Engine
===========================
Wires genome, network, operators, and speciation into a
complete generational loop.

Quickstart
----------
    from neat import NEATConfig, NEATSimulator
    from neat_network import Network

    def my_fitness(genome):
        net = Network.from_genome(genome)
        net.reset()
        output = net.activate([0.0, 1.0])
        return 1.0 - abs(output[0] - 1.0)   # example: XOR-style

    cfg = NEATConfig(n_inputs=2, n_outputs=1)
    sim = NEATSimulator(cfg, fitness_fn=my_fitness)
    result = sim.run()
    print(result.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np

from neat_genome import Genome, InnovationTracker, make_genome
from neat_network import Network
from neat_operators import mutate, crossover
from neat_speciation import SpeciationManager


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NEATConfig:
    # ── network topology ──────────────────────────────────────────────────────
    n_inputs:  int = 2
    n_outputs: int = 1
    output_activation:   str   = "sigmoid"   # activation for output nodes
    initial_connection:  str   = "full"      # "full" | "partial" | "none"
    initial_weight_std:  float = 1.0
    allow_recurrent:     bool  = True        # allow looping connections

    # ── population ────────────────────────────────────────────────────────────
    population_size: int = 150
    elitism_per_species: int = 1            # top N of each species skip mutation

    # ── mutation rates ────────────────────────────────────────────────────────
    weight_mutate_rate:    float = 0.8
    weight_replace_rate:   float = 0.1
    weight_perturb_std:    float = 0.3
    bias_mutate_rate:      float = 0.3
    add_connection_rate:   float = 0.05
    add_node_rate:         float = 0.03
    toggle_rate:           float = 0.01
    activation_mutate_rate: float = 0.01

    # ── crossover ─────────────────────────────────────────────────────────────
    crossover_rate:        float = 0.75     # prob of crossover vs cloning
    disable_gene_prob:     float = 0.75     # prob to disable if disabled in either parent

    # ── speciation ────────────────────────────────────────────────────────────
    compatibility_threshold: float = 3.0
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    stagnation_limit:    int   = 15         # species killed after N gens stagnant
    survival_ratio:      float = 0.5        # fraction of each species that survives culling

    # ── stopping ──────────────────────────────────────────────────────────────
    max_generations:   int   = 500
    fitness_threshold: Optional[float] = None   # stop when best fitness >= this

    # ── logging ───────────────────────────────────────────────────────────────
    verbose:     bool = True
    print_every: int  = 10

    # ── reproducibility ───────────────────────────────────────────────────────
    seed: Optional[int] = 42


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationRecord:
    generation:    int
    best_fitness:  float
    mean_fitness:  float
    n_species:     int
    n_nodes_best:  int
    n_conns_best:  int


@dataclass
class NEATResult:
    best_genome:      Genome
    best_fitness:     float
    generations_run:  int
    stop_reason:      str
    history:          List[GenerationRecord]
    elapsed_seconds:  float

    def summary(self) -> str:
        g = self.best_genome
        en = sum(1 for c in g.connections.values() if c.enabled)
        return "\n".join([
            "=" * 54,
            "  NEAT Simulation - Results",
            "=" * 54,
            f"  Generations   : {self.generations_run}",
            f"  Stop reason   : {self.stop_reason}",
            f"  Best fitness  : {self.best_fitness:.6f}",
            f"  Elapsed       : {self.elapsed_seconds:.2f}s",
            f"  Best network  : {len(g.nodes)} nodes, "
            f"{len(g.connections)} connections ({en} enabled)",
            f"  Species count : {self.history[-1].n_species if self.history else '?'}",
            "=" * 54,
        ])

    def best_network(self) -> Network:
        """Decode the best genome into a runnable Network."""
        return Network.from_genome(self.best_genome)

    def plot(self, save_path: str = None, show: bool = True):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        h = self.history
        gens  = [r.generation   for r in h]
        best  = [r.best_fitness  for r in h]
        mean  = [r.mean_fitness  for r in h]
        nspc  = [r.n_species     for r in h]
        nnodes = [r.n_nodes_best for r in h]
        nconns = [r.n_conns_best for r in h]

        fig = plt.figure(figsize=(14, 9))
        fig.suptitle("NEAT Evolution", fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(gens, best, color="#2196F3", lw=2, label="Best")
        ax1.plot(gens, mean, color="#FF9800", lw=1.5, ls="--", label="Mean")
        ax1.set(title="Fitness", xlabel="Generation", ylabel="Fitness")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(gens, nspc, color="#9C27B0", lw=2)
        ax2.set(title="Number of Species", xlabel="Generation", ylabel="Species")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(gens, nnodes, color="#4CAF50", lw=2, label="Nodes")
        ax3.set(title="Best Genome Complexity", xlabel="Generation", ylabel="Count")
        ax3_r = ax3.twinx()
        ax3_r.plot(gens, nconns, color="#F44336", lw=1.5, ls="--", label="Connections")
        ax3_r.set_ylabel("Connections", color="#F44336")
        ax3.grid(alpha=0.3)
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_r.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        ax4 = fig.add_subplot(gs[1, 1])
        _draw_network(ax4, self.best_genome)
        ax4.set_title("Best Network Topology")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()
        plt.close()


def _draw_network(ax, genome: Genome):
    """Simple network topology visualiser."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    inp  = sorted(genome.input_ids)
    out  = sorted(genome.output_ids)
    hid  = sorted(genome.hidden_ids)

    positions = {}
    for i, nid in enumerate(inp):
        positions[nid] = (0.0, (i + 1) / (len(inp) + 1))
    for i, nid in enumerate(out):
        positions[nid] = (1.0, (i + 1) / (len(out) + 1))
    for i, nid in enumerate(hid):
        positions[nid] = (0.5 + 0.05 * (i % 3 - 1), (i + 1) / (len(hid) + 1))

    for conn in genome.connections.values():
        if not conn.enabled: continue
        if conn.in_node not in positions or conn.out_node not in positions: continue
        x1, y1 = positions[conn.in_node]
        x2, y2 = positions[conn.out_node]
        color = "#2196F3" if conn.weight >= 0 else "#F44336"
        lw = min(abs(conn.weight), 3.0)
        # recurrent: draw curved
        if conn.in_node == conn.out_node or (x2 <= x1 and conn.in_node != conn.out_node):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                        connectionstyle="arc3,rad=0.4"))
        else:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    colors = {nid: "#4CAF50" for nid in inp}
    colors.update({nid: "#FF9800" for nid in out})
    colors.update({nid: "#9C27B0" for nid in hid})

    for nid, (x, y) in positions.items():
        c = plt.Circle((x, y), 0.03, color=colors.get(nid, "gray"), zorder=5)
        ax.add_patch(c)
        ax.text(x, y + 0.05, str(nid), ha="center", va="bottom", fontsize=6)

    ax.set_xlim(-0.1, 1.1); ax.set_ylim(0, 1.1)
    ax.set_aspect("equal"); ax.axis("off")
    legend = [
        mpatches.Patch(color="#4CAF50", label="Input"),
        mpatches.Patch(color="#FF9800", label="Output"),
        mpatches.Patch(color="#9C27B0", label="Hidden"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="lower right")


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class NEATSimulator:
    """
    Full NEAT simulator with speciation and innovation tracking.

    Parameters
    ----------
    cfg        : NEATConfig
    fitness_fn : Callable[[Genome], float]
                 Receives a Genome, must return a float.
                 Use Network.from_genome(genome) inside to build and run the net.
    """

    def __init__(self, cfg: NEATConfig, fitness_fn: Callable[[Genome], float]):
        self.cfg        = cfg
        self.fitness_fn = fitness_fn
        self.rng        = np.random.default_rng(cfg.seed)
        self.tracker    = InnovationTracker()
        self.speciator  = SpeciationManager(
            compatibility_threshold=cfg.compatibility_threshold,
            c1=cfg.c1, c2=cfg.c2, c3=cfg.c3,
            stagnation_limit=cfg.stagnation_limit,
        )
        self._genome_counter = 0

    def _new_id(self) -> int:
        self._genome_counter += 1
        return self._genome_counter

    def run(self) -> NEATResult:
        cfg = self.cfg
        t0  = time.perf_counter()

        # ── initialise population ─────────────────────────────────────────────
        population: List[Genome] = [
            make_genome(
                genome_id=self._new_id(),
                n_inputs=cfg.n_inputs,
                n_outputs=cfg.n_outputs,
                tracker=self.tracker,
                rng=self.rng,
                output_activation=cfg.output_activation,
                initial_connection=cfg.initial_connection,
                initial_weight_std=cfg.initial_weight_std,
            )
            for _ in range(cfg.population_size)
        ]

        history: List[GenerationRecord] = []
        best_genome:  Optional[Genome] = None
        best_fitness: float = -np.inf
        stop_reason = "max_generations"

        for gen in range(cfg.max_generations):
            self.tracker.new_generation()

            # ── evaluate ──────────────────────────────────────────────────────
            for genome in population:
                genome.fitness = float(self.fitness_fn(genome))

            # ── track best ────────────────────────────────────────────────────
            gen_best = max(population, key=lambda g: g.fitness)
            if gen_best.fitness > best_fitness:
                best_fitness = gen_best.fitness
                best_genome  = gen_best.copy(gen_best.genome_id)

            mean_fit = float(np.mean([g.fitness for g in population]))

            # ── speciate ──────────────────────────────────────────────────────
            self.speciator.speciate(population, self.rng)
            self.speciator.cull(cfg.survival_ratio)

            # ── record ────────────────────────────────────────────────────────
            history.append(GenerationRecord(
                generation=gen,
                best_fitness=gen_best.fitness,
                mean_fitness=mean_fit,
                n_species=len(self.speciator.species),
                n_nodes_best=len(gen_best.nodes),
                n_conns_best=len(gen_best.connections),
            ))

            # ── log ───────────────────────────────────────────────────────────
            if cfg.verbose and gen % cfg.print_every == 0:
                print(f"[Gen {gen:>4}]  best={gen_best.fitness:>10.5f}"
                      f"  mean={mean_fit:>10.5f}"
                      f"  species={len(self.speciator.species):>3}"
                      f"  nodes={len(gen_best.nodes):>3}"
                      f"  conns={len(gen_best.connections):>3}")

            # ── stopping ──────────────────────────────────────────────────────
            if cfg.fitness_threshold is not None and best_fitness >= cfg.fitness_threshold:
                stop_reason = "fitness_threshold_reached"
                break

            # ── reproduce ─────────────────────────────────────────────────────
            population = self._reproduce(population)

        elapsed = time.perf_counter() - t0
        if cfg.verbose:
            print(f"\nStopped: {stop_reason}  |  gens={len(history)}"
                  f"  |  best={best_fitness:.6f}")

        return NEATResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            generations_run=len(history),
            stop_reason=stop_reason,
            history=history,
            elapsed_seconds=elapsed,
        )

    def _reproduce(self, population: List[Genome]) -> List[Genome]:
        cfg = self.cfg
        allocations = self.speciator.allocate_offspring(
            cfg.population_size, cfg.elitism_per_species
        )
        next_gen: List[Genome] = []

        for sp_id, n_offspring in allocations.items():
            if sp_id not in self.speciator.species:
                continue
            sp = self.speciator.species[sp_id]
            members = sp.members
            if not members:
                continue

            members.sort(key=lambda g: g.fitness, reverse=True)

            # elitism: champion skips mutation
            elite_n = min(cfg.elitism_per_species, len(members))
            for i in range(elite_n):
                if len(next_gen) < cfg.population_size:
                    next_gen.append(members[i].copy(self._new_id()))

            # breed the rest
            for _ in range(n_offspring - elite_n):
                if len(next_gen) >= cfg.population_size:
                    break

                if len(members) > 1 and self.rng.random() < cfg.crossover_rate:
                    p1, p2 = self.rng.choice(members, size=2, replace=False)
                    child = crossover(p1, p2, self._new_id(), self.rng,
                                      cfg.disable_gene_prob)
                else:
                    child = members[int(self.rng.integers(0, len(members)))].copy(self._new_id())

                child = mutate(
                    child, self.tracker, self.rng,
                    weight_mutate_rate=cfg.weight_mutate_rate,
                    weight_replace_rate=cfg.weight_replace_rate,
                    weight_perturb_std=cfg.weight_perturb_std,
                    bias_mutate_rate=cfg.bias_mutate_rate,
                    add_connection_rate=cfg.add_connection_rate,
                    add_node_rate=cfg.add_node_rate,
                    toggle_rate=cfg.toggle_rate,
                    activation_mutate_rate=cfg.activation_mutate_rate,
                    allow_recurrent=cfg.allow_recurrent,
                )
                next_gen.append(child)

        # top up if rounding left us short
        while len(next_gen) < cfg.population_size:
            src = population[int(self.rng.integers(0, len(population)))]
            next_gen.append(mutate(src.copy(self._new_id()), self.tracker, self.rng))

        return next_gen[:cfg.population_size]
