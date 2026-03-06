"""
evo_sim.py — Evolution Simulation Engine
==========================================
All internals in one place: config, fitness functions, population,
selection, crossover, mutation, simulator, and plotting.

Quickstart
----------
    from evo_sim import SimConfig, EvolutionSimulator

    cfg = SimConfig()               # all defaults
    result = EvolutionSimulator(cfg).run()
    print(result.summary())
    result.plot()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import(
    SimConfig, FitnessConfig, CrossoverConfig,
    MutationConfig, SelectionConfig, PopulationConfig 
)

# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _sphere(g):     return float(np.sum(g ** 2))
def _rastrigin(g):  return float(10 * len(g) + np.sum(g**2 - 10 * np.cos(2 * np.pi * g)))
def _rosenbrock(g): return float(np.sum(100 * (g[1:] - g[:-1]**2)**2 + (1 - g[:-1])**2))
def _ackley(g):
    n = len(g)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(g**2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * g)) / n) + 20 + np.e)
def _griewank(g):
    return float(np.sum(g**2) / 4000
                 - np.prod(np.cos(g / np.sqrt(np.arange(1, len(g) + 1)))) + 1)
def _schwefel(g):
    return float(418.9829 * len(g) - np.sum(g * np.sin(np.sqrt(np.abs(g)))))

_FITNESS_REGISTRY = {
    "sphere": _sphere, "rastrigin": _rastrigin, "rosenbrock": _rosenbrock,
    "ackley": _ackley, "griewank": _griewank,   "schwefel": _schwefel,
}

def _get_fitness_fn(cfg: FitnessConfig) -> Callable[[np.ndarray], float]:
    if cfg.function == "custom":
        if cfg.custom_fn is None:
            raise ValueError("fitness.function='custom' but fitness.custom_fn is None.")
        return cfg.custom_fn
    if cfg.function not in _FITNESS_REGISTRY:
        raise ValueError(f"Unknown fitness '{cfg.function}'. Options: {list(_FITNESS_REGISTRY)}")
    return _FITNESS_REGISTRY[cfg.function]


# ══════════════════════════════════════════════════════════════════════════════
#  POPULATION
# ══════════════════════════════════════════════════════════════════════════════

def _init_population(cfg: PopulationConfig, rng: np.random.Generator) -> np.ndarray:
    lo, hi, n, g = cfg.gene_min, cfg.gene_max, cfg.size, cfg.genome_length
    if cfg.init_strategy == "uniform":   return rng.uniform(lo, hi, (n, g))
    if cfg.init_strategy == "gaussian":  return np.clip(rng.standard_normal((n, g)), lo, hi)
    if cfg.init_strategy == "binary":    return rng.choice([lo, hi], (n, g))
    raise ValueError(f"Unknown init_strategy '{cfg.init_strategy}'.")

def _diversity(population: np.ndarray) -> float:
    return float(np.mean(np.std(population, axis=0)))


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _select_parents(pop, scores, cfg: SelectionConfig, rng, n, maximize) -> np.ndarray:
    s = scores if maximize else -scores
    if cfg.method == "tournament":
        out = np.empty((n, pop.shape[1]))
        for i in range(n):
            idx = rng.integers(0, len(pop), cfg.tournament_size)
            out[i] = pop[idx[np.argmax(s[idx])]]
        return out
    if cfg.method == "roulette":
        p = (s - s.min() + 1e-10); p /= p.sum()
        return pop[rng.choice(len(pop), n, p=p)]
    if cfg.method == "rank":
        ranks = np.argsort(np.argsort(s)) + 1; p = ranks / ranks.sum()
        return pop[rng.choice(len(pop), n, p=p)]
    if cfg.method == "truncation":
        k = max(1, int(len(pop) * cfg.truncation_ratio))
        return pop[rng.choice(np.argsort(s)[-k:], n)]
    raise ValueError(f"Unknown selection method '{cfg.method}'.")


# ══════════════════════════════════════════════════════════════════════════════
#  CROSSOVER
# ══════════════════════════════════════════════════════════════════════════════

def _crossover(p1, p2, cfg: CrossoverConfig, rng):
    if rng.random() > cfg.rate:
        return p1.copy(), p2.copy()
    L = len(p1)
    if cfg.method == "single_point":
        pt = rng.integers(1, L)
        return np.r_[p1[:pt], p2[pt:]], np.r_[p2[:pt], p1[pt:]]
    if cfg.method == "two_point":
        a, b = sorted(rng.integers(1, L, 2))
        return np.r_[p1[:a], p2[a:b], p1[b:]], np.r_[p2[:a], p1[a:b], p2[b:]]
    if cfg.method == "uniform":
        m = rng.random(L) < cfg.uniform_prob
        return np.where(m, p1, p2), np.where(m, p2, p1)
    if cfg.method == "blend":
        lo = np.minimum(p1, p2) - cfg.blend_alpha * np.abs(p1 - p2)
        hi = np.maximum(p1, p2) + cfg.blend_alpha * np.abs(p1 - p2)
        return rng.uniform(lo, hi), rng.uniform(lo, hi)
    raise ValueError(f"Unknown crossover method '{cfg.method}'.")


# ══════════════════════════════════════════════════════════════════════════════
#  MUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _mutate(genome, cfg: MutationConfig, pop_cfg: PopulationConfig, rng, rate=None):
    rate = rate or cfg.rate
    child = genome.copy()
    mask = rng.random(len(child)) < rate
    if not np.any(mask): return child
    if cfg.strategy == "gaussian":
        child[mask] += rng.normal(0, cfg.gaussian_sigma, mask.sum())
    elif cfg.strategy == "uniform":
        child[mask] = rng.uniform(pop_cfg.gene_min, pop_cfg.gene_max, mask.sum())
    elif cfg.strategy == "flip":
        child[mask] = np.where(child[mask] == pop_cfg.gene_min, pop_cfg.gene_max, pop_cfg.gene_min)
    elif cfg.strategy == "creep":
        child[mask] += rng.choice([-1, 1], mask.sum()) * cfg.creep_step
    else:
        raise ValueError(f"Unknown mutation strategy '{cfg.strategy}'.")
    return np.clip(child, pop_cfg.gene_min, pop_cfg.gene_max)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float


@dataclass
class SimResult:
    best_genome: np.ndarray
    best_fitness: float
    generations_run: int
    stop_reason: str
    history: List[GenerationRecord]
    elapsed_seconds: float

    def summary(self) -> str:
        return "\n".join([
            "=" * 50,
            "  Evolution Simulation — Results",
            "=" * 50,
            f"  Generations : {self.generations_run}",
            f"  Stop reason : {self.stop_reason}",
            f"  Best fitness: {self.best_fitness:.6f}",
            f"  Elapsed     : {self.elapsed_seconds:.2f}s",
            f"  Best genome (first 8): {self.best_genome[:8].round(4).tolist()}",
            "=" * 50,
        ])

    def plot(self, title: str = "Evolution Simulation", save_path: str = None, show: bool = True):
        """4-panel plot: fitness curves, diversity, improvement delta, genome heatmap."""
        h = self.history
        gens  = [r.generation   for r in h]
        best  = [r.best_fitness  for r in h]
        mean  = [r.mean_fitness  for r in h]
        worst = [r.worst_fitness for r in h]
        divs  = [r.diversity     for r in h]

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(gens, best,  label="Best",  color="#2196F3", lw=2)
        ax1.plot(gens, mean,  label="Mean",  color="#FF9800", lw=1.5, ls="--")
        ax1.plot(gens, worst, label="Worst", color="#F44336", lw=1,   ls=":")
        ax1.set(title="Fitness Over Generations", xlabel="Generation", ylabel="Fitness")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(gens, divs, color="#4CAF50", lw=2)
        ax2.fill_between(gens, divs, alpha=0.2, color="#4CAF50")
        ax2.set(title="Population Diversity", xlabel="Generation", ylabel="Mean gene std")
        ax2.grid(alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        deltas = np.diff(best)
        ax3.bar(gens[1:], deltas,
                color=["#2196F3" if d <= 0 else "#F44336" for d in deltas],
                width=max(1, len(gens) // 200), alpha=0.7)
        ax3.axhline(0, color="black", lw=0.8)
        ax3.set(title="Per-Generation Improvement", xlabel="Generation", ylabel="Δ Best Fitness")
        ax3.grid(alpha=0.3, axis="y")

        ax4 = fig.add_subplot(gs[1, 1])
        g = self.best_genome
        cols = min(10, len(g)); rows = int(np.ceil(len(g) / cols))
        pad = np.full(rows * cols, np.nan); pad[:len(g)] = g
        im = ax4.imshow(pad.reshape(rows, cols), cmap="RdYlGn", aspect="auto")
        ax4.set(title=f"Best Genome  (fitness={self.best_fitness:.5f})",
                xlabel="Gene (col)", ylabel="Gene (row)")
        plt.colorbar(im, ax=ax4, shrink=0.8)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()
        plt.close()

    def print_table(self, every: int = 10):
        print(f"\n{'Gen':>5}  {'Best':>14}  {'Mean':>14}  {'Diversity':>10}")
        print("-" * 50)
        for r in self.history:
            if r.generation % every == 0:
                print(f"{r.generation:>5}  {r.best_fitness:>14.6f}"
                      f"  {r.mean_fitness:>14.6f}  {r.diversity:>10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class EvolutionSimulator:
    """
    Main simulation engine.

    Parameters
    ----------
    cfg : SimConfig

    Example
    -------
    >>> sim = EvolutionSimulator(SimConfig())
    >>> result = sim.run()
    >>> print(result.summary())
    >>> result.plot()
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.fitness_fn = _get_fitness_fn(cfg.fitness)

    def run(self) -> SimResult:
        cfg = self.cfg
        t0 = time.perf_counter()

        pop = _init_population(cfg.population, self.rng)
        history: List[GenerationRecord] = []
        best_fitness = np.inf if not cfg.fitness.maximize else -np.inf
        best_genome: Optional[np.ndarray] = None
        stagnation = 0
        stop_reason = "max_generations"

        for gen in range(cfg.stopping.max_generations):

            scores = np.array([self.fitness_fn(ind) for ind in pop])

            best_idx = int(np.argmax(scores) if cfg.fitness.maximize else np.argmin(scores))
            gen_best = float(scores[best_idx])
            improved = (gen_best > best_fitness) if cfg.fitness.maximize else (gen_best < best_fitness)

            if improved and abs(gen_best - best_fitness) > cfg.stopping.stagnation_tolerance:
                stagnation = 0
                best_fitness = gen_best
                best_genome = pop[best_idx].copy()
            else:
                stagnation += 1

            if cfg.logging.save_history:
                history.append(GenerationRecord(
                    generation=gen,
                    best_fitness=gen_best,
                    mean_fitness=float(np.mean(scores)),
                    worst_fitness=float(np.max(scores) if not cfg.fitness.maximize else np.min(scores)),
                    diversity=_diversity(pop) if cfg.logging.track_diversity else 0.0,
                ))

            if cfg.logging.verbose and gen % cfg.logging.print_every == 0:
                div = f"  diversity={history[-1].diversity:.4f}" if history else ""
                print(f"[Gen {gen:>4}]  best={gen_best:>12.5f}  mean={np.mean(scores):>12.5f}{div}")

            if cfg.stopping.target_fitness is not None:
                hit = best_fitness >= cfg.stopping.target_fitness if cfg.fitness.maximize \
                      else best_fitness <= cfg.stopping.target_fitness
                if hit: stop_reason = "target_fitness_reached"; break

            if cfg.stopping.stagnation_limit and stagnation >= cfg.stopping.stagnation_limit:
                stop_reason = "stagnation"; break

            pop = self._evolve(pop, scores)

        if cfg.logging.verbose:
            print(f"\nStopped: {stop_reason}  |  gens={len(history)}  |  best={best_fitness:.6f}")

        return SimResult(
            best_genome=best_genome if best_genome is not None else pop[0],
            best_fitness=best_fitness,
            generations_run=len(history),
            stop_reason=stop_reason,
            history=history,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def _evolve(self, pop: np.ndarray, scores: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        new_pop = np.empty_like(pop)
        elite_n = cfg.selection.elitism_count

        if elite_n > 0:
            elite_idx = np.argsort(scores)[-elite_n:] if cfg.fitness.maximize \
                        else np.argsort(scores)[:elite_n]
            new_pop[:elite_n] = pop[elite_idx]

        mut_rate = cfg.mutation.rate
        if cfg.mutation.adaptive and _diversity(pop) < 0.05:
            mut_rate = min(mut_rate * 3.0, 0.5)

        needed = cfg.population.size - elite_n
        parents = _select_parents(pop, scores, cfg.selection, self.rng,
                                  needed * 2, cfg.fitness.maximize)

        i, child_idx = 0, elite_n
        while child_idx < cfg.population.size:
            c1, c2 = _crossover(parents[i % len(parents)], parents[(i+1) % len(parents)],
                                 cfg.crossover, self.rng)
            c1 = _mutate(c1, cfg.mutation, cfg.population, self.rng, mut_rate)
            c2 = _mutate(c2, cfg.mutation, cfg.population, self.rng, mut_rate)
            new_pop[child_idx] = c1; child_idx += 1
            if child_idx < cfg.population.size:
                new_pop[child_idx] = c2; child_idx += 1
            i += 2

        return np.clip(new_pop, cfg.population.gene_min, cfg.population.gene_max)
