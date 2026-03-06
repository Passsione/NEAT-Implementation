"""
config.py — Simulation Parameters
===================================
All knobs and dials for the evolution simulation.
Edit these values to control every aspect of the run.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional
import numpy as np


@dataclass
class PopulationConfig:
    size: int = 200             # individuals per generation
    genome_length: int = 20     # genes per individual
    gene_min: float = -5.0      # lower gene bound
    gene_max: float = 5.0       # upper gene bound
    init_strategy: str = "uniform"  # "uniform" | "gaussian" | "binary"


@dataclass
class SelectionConfig:
    method: str = "tournament"  # "tournament" | "roulette" | "rank" | "truncation"
    tournament_size: int = 5
    truncation_ratio: float = 0.4
    elitism_count: int = 2      # top N individuals always survive


@dataclass
class MutationConfig:
    rate: float = 0.02          # per-gene mutation probability
    strategy: str = "gaussian"  # "gaussian" | "uniform" | "flip" | "creep"
    gaussian_sigma: float = 0.3
    creep_step: float = 0.1
    adaptive: bool = False      # auto-boost rate when diversity collapses


@dataclass
class CrossoverConfig:
    rate: float = 0.8           # crossover probability per pair
    method: str = "single_point"  # "single_point" | "two_point" | "uniform" | "blend"
    uniform_prob: float = 0.5
    blend_alpha: float = 0.5    # BLX-α alpha


@dataclass
class FitnessConfig:
    # built-ins: "sphere" | "rastrigin" | "rosenbrock" | "ackley" | "griewank" | "schwefel"
    function: str = "rastrigin"
    custom_fn: Optional[Callable[[np.ndarray], float]] = None  # used when function="custom"
    maximize: bool = False      # False = minimisation


@dataclass
class StoppingConfig:
    max_generations: int = 500
    target_fitness: Optional[float] = None      # stop when best reaches this
    stagnation_limit: Optional[int] = 50        # stop after N gens without improvement
    stagnation_tolerance: float = 1e-6


@dataclass
class LoggingConfig:
    verbose: bool = True
    print_every: int = 10
    track_diversity: bool = True
    save_history: bool = True


@dataclass
class SimConfig:
    """Master config — edit any sub-config to control the simulation."""
    population: PopulationConfig = field(default_factory=PopulationConfig)
    selection:  SelectionConfig  = field(default_factory=SelectionConfig)
    mutation:   MutationConfig   = field(default_factory=MutationConfig)
    crossover:  CrossoverConfig  = field(default_factory=CrossoverConfig)
    fitness:    FitnessConfig    = field(default_factory=FitnessConfig)
    stopping:   StoppingConfig   = field(default_factory=StoppingConfig)
    logging:    LoggingConfig    = field(default_factory=LoggingConfig)
    seed: Optional[int] = 42

