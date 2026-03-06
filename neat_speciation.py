"""
neat_speciation.py - Species Management
========================================
Genomes are grouped into species by structural similarity.
Species protect new structural innovations from immediate elimination
by only competing within their own species.

Compatibility distance:
  delta = (c1 * E / N) + (c2 * D / N) + c3 * W_bar

  E = excess genes, D = disjoint genes, N = genes in larger genome,
  W_bar = mean weight difference of matching genes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from neat_genome import Genome


# ══════════════════════════════════════════════════════════════════════════════
#  COMPATIBILITY DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def compatibility_distance(
    g1: Genome, g2: Genome,
    c1: float = 1.0,   # excess gene coefficient
    c2: float = 1.0,   # disjoint gene coefficient
    c3: float = 0.4,   # weight difference coefficient
) -> float:
    """Compute the NEAT compatibility distance between two genomes."""
    innov1 = set(g1.connections.keys())
    innov2 = set(g2.connections.keys())

    if not innov1 and not innov2:
        return 0.0

    max1 = max(innov1) if innov1 else 0
    max2 = max(innov2) if innov2 else 0

    matching_weights = []
    disjoint = 0
    excess   = 0

    all_innovs = innov1 | innov2
    excess_threshold = min(max1, max2)

    for innov in all_innovs:
        in1 = innov in innov1
        in2 = innov in innov2
        if in1 and in2:
            matching_weights.append(
                abs(g1.connections[innov].weight - g2.connections[innov].weight)
            )
        else:
            if innov > excess_threshold:
                excess += 1
            else:
                disjoint += 1

    N = max(len(innov1), len(innov2), 1)
    W_bar = float(np.mean(matching_weights)) if matching_weights else 0.0

    return (c1 * excess / N) + (c2 * disjoint / N) + c3 * W_bar


# ══════════════════════════════════════════════════════════════════════════════
#  SPECIES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Species:
    species_id:    int
    representative: Genome          # used for distance comparisons
    members:       List[Genome] = field(default_factory=list)
    best_fitness:  float = -np.inf
    stagnation:    int   = 0        # gens without improvement
    age:           int   = 0

    def update_representative(self, rng: np.random.Generator):
        if self.members:
            self.representative = rng.choice(self.members)

    def adjusted_fitnesses(self) -> List[float]:
        """Fitness sharing: divide each genome's fitness by species size."""
        n = len(self.members)
        return [m.fitness / n for m in self.members]

    def total_adjusted_fitness(self) -> float:
        return sum(self.adjusted_fitnesses())


# ══════════════════════════════════════════════════════════════════════════════
#  SPECIATION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class SpeciationManager:
    """
    Maintains the set of species across generations.

    Each generation:
      1. speciate(population)  — assign every genome to a species
      2. compute offspring counts via allocate_offspring()
      3. cull weak performers within each species
    """

    def __init__(
        self,
        compatibility_threshold: float = 3.0,
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 0.4,
        stagnation_limit: int = 15,       # species dies after N gens without improvement
        min_species_size: int = 2,
    ):
        self.threshold          = compatibility_threshold
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.stagnation_limit   = stagnation_limit
        self.min_species_size   = min_species_size

        self.species: Dict[int, Species] = {}
        self._next_species_id = 1

    def speciate(self, population: List[Genome], rng: np.random.Generator):
        """Assign each genome in population to the nearest species."""
        # clear members (keep representatives)
        for sp in self.species.values():
            sp.members = []

        for genome in population:
            placed = False
            for sp in self.species.values():
                dist = compatibility_distance(
                    genome, sp.representative, self.c1, self.c2, self.c3
                )
                if dist < self.threshold:
                    sp.members.append(genome)
                    genome.species_id = sp.species_id
                    placed = True
                    break

            if not placed:
                sp_id = self._next_species_id
                self._next_species_id += 1
                new_sp = Species(species_id=sp_id, representative=genome, members=[genome])
                genome.species_id = sp_id
                self.species[sp_id] = new_sp

        # remove empty species
        self.species = {k: v for k, v in self.species.items() if v.members}

        # update representatives and stagnation
        for sp in self.species.values():
            sp.age += 1
            sp.update_representative(rng)
            gen_best = max(m.fitness for m in sp.members)
            if gen_best > sp.best_fitness + 1e-6:
                sp.best_fitness = gen_best
                sp.stagnation   = 0
            else:
                sp.stagnation  += 1

    def allocate_offspring(
        self,
        population_size: int,
        elitism_per_species: int = 1,
    ) -> Dict[int, int]:
        """
        Compute how many offspring each species gets next generation,
        proportional to its total adjusted fitness.
        Stagnant species are penalised / removed.
        Minimum 1 offspring for any surviving species.
        """
        surviving = {
            sid: sp for sid, sp in self.species.items()
            if sp.stagnation < self.stagnation_limit or len(sp.members) <= 1
        }

        totals = {sid: max(sp.total_adjusted_fitness(), 1e-6)
                  for sid, sp in surviving.items()}
        grand_total = sum(totals.values())

        allocations: Dict[int, int] = {}
        for sid, total in totals.items():
            allocations[sid] = max(1, int(round(population_size * total / grand_total)))

        if not allocations:
            # all species stagnant — return empty; simulator will top up from population
            return {}

        # fix rounding
        diff = population_size - sum(allocations.values())
        if diff != 0:
            largest = max(allocations, key=allocations.get)
            allocations[largest] += diff

        return allocations

    def cull(self, survival_ratio: float = 0.5, min_survivors: int = 2):
        """Remove the bottom fraction of each species by fitness."""
        for sp in self.species.values():
            sp.members.sort(key=lambda g: g.fitness, reverse=True)
            keep = max(min_survivors, int(np.ceil(len(sp.members) * survival_ratio)))
            sp.members = sp.members[:keep]
