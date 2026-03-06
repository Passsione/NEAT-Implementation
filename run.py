"""
run.py - Configure and run your NEAT simulation
=================================================
MODE = "train"      silent training, no window
MODE = "watch"      evolve with live visualiser each generation
MODE = "watch_only" watch a random population, no evolution
"""

import numpy as np
from neat import NEATConfig, NEATSimulator
from neat_network import Network
from neat_genome import Genome

MODE = "watch"   # "train" | "watch" | "watch_only"


# ══════════════════════════════════════════════════════════════════════════════
#  YOUR FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
# Inputs:  observation from your environment (size = cfg.n_inputs)
# Outputs: network actions              (size = cfg.n_outputs)
# Returns: float — higher is better

def my_fitness_fn(genome: Genome) -> float:
    from visualiser import AgentEnv
    import pygame
    net = Network.from_genome(genome)
    env = AgentEnv(np.random.default_rng(0), goals=[], obstacles=[])
    obs = env.reset()
    net.reset()
    total = 0.0
    for _ in range(200):
        action = net.activate(obs, n_passes=2)
        obs, reward, done = env.step(action)
        total += reward
        if done:
            break
    return total


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

cfg = NEATConfig(
    n_inputs  = 6,          # must match env observation size
    n_outputs = 2,          # must match env action size
    output_activation  = "sigmoid",
    initial_connection = "partial",
    allow_recurrent    = True,

    population_size     = 250,
    elitism_per_species = 1,

    weight_mutate_rate    = 0.8,
    weight_replace_rate   = 0.1,
    weight_perturb_std    = 0.3,
    bias_mutate_rate      = 0.3,
    add_connection_rate   = 0.05,
    add_node_rate         = 0.03,
    toggle_rate           = 0.01,
    activation_mutate_rate = 0.01,

    crossover_rate     = 0.75,
    disable_gene_prob  = 0.75,

    compatibility_threshold = 3.0,
    stagnation_limit   = 15,
    survival_ratio     = 0.5,

    max_generations    = 200,
    fitness_threshold  = None,

    verbose      = True,
    print_every  = 1,
    seed         = 42,
)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if MODE == "train":
        sim    = NEATSimulator(cfg, fitness_fn=my_fitness_fn)
        result = sim.run()
        print(result.summary())
        result.plot(save_path="neat_results.png", show=True)

    elif MODE == "watch_only":
        from visualiser import NEATVisualiser
        from neat_genome import make_genome, InnovationTracker
        tracker = InnovationTracker()
        rng     = np.random.default_rng(cfg.seed)
        pop = [make_genome(i + 1, cfg.n_inputs, cfg.n_outputs,
                           tracker, rng, cfg.output_activation,
                           cfg.initial_connection)
               for i in range(cfg.population_size)]
        vis = NEATVisualiser(pop, n_inputs=cfg.n_inputs, n_outputs=cfg.n_outputs,
                             steps_per_gen=300, fps_target=60)
        vis.run()

    elif MODE == "watch":
        from visualiser import NEATVisualiser
        from neat_genome import make_genome, InnovationTracker

        tracker    = InnovationTracker()
        rng        = np.random.default_rng(cfg.seed)
        population = [make_genome(i + 1, cfg.n_inputs, cfg.n_outputs,
                                  tracker, rng, cfg.output_activation,
                                  cfg.initial_connection)
                      for i in range(cfg.population_size)]

        vis = NEATVisualiser(
            population,
            n_inputs      = cfg.n_inputs,
            n_outputs     = cfg.n_outputs,
            steps_per_gen = 300,
            fps_target    = 60,
        )

        sim       = NEATSimulator(cfg, fitness_fn=my_fitness_fn)
        best_fit  = -np.inf
        best_ever = None

        for gen in range(cfg.max_generations):
            vis.update_population(population)
            rewards = vis.run_generation(gen)

            for genome, r in zip(population, rewards):
                genome.fitness = r

            gen_best = max(population, key=lambda g: g.fitness)
            if gen_best.fitness > best_fit:
                best_fit  = gen_best.fitness
                best_ever = gen_best.copy(gen_best.genome_id)

            print(f"[Gen {gen+1:>3}]  best={gen_best.fitness:.4f}"
                  f"  mean={np.mean(rewards):.4f}"
                  f"  nodes={len(gen_best.nodes)}"
                  f"  conns={len(gen_best.connections)}")

            import pygame
            if not pygame.get_init() or not pygame.display.get_surface():
                break

            if cfg.fitness_threshold and best_fit >= cfg.fitness_threshold:
                break

            tracker.new_generation()
            sim.speciator.speciate(population, rng)
            sim.speciator.cull(cfg.survival_ratio)
            population = sim._reproduce(population)

        print(f"\nDone. Best fitness: {best_fit:.4f}")
