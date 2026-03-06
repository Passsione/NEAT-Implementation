"""
visualiser.py  —  NEAT Pygame Visualiser
==========================================
Full interactive debugging UI for your NEAT population.

Controls
--------
  SPACE          pause / resume
  R              reset generation
  LEFT / RIGHT   step one frame when paused
  +  /  -        speed up / slow down
  T              toggle trails
  G              toggle goal markers
  ESC / Q        quit

Mouse — World panel
-------------------
  Hover  agent       peek at network/stats in inspector panel
  Click  agent       pin inspector to that agent (click again to unpin)
  Middle-drag        drag a pinned agent around
  Left-click  empty  place a GOAL  (click again to remove)
  Right-click empty  place an OBSTACLE (click again to remove)

Inspector panel (right side)
-----------------------------
  Genome stats      — nodes, connections, species, fitness, innovation
  Input bars        — live bar chart of what the network currently sees
  Output bars       — live bar chart of what the network is outputting
  Network diagram   — node graph with activation brightness + weighted edges
                      recurrent connections shown as curved arcs

Bottom strip
------------
  Sparkline of best fitness per generation.

Swapping in your real environment
----------------------------------
  1. Replace AgentEnv with your own class implementing:
       reset()  -> List[float]
       step(action: List[float]) -> (obs: List[float], reward: float, done: bool)
     Keep env.x, env.y, env.trail_x, env.trail_y, env.reward accessible.
  2. Update cfg.n_inputs / cfg.n_outputs in run.py to match.
"""

from __future__ import annotations

import math
import time
import colorsys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
import pygame.gfxdraw

from neat_genome import Genome, NODE_INPUT, NODE_OUTPUT, NODE_HIDDEN
from neat_network import Network


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":           (10,  12,  20),
    "panel":        (16,  18,  30),
    "border":       (35,  38,  60),
    "text":         (180, 185, 210),
    "text_dim":     (90,  95, 120),
    "text_bright":  (230, 235, 255),
    "accent":       (80,  180, 255),
    "accent2":      (255, 140,  60),
    "green":        (60,  220, 120),
    "red":          (255,  70,  70),
    "yellow":       (255, 210,  50),
    "purple":       (160,  80, 255),
    "agent_best":   (255, 215,   0),
    "agent_hover":  (255, 255, 255),
    "trail":        (80,  180, 255),
    "goal":         (60,  220, 120),
    "obstacle":     (200,  50,  50),
    "grid":         (20,  22,  36),
    "spark_line":   (80,  180, 255),
    "spark_fill":   (20,  40,  80),
    "node_input":   (60,  180, 255),
    "node_output":  (255, 140,  60),
    "node_hidden":  (160,  80, 255),
    "edge_pos":     (80,  200, 120),
    "edge_neg":     (220,  80,  80),
    "bar_in":       (60,  180, 255),
    "bar_out":      (255, 140,  60),
    "paused":       (255, 210,  50),
}


# ══════════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER ENVIRONMENT   (replace with your own)
# ══════════════════════════════════════════════════════════════════════════════

class AgentEnv:
    """
    2-D world placeholder.  Agents move by velocity output.

    Replace this class with your own — the visualiser only requires:
        reset()  -> List[float]
        step(action) -> (obs, reward, done)
        .x  .y  .trail_x  .trail_y  .reward
    """

    WORLD_W = 800
    WORLD_H = 600

    def __init__(self, rng: np.random.Generator,
                 goals: List[Tuple[float, float]],
                 obstacles: List[pygame.Rect]):
        self.rng       = rng
        self.goals     = goals
        self.obstacles = obstacles
        self.x = self.y = 400.0
        self.trail_x: deque = deque(maxlen=80)
        self.trail_y: deque = deque(maxlen=80)
        self.reward = 0.0
        self.done   = False

    def reset(self) -> List[float]:
        self.x = float(self.rng.uniform(80, self.WORLD_W - 80))
        self.y = float(self.rng.uniform(80, self.WORLD_H - 80))
        self.trail_x.clear(); self.trail_x.append(self.x)
        self.trail_y.clear(); self.trail_y.append(self.y)
        self.reward = 0.0
        self.done   = False
        return self._obs()

    def step(self, action: List[float]) -> Tuple[List[float], float, bool]:
        speed = 4.0
        dx = (action[0] - 0.5) * 2.0 * speed
        dy = (action[1] - 0.5) * 2.0 * speed if len(action) > 1 else 0.0


        nx = float(np.clip(self.x + dx, 0, self.WORLD_W))
        ny = float(np.clip(self.y + dy, 0, self.WORLD_H))

        # obstacle collision — don't move into obstacle
        pt = pygame.Rect(int(nx) - 3, int(ny) - 3, 6, 6)
        if not any(pt.colliderect(o) for o in self.obstacles):
            self.x, self.y = nx, ny

        self.trail_x.append(self.x)
        self.trail_y.append(self.y)

        # reward
        if self.goals:
            dists = [math.hypot(self.x - gx, self.y - gy) for gx, gy in self.goals]
            self.reward = float(1.0 / (1.0 + min(dists)))
        else:
            self.reward = float(
                math.hypot(self.x - list(self.trail_x)[0],
                           self.y - list(self.trail_y)[0]) / self.WORLD_W
            )

        if self.obstacles:
            dists = [math.hypot(self.x - o.centerx, self.y - o.centery) for o in self.obstacles]
            self.reward = max(0, self.reward - float(1.0 / (1.0 + min(dists))))


        return self._obs(), self.reward, self.done

    def _obs(self) -> List[float]:
        nx = self.x / self.WORLD_W
        ny = self.y / self.WORLD_H

        obSpace = [nx, ny, 0.0, 0.0, 0.0, 0.0]

        if self.goals:
            dists = [(math.hypot(self.x - gx, self.y - gy), gx, gy)
                     for gx, gy in self.goals]
            _, gx, gy = min(dists)
            obSpace[2] = (gx - self.x) / self.WORLD_W
            obSpace[3] = (gy - self.y) / self.WORLD_H
        if self.obstacles:
            dists = [(math.hypot(self.x - o.centerx, self.y - o.centery), o.centerx, o.centery)
                     for o in self.obstacles]
            _, ox, oy = min(dists)
            obSpace[4] = (ox - self.x) / self.WORLD_W
            obSpace[5] = (oy - self.y) / self.WORLD_H

        return obSpace


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    genome:           Genome
    network:          Network
    env:              AgentEnv
    obs:              List[float]       = field(default_factory=list)
    reward:           float             = 0.0
    alive:            bool              = True
    node_activations: Dict[int, float]  = field(default_factory=dict)
    last_inputs:      List[float]       = field(default_factory=list)
    last_outputs:     List[float]       = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISER
# ══════════════════════════════════════════════════════════════════════════════

class NEATVisualiser:
    """
    Parameters
    ----------
    population    : List[Genome]
    n_inputs      : observation size (must match your env / network)
    n_outputs     : action size
    steps_per_gen : timesteps before auto-advancing generation
    fps_target    : frames per second cap
    seed          : rng seed
    """
    AGENT_R  = 5
    WORLD_W  = AgentEnv.WORLD_W + 2 * AGENT_R
    WORLD_H  = AgentEnv.WORLD_H + 2 * AGENT_R
    INSP_W   = 400
    BOT_H    = 90
    WIN_W    = WORLD_W + INSP_W   
    WIN_H    = WORLD_H + BOT_H    
    GOAL_R   = 16

    def __init__(
        self,
        population:    List[Genome],
        n_inputs:      int = 4,
        n_outputs:     int = 2,
        steps_per_gen: int = 300,
        fps_target:    int = 60,
        seed:          int = 0,
    ):
        self.population    = population
        self.n_inputs      = n_inputs
        self.n_outputs     = n_outputs
        self.steps_per_gen = steps_per_gen
        self.fps_target    = fps_target
        self.rng           = np.random.default_rng(seed)

        self.step       = 0
        self.generation = 0
        self.paused     = False
        self.speed      = 1
        self.show_trails = True
        self.show_goals  = True

        self.goals:     List[Tuple[float, float]] = []
        self.obstacles: List[pygame.Rect]         = []

        self.hovered_agent: Optional[Agent] = None
        self.pinned_agent:  Optional[Agent] = None
        self.dragging:      Optional[Agent] = None

        self.gen_best_history: List[float] = []
        self.agents: List[Agent] = []

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption("NEAT Visualiser")
        self.clock = pygame.time.Clock()

        self._fsm  = pygame.font.SysFont("monospace", 11)
        self._fmd  = pygame.font.SysFont("monospace", 13)
        self._flg  = pygame.font.SysFont("monospace", 15, bold=True)
        self._fhd  = pygame.font.SysFont("monospace", 17, bold=True)

        self._build_agents()

    # ── agent construction ────────────────────────────────────────────────────

    def _build_agents(self):
        self.agents = []
        for genome in self.population:
            net = Network.from_genome(genome)
            env = AgentEnv(
                np.random.default_rng(int(self.rng.integers(0, 2**31))),
                self.goals, self.obstacles,
            )
            obs = env.reset()
            net.reset()
            self.agents.append(Agent(
                genome=genome, network=net, env=env,
                obs=obs, last_inputs=list(obs),
                last_outputs=[0.0] * self.n_outputs,
            ))

    def update_population(self, population: List[Genome]):
        self.population = population
        self._build_agents()
        self.step = 0

    # ── public entry points ───────────────────────────────────────────────────

    def run(self, n_generations: int = 9999):
        """Standalone mode — blocks until window closed."""
        running = True
        while running and self.generation < n_generations:
            running = self._handle_events()
            if not self.paused:
                for _ in range(self.speed):
                    self._tick()
            if self.step >= self.steps_per_gen:
                rewards = [a.reward for a in self.agents]
                self.gen_best_history.append(max(rewards) if rewards else 0.0)
                self.generation += 1
                self._build_agents()
                self.step = 0
            self._draw()
            self.clock.tick(self.fps_target)
        pygame.quit()

    def run_generation(self, generation: int) -> List[float]:
        """Integrated mode — call from your NEAT loop each generation."""
        self.generation = generation
        self._build_agents()
        self.step = 0

        running = True
        while running and self.step < self.steps_per_gen:
            running = self._handle_events()
            if not self.paused:
                for _ in range(self.speed):
                    self._tick()
                    if self.step >= self.steps_per_gen:
                        break
            self._draw()
            self.clock.tick(self.fps_target)

        rewards = [a.reward for a in self.agents]
        self.gen_best_history.append(max(rewards) if rewards else 0.0)
        return rewards

    # ── simulation ────────────────────────────────────────────────────────────

    def _tick(self):
        if self.step >= self.steps_per_gen:
            return
        self.step += 1
        for agent in self.agents:
            if not agent.alive:
                continue
            action = agent.network.activate(agent.obs, n_passes=2)
            agent.last_inputs  = list(agent.obs)
            agent.last_outputs = list(action)
            agent.node_activations = dict(agent.network.state)
            obs, reward, done = agent.env.step(action)
            agent.obs    = obs
            agent.reward = reward
            if done:
                agent.alive = False

    # ── events ────────────────────────────────────────────────────────────────

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            elif ev.type == pygame.KEYDOWN:
                if not self._on_key(ev):
                    return False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                self._on_click(ev)
            elif ev.type == pygame.MOUSEBUTTONUP:
                self.dragging = None
            elif ev.type == pygame.MOUSEMOTION:
                self._on_motion(ev)
        return True

    def _on_key(self, ev) -> bool:
        k = ev.key
        if k == pygame.K_q:  return False
        if k == pygame.K_SPACE:   self.paused = not self.paused
        if k == pygame.K_r:       self._build_agents(); self.step = 0
        if k == pygame.K_RIGHT and self.paused: self._tick()
        if k in (pygame.K_PLUS, pygame.K_EQUALS): self.speed = min(self.speed * 2, 16)
        if k == pygame.K_MINUS:   self.speed = max(self.speed // 2, 1)
        if k == pygame.K_t:       self.show_trails = not self.show_trails
        if k == pygame.K_g:       self.show_goals  = not self.show_goals
        return True

    def _on_click(self, ev):
        mx, my = ev.pos
        in_world = mx < self.WORLD_W and my < self.WORLD_H

        if ev.button == 2 and self.pinned_agent:
            self.dragging = self.pinned_agent
            return

        if not in_world:
            return

        if ev.button == 1:
            hit = self._agent_at(mx, my)
            if hit:
                self.pinned_agent = None if hit is self.pinned_agent else hit
            else:
                # toggle goal
                for i, (gx, gy) in enumerate(self.goals):
                    if math.hypot(mx - gx, my - gy) < self.GOAL_R + 6:
                        self.goals.pop(i); return
                self.goals.append((float(mx), float(my)))

        elif ev.button == 3:
            # toggle obstacle
            for i, obs in enumerate(self.obstacles):
                if obs.collidepoint(mx, my):
                    self.obstacles.pop(i); return
            self.obstacles.append(pygame.Rect(mx - 20, my - 20, 40, 40))

    def _on_motion(self, ev):
        mx, my = ev.pos
        if self.dragging and mx < self.WORLD_W:
            self.dragging.env.x = float(mx)
            self.dragging.env.y = float(my)
            return
        if mx < self.WORLD_W and my < self.WORLD_H:
            self.hovered_agent = self._agent_at(mx, my, radius=16)
        else:
            self.hovered_agent = None

    def _agent_at(self, mx, my, radius=12) -> Optional[Agent]:
        best_d, best_a = radius, None
        for a in self.agents:
            d = math.hypot(a.env.x - mx, a.env.y - my)
            if d < best_d:
                best_d, best_a = d, a
        return best_a

    # ══════════════════════════════════════════════════════════════════════════
    #  DRAWING
    # ══════════════════════════════════════════════════════════════════════════

    def _draw(self):
        self.screen.fill(C["bg"])
        self._draw_world()
        self._draw_inspector()
        self._draw_bottom()
        self._draw_hud()
        pygame.display.flip()

    # ── world ─────────────────────────────────────────────────────────────────

    def _draw_world(self):
        surf = pygame.Surface((self.WORLD_W, self.WORLD_H))
        surf.fill(C["bg"])
        self._draw_grid(surf)
        if self.show_goals:    self._draw_goals(surf)
        self._draw_obstacles(surf)
        if self.show_trails:   self._draw_trails(surf)
        self._draw_agents(surf)
        self.screen.blit(surf, (0, 0))
        pygame.draw.rect(self.screen, C["border"],
                         (0, 0, self.WORLD_W, self.WORLD_H), 1)

    def _draw_grid(self, surf):
        for x in range(0, self.WORLD_W + 1, 80):
            pygame.draw.line(surf, C["grid"], (x, 0), (x, self.WORLD_H))
        for y in range(0, self.WORLD_H + 1, 80):
            pygame.draw.line(surf, C["grid"], (0, y), (self.WORLD_W, y))

    def _draw_goals(self, surf):
        t = time.time()
        for gx, gy in self.goals:
            ix, iy = int(gx), int(gy)
            pulse  = self.GOAL_R + int(4 * math.sin(t * 3))
            pygame.gfxdraw.aacircle(surf, ix, iy, pulse,       (*C["goal"], 70))
            pygame.gfxdraw.aacircle(surf, ix, iy, self.GOAL_R, C["goal"])
            pygame.gfxdraw.filled_circle(surf, ix, iy, 5,      C["goal"])
            lbl = self._fsm.render("GOAL", True, C["green"])
            surf.blit(lbl, (ix + self.GOAL_R + 4, iy - 6))

    def _draw_obstacles(self, surf):
        for obs in self.obstacles:
            pygame.draw.rect(surf, (*C["obstacle"], 150), obs)
            pygame.draw.rect(surf, C["obstacle"], obs, 2)
            lbl = self._fsm.render("OBS", True, C["red"])
            surf.blit(lbl, (obs.x + 4, obs.y + obs.height // 2 - 6))

    def _draw_trails(self, surf):
        ts = pygame.Surface((self.WORLD_W, self.WORLD_H), pygame.SRCALPHA)
        focus = self.pinned_agent or self.hovered_agent
        rewards = [a.reward for a in self.agents]
        max_r   = max(rewards) if rewards else 1.0

        for agent in self.agents:
            tx = list(agent.env.trail_x)
            ty = list(agent.env.trail_y)
            if len(tx) < 2:
                continue
            is_focus = agent is focus
            is_best  = agent.reward >= max_r - 1e-9
            base     = C["agent_best"] if is_best else C["trail"]
            amax     = 200 if is_focus else (100 if is_best else 45)
            lw       = 2 if is_focus or is_best else 1
            n        = len(tx)
            for i in range(1, n):
                a = int(amax * i / n)
                pygame.draw.line(ts, (*base, a),
                                 (int(tx[i-1]), int(ty[i-1])),
                                 (int(tx[i]),   int(ty[i])), lw)
        surf.blit(ts, (0, 0))

    def _draw_agents(self, surf):
        focus   = self.pinned_agent or self.hovered_agent
        rewards = [a.reward for a in self.agents]
        max_r   = max(rewards) if rewards else 1.0
        # non-focused first, focused on top
        order   = sorted(self.agents, key=lambda a: a is focus)
        for agent in order:
            self._draw_dot(surf, agent, max_r, agent is focus)

    def _draw_dot(self, surf, agent, max_r, focused):
        x, y   = int(agent.env.x), int(agent.env.y)
        r_norm = agent.reward / max(max_r, 1e-9)
        is_best = agent.reward >= max_r - 1e-9

        if focused:
            col = C["agent_hover"]
            r   = self.AGENT_R + 3
            pygame.gfxdraw.aacircle(surf, x, y, r + 6, (*C["accent"], 140))
            pygame.gfxdraw.aacircle(surf, x, y, r + 6, (*C["accent"], 140))
        elif is_best:
            col = C["agent_best"]
            r   = self.AGENT_R + 2
            pygame.gfxdraw.aacircle(surf, x, y, r + 4, (*C["yellow"], 80))
        else:
            h   = 0.55 + r_norm * 0.28
            rgb = colorsys.hsv_to_rgb(h, 0.85, 0.92)
            col = tuple(int(c * 255) for c in rgb)
            r   = self.AGENT_R

        pygame.gfxdraw.filled_circle(surf, x, y, r, col)
        pygame.gfxdraw.aacircle(surf, x, y, r, col)

        if focused:
            lbl = self._fsm.render(
                f"#{agent.genome.genome_id}  r={agent.reward:.3f}", True, C["text_bright"]
            )
            surf.blit(lbl, (x + r + 5, y - 7))

    # ── inspector ─────────────────────────────────────────────────────────────

    def _draw_inspector(self):
        panel = pygame.Surface((self.INSP_W, self.WORLD_H))
        panel.fill(C["panel"])
        subject = self.pinned_agent or self.hovered_agent

        if subject is None:
            for i, line in enumerate([
                "Hover agent to inspect",
                "Click to pin",
                "Middle-drag to move pinned",
            ]):
                s = self._fmd.render(line, True, C["text_dim"])
                panel.blit(s, (self.INSP_W // 2 - s.get_width() // 2,
                               self.WORLD_H // 2 - 20 + i * 22))
        else:
            y = 10
            y = self._insp_header(panel, subject, y)
            y = self._insp_stats(panel, subject, y)
            y = self._insp_bars(panel, subject, y)
            self._insp_network(panel, subject, y)

        pygame.draw.rect(panel, C["border"], (0, 0, self.INSP_W, self.WORLD_H), 1)
        self.screen.blit(panel, (self.WORLD_W, 0))

    def _insp_header(self, surf, agent, y) -> int:
        pinned = agent is self.pinned_agent
        label  = "PINNED" if pinned else "HOVER"
        col    = C["accent2"] if pinned else C["accent"]
        s = self._fhd.render(f"Agent #{agent.genome.genome_id}  [{label}]", True, col)
        surf.blit(s, (10, y));  y += s.get_height() + 4
        pygame.draw.line(surf, C["border"], (10, y), (self.INSP_W - 10, y))
        return y + 6

    def _insp_stats(self, surf, agent, y) -> int:
        g  = agent.genome
        en = sum(1 for c in g.connections.values() if c.enabled)
        rows = [
            ("Fitness",   f"{agent.reward:.5f}",                  C["green"]),
            ("Nodes",     str(len(g.nodes)),                       C["text"]),
            ("Conns",     f"{en} / {len(g.connections)} enabled",  C["text"]),
            ("Species",   str(g.species_id or "?"),                C["purple"]),
            ("Max innov", str(max(g.connections.keys(), default=0)), C["text_dim"]),
            ("Alive",     str(agent.alive),                        C["green"] if agent.alive else C["red"]),
        ]
        for lbl, val, col in rows:
            ls = self._fmd.render(f"{lbl:<13}", True, C["text_dim"])
            vs = self._fmd.render(val, True, col)
            surf.blit(ls, (12, y));  surf.blit(vs, (12 + ls.get_width(), y))
            y += 17
        y += 4
        pygame.draw.line(surf, C["border"], (10, y), (self.INSP_W - 10, y))
        return y + 6

    def _insp_bars(self, surf, agent, y) -> int:
        bar_maxh = 44
        pad      = 4

        for title, values, col in [
            ("INPUTS",  agent.last_inputs,  C["bar_in"]),
            ("OUTPUTS", agent.last_outputs, C["bar_out"]),
        ]:
            hdr = self._fsm.render(title, True, C["text_dim"])
            surf.blit(hdr, (12, y));  y += hdr.get_height() + 2

            n     = max(len(values), 1)
            bar_w = max(1, (self.INSP_W - 30 - pad * (n - 1)) // n)

            for i, v in enumerate(values):
                bx = 12 + i * (bar_w + pad)
                h  = int(abs(float(v)) * bar_maxh)
                pygame.draw.rect(surf, C["border"], (bx, y, bar_w, bar_maxh), 1)
                if h > 0:
                    intensity = 0.35 + 0.65 * abs(float(v))
                    fc = tuple(min(255, int(c * intensity)) for c in col)
                    pygame.draw.rect(surf, fc,
                                     (bx, y + bar_maxh - h, bar_w, h))
                lbl = self._fsm.render(f"{float(v):.2f}", True, C["text_dim"])
                surf.blit(lbl, (bx, y + bar_maxh + 2))

            y += bar_maxh + 18

        pygame.draw.line(surf, C["border"], (10, y), (self.INSP_W - 10, y))
        return y + 6

    def _insp_network(self, surf, agent, y):
        """Network topology with live activation colours."""
        genome = agent.genome
        acts   = agent.node_activations
        if not genome.nodes:
            return

        hdr = self._fsm.render("NETWORK", True, C["text_dim"])
        surf.blit(hdr, (12, y));  y += hdr.get_height() + 4

        avail_w = self.INSP_W - 20
        avail_h = self.WORLD_H - y - 14

        inp_ids = sorted(genome.input_ids)
        out_ids = sorted(genome.output_ids)
        hid_ids = sorted(genome.hidden_ids)

        col_groups = [g for g in [inp_ids, hid_ids, out_ids] if g]
        if not col_groups:
            return

        col_xs = [
            int(12 + avail_w * (i + 1) / (len(col_groups) + 1))
            for i in range(len(col_groups))
        ]

        node_pos: Dict[int, Tuple[int, int]] = {}
        for ci, (cx, group) in enumerate(zip(col_xs, col_groups)):
            for ri, nid in enumerate(group):
                ny2 = int(y + avail_h * (ri + 1) / (len(group) + 1))
                node_pos[nid] = (cx, ny2)

        # edges
        edge_surf = pygame.Surface((self.INSP_W, self.WORLD_H), pygame.SRCALPHA)
        for conn in genome.connections.values():
            if not conn.enabled:
                continue
            p1 = node_pos.get(conn.in_node)
            p2 = node_pos.get(conn.out_node)
            if not p1 or not p2:
                continue
            w_a   = min(abs(conn.weight) / 5.0, 1.0)
            lw    = max(1, int(w_a * 4))
            base  = C["edge_pos"] if conn.weight >= 0 else C["edge_neg"]
            alpha = int(55 + 160 * w_a)

            is_rec = p2[0] <= p1[0] + 5
            if is_rec:
                # curved recurrent arc
                pts = [
                    p1,
                    (p1[0] - 28, p1[1] - 28),
                    (p2[0] - 28, p2[1] + 28),
                    p2,
                ]
                pygame.draw.lines(edge_surf, (*base, alpha), False, pts, lw)
                # arrowhead
                dx = p2[0] - pts[-2][0]; dy = p2[1] - pts[-2][1]
                ang = math.atan2(dy, dx)
                ar  = 6
                tip = p2
                al  = (int(tip[0] - ar * math.cos(ang - 0.4)),
                       int(tip[1] - ar * math.sin(ang - 0.4)))
                ar2 = (int(tip[0] - ar * math.cos(ang + 0.4)),
                       int(tip[1] - ar * math.sin(ang + 0.4)))
                pygame.draw.polygon(edge_surf, (*base, alpha), [tip, al, ar2])
            else:
                pygame.draw.line(edge_surf, (*base, alpha), p1, p2, lw)
                dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
                ang = math.atan2(dy, dx)
                ar  = 6
                al  = (int(p2[0] - ar * math.cos(ang - 0.4)),
                       int(p2[1] - ar * math.sin(ang - 0.4)))
                ar2 = (int(p2[0] - ar * math.cos(ang + 0.4)),
                       int(p2[1] - ar * math.sin(ang + 0.4)))
                pygame.draw.polygon(edge_surf, (*base, alpha), [p2, al, ar2])
        surf.blit(edge_surf, (0, 0))

        # nodes
        NODE_R = 10
        for nid, (nx2, ny2) in node_pos.items():
            node    = genome.nodes[nid]
            act_val = float(acts.get(nid, 0.0))
            bright  = float(np.clip((act_val + 1) / 2.0, 0.0, 1.0))

            base_col = (
                C["node_input"]  if node.node_type == NODE_INPUT  else
                C["node_output"] if node.node_type == NODE_OUTPUT else
                C["node_hidden"]
            )
            lit = tuple(min(255, int(b + (255 - b) * bright * 0.75))
                        for b in base_col)

            pygame.gfxdraw.filled_circle(surf, nx2, ny2, NODE_R, lit)
            pygame.gfxdraw.aacircle(surf, nx2, ny2, NODE_R, base_col)

            id_s  = self._fsm.render(str(nid),       True, C["text_bright"])
            act_s = self._fsm.render(f"{act_val:.2f}", True, C["text_dim"])
            surf.blit(id_s,  (nx2 - id_s.get_width()  // 2, ny2 - 7))
            surf.blit(act_s, (nx2 - act_s.get_width() // 2, ny2 + NODE_R + 1))

            # activation function label
            fn_s = self._fsm.render(node.activation[:3], True, C["text_dim"])
            surf.blit(fn_s, (nx2 - fn_s.get_width() // 2, ny2 + NODE_R + 12))

    # ── bottom strip ──────────────────────────────────────────────────────────

    def _draw_bottom(self):
        bw, bh = self.WIN_W, self.BOT_H
        strip  = pygame.Surface((bw, bh))
        strip.fill(C["panel"])

        pts = self.gen_best_history
        if len(pts) >= 2:
            mn, mx = min(pts), max(pts)
            rng    = mx - mn if mx != mn else 1.0
            mg     = 12
            pw, ph = bw - mg * 2, bh - mg * 2

            poly = [(mg, bh - mg)]
            line = []
            for i, v in enumerate(pts):
                px = int(mg + i / (len(pts) - 1) * pw)
                py = int(bh - mg - (v - mn) / rng * ph)
                poly.append((px, py));  line.append((px, py))
            poly.append((mg + pw, bh - mg))
            if len(poly) >= 3:
                pygame.draw.polygon(strip, C["spark_fill"], poly)
            if len(line) >= 2:
                pygame.draw.lines(strip, C["spark_line"], False, line, 2)
            # current gen dot
            lx, ly = line[-1]
            pygame.draw.circle(strip, C["accent"], (lx, ly), 4)

        rewards = [a.reward for a in self.agents]
        best_r  = max(rewards) if rewards else 0.0
        mean_r  = float(np.mean(rewards)) if rewards else 0.0

        info = self._fmd.render(
            f"Gen {self.generation + 1}  |  Step {self.step}/{self.steps_per_gen}"
            f"  |  {len(self.agents)} agents  |  Speed {self.speed}x"
            f"  |  Best {best_r:.4f}  Mean {mean_r:.4f}",
            True, C["text"]
        )
        strip.blit(info, (12, 4))

        keys = self._fsm.render(
            "SPACE pause  R reset  +/- speed  T trails  G goals  "
            "LClick goal  RClick obstacle  Click agent pin",
            True, C["text_dim"]
        )
        strip.blit(keys, (12, 22))

        pygame.draw.rect(strip, C["border"], (0, 0, bw, bh), 1)
        self.screen.blit(strip, (0, self.WORLD_H))

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self):
        if self.paused:
            lbl = self._flg.render("  PAUSED — SPACE to resume  ", True, C["paused"])
            bg  = pygame.Surface((lbl.get_width() + 6, lbl.get_height() + 6))
            bg.fill(C["panel"])
            cx  = self.WORLD_W // 2
            self.screen.blit(bg,  (cx - bg.get_width()  // 2, 8))
            self.screen.blit(lbl, (cx - lbl.get_width() // 2, 10))
