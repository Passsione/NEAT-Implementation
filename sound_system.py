"""
sound_system.py  —  Continuous Wave Sound Simulation for NEAT Agents
======================================================================
Sound is described by 4 continuous wave parameters, not phoneme labels.
The network learns to produce and perceive arbitrary wave combinations;
phoneme-like structure can emerge without being pre-specified.

Encoding
--------
  SOUND_OUTPUTS = 4   (appended after movement outputs)
    [0] frequency   0→1  maps to  80–4000 Hz  (log scale)
    [1] amplitude   0→1  emission gate + loudness
    [2] timbre      0→1  0 = pure sine/voiced, 1 = noise/unvoiced
    [3] formant     0→1  maps to formant-ratio shaping (vowel colour)

  SOUND_INPUTS  = 4   (appended after position/goal/obstacle inputs)
    [0] heard_amplitude   RMS of superposed wave at listener
    [1] heard_frequency   amplitude-weighted centroid frequency
    [2] heard_timbre      weighted average timbre of heard events
    [3] heard_formant     weighted average formant value of heard events

  Total cfg changes:
    n_inputs  = 6 + 4  = 10
    n_outputs = 2 + 4  =  6

Wave physics
------------
  A(r, t) = amp · exp(-r / DECAY_RADIUS) · cos(2π·f·t − r/WAVE_SPEED)
  Superposition across all active events — interference is exact.
  Each SoundEvent carries its own frequency so the centroid sensor
  correctly tracks which pitches are arriving at the listener.

Dependencies: numpy only (scipy optional for audio synthesis)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

WAVE_SPEED    = 200.0    # world-units / second
DECAY_RADIUS  = 100.0    # world-units — 1/e falloff distance
MAX_EVENTS    = 48       # hard cap on simultaneous live events
EMIT_COOLDOWN = 0.06     # seconds minimum between emissions per agent
AMP_THRESHOLD = 0.12     # below this amplitude → no emission

FREQ_MIN = 80.0          # Hz — lowest emittable frequency
FREQ_MAX = 4000.0        # Hz — highest emittable frequency

ENABLE_AUDIO  = False    # set True for pygame.mixer synthesis

# I/O dimensions — import these into run.py
SOUND_OUTPUTS = 4
SOUND_INPUTS  = 8


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_freq(x: float) -> float:
    """Map [0, 1] → [FREQ_MIN, FREQ_MAX] on a log scale."""
    return float(FREQ_MIN * (FREQ_MAX / FREQ_MIN) ** np.clip(x, 0.0, 1.0))


def _safe(x) -> float:
    v = float(x)
    return 0.0 if (math.isnan(v) or math.isinf(v)) else float(np.clip(v, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  SOUND EVENT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SoundEvent:
    x:         float    # world origin
    y:         float
    frequency: float    # Hz
    amplitude: float    # 0–1
    timbre:    float    # 0 = voiced/tonal, 1 = noisy/unvoiced
    formant:   float    # 0–1 continuous vowel-space parameter
    emitter_id: int
    time_born: float
    ttl:       float = 0.8   # seconds


# ══════════════════════════════════════════════════════════════════════════════
#  SOUND EMITTER
# ══════════════════════════════════════════════════════════════════════════════

class VocalResonator:
    """
    Replaces SoundEmitter. Acts as a continuous analog oscillator.
    """
    def __init__(self, emitter_id: int, slew_rate: float = 0.15):
        self.emitter_id = emitter_id
        self.slew_rate = slew_rate
        
        # Physical State
        self.curr_freq = FREQ_MIN
        self.curr_amp = 0.0
        self.phase_acc = 0.0  # The "Analog" magic: tracks position in the wave
        
        # Position History (for speed-of-sound delays)
        self.history = [] # Stores (time, x, y, freq, amp, timbre, formant)
        self.max_history_seconds = 2.0

    def update(self, nn_outputs: List[float], x: float, y: float, sim_time: float, dt: float):
        # 1. Decode NN outputs (0->1) to physical ranges
        target_freq = math.exp(math.log(FREQ_MIN) + nn_outputs[0] * (math.log(FREQ_MAX) - math.log(FREQ_MIN)))
        target_amp = nn_outputs[1]
        
        # 2. Apply Slew (Analog Smoothing)
        # Prevents "digital clicking" by physically sliding to new values
        self.curr_freq += (target_freq - self.curr_freq) * self.slew_rate
        self.curr_amp += (target_amp - self.curr_amp) * self.slew_rate

        # 3. Update Phase Accumulator
        # This ensures the wave is continuous regardless of frequency changes
        self.phase_acc += 2.0 * math.pi * self.curr_freq * dt
        self.phase_acc %= 2.0 * math.pi 

        # 4. Record state into history for the SoundField to "propagate"
        state = {
            't': sim_time,
            'x': x, 'y': y,
            'freq': self.curr_freq,
            'amp': self.curr_amp,
            'phase': self.phase_acc,
            'timbre': nn_outputs[2],
            'formant': nn_outputs[3]
        }
        self.history.append(state)
        
        # Cleanup old history
        # if len(self.history) > (self.max_history_seconds / dt):
        #     self.history.pop(0)



    def update(self, nn_outputs: List[float], x: float, y: float, sim_time: float, dt: float):
        # 1. Decode NN outputs (0->1) to physical ranges
        target_freq = math.exp(math.log(FREQ_MIN) + nn_outputs[0] * (math.log(FREQ_MAX) - math.log(FREQ_MIN)))
        target_amp = nn_outputs[1]
        
        # 2. Apply Slew (Analog Smoothing)
        # Prevents "digital clicking" by physically sliding to new values
        self.curr_freq += (target_freq - self.curr_freq) * self.slew_rate
        self.curr_amp += (target_amp - self.curr_amp) * self.slew_rate

        # 3. Update Phase Accumulator
        # This ensures the wave is continuous regardless of frequency changes
        self.phase_acc += 2.0 * math.pi * self.curr_freq * dt
        self.phase_acc %= 2.0 * math.pi 

        # 4. Record state into history for the SoundField to "propagate"
        state = {
            't': sim_time,
            'x': x, 'y': y,
            'freq': self.curr_freq,
            'amp': self.curr_amp,
            'phase': self.phase_acc,
            'timbre': nn_outputs[2],
            'formant': nn_outputs[3]
        }
        self.history.append(state)
        
        # Cleanup old history
        if len(self.history) > (self.max_history_seconds / dt):
            self.history.pop(0)


# ══════════════════════════════════════════════════════════════════════════════
#  SOUND FIELD
# ══════════════════════════════════════════════════════════════════════════════

class SoundField:
    """
    Shared world object. Maintains active SoundEvents and evaluates
    the superposed wave analytically at any listener position.

    Physics per event at distance r, time t:
        phase    = 2π·f·t − r / WAVE_SPEED
        atten    = amp · exp(-r / DECAY_RADIUS)
        envelope = 1 − (age/ttl)²
        contrib  = atten · cos(phase) · envelope

    Superposition is simple addition — interference (constructive and
    destructive) emerges naturally from phase differences.
    """

    def __init__(self):
        self.events:   List[SoundEvent] = []
        self.sim_time: float = 0.0
        self.noise_floor = 0.05 
        self.noise_seed = np.random.default_rng()

    

    def reset(self):
        self.events = []
        self.sim_time = 0.0

    def emit(self, event: SoundEvent):
        self.events.append(event)
        if len(self.events) > MAX_EVENTS:
            # drop quietest to stay under cap
            self.events.sort(key=lambda e: e.amplitude)
            self.events.pop(0)

    def get_background_interference(self, num_bands: int) -> List[float]:
        """Simulates environmental 'Pink Noise' interference."""
        # Generates small random fluctuations per frequency band
        return [self.noise_seed.uniform(0, self.noise_floor) for _ in range(num_bands)]
    
    def step(self, dt: float):
        self.sim_time += dt
        t = self.sim_time
        self.events = [e for e in self.events if (t - e.time_born) < e.ttl]

    def _attenuated_strength(self, ev: SoundEvent, lx: float, ly: float) -> float:
        """Scalar strength of one event at a listener — used for sensor weighting."""
        age = self.sim_time - ev.time_born
        if age <= 0 or age >= ev.ttl:
            return 0.0
        r = math.hypot(lx - ev.x, ly - ev.y)
        r = max(r, 1e-3)
        atten    = ev.amplitude / (1.0 + (r / DECAY_RADIUS)**2)
        envelope = 1.0 - (age / ev.ttl) ** 2
        return float(atten * envelope)
    
    def sample_cochlea(
        self,
        lx: float, ly: float,
        num_bands: int = 4,
        exclude_id: Optional[int] = None,
    ) -> List[float]:
        """
        Mechanical Fourier Transform: Sorts incoming waves into logarithmic 
        frequency bands and calculates the instantaneous wave interference per band.
        """
        # Initialize frequency bands
        bands = [0.0] * num_bands
        log_min = math.log(FREQ_MIN)
        log_max = math.log(FREQ_MAX)
        log_range = log_max - log_min

        t = self.sim_time
        for ev in self.events:
            if exclude_id is not None and ev.emitter_id == exclude_id:
                continue
            
            # Attenuated physical strength
            s = self._attenuated_strength(ev, lx, ly)
            if s < 1e-5:
                continue
            
            # Which band does this frequency excite?
            freq_norm = (math.log(ev.frequency) - log_min) / log_range
            band_idx = int(np.clip(freq_norm * num_bands, 0, num_bands - 1))
            
            # Calculate true phase interference at the ear's exact coordinate
            r = max(math.hypot(lx - ev.x, ly - ev.y), 1e-3)
            phase = 2.0 * math.pi * ev.frequency * t - r / WAVE_SPEED #(ev.frequency * (t_array - dist / WAVE_SPEED)) % 1.0
            instantaneous = s * math.cos(phase)
            
            # Add to the physical excitation of the cochlear hair cells
            bands[band_idx] += instantaneous
        
        noise = self.get_background_interference(num_bands)
        return [abs(b) + n for b, n in zip(bands, noise)]

    def sample(self, lx: float, ly: float) -> float:
        """
        Instantaneous superposed amplitude at (lx, ly).
        Includes wave phase — oscillates over time, giving true interference.
        """
        total = 0.0
        t = self.sim_time
        for ev in self.events:
            s = self._attenuated_strength(ev, lx, ly)
            if s < 1e-5:
                continue
            r     = max(math.hypot(lx - ev.x, ly - ev.y), 1e-3)
            phase = 2.0 * math.pi * ev.frequency * t - r / WAVE_SPEED #(ev.frequency * (t_array - dist / WAVE_SPEED)) % 1.0
            total += s * math.cos(phase)
        return float(np.clip(total, -2.0, 2.0))

    def sample_properties(
        self,
        lx: float, ly: float,
        exclude_id: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Return the 4 perceptual properties of sound arriving at (lx, ly):
            (amplitude_rms, freq_centroid_norm, timbre, formant)
        All in [0, 1].
        """
        total_strength = 0.0
        rms_accum      = 0.0
        freq_accum     = 0.0
        timbre_accum   = 0.0
        formant_accum  = 0.0

        t = self.sim_time
        for ev in self.events:
            if exclude_id is not None and ev.emitter_id == exclude_id:
                continue
            s = self._attenuated_strength(ev, lx, ly)
            if s < 1e-5:
                continue
            r     = max(math.hypot(lx - ev.x, ly - ev.y), 1e-3)
            phase = 2.0 * math.pi * ev.frequency * t - r / WAVE_SPEED
            instantaneous = s * math.cos(phase)

            rms_accum      += instantaneous ** 2
            freq_accum     += s * ev.frequency
            timbre_accum   += s * ev.timbre
            formant_accum  += s * ev.formant
            total_strength += s

        if total_strength < 1e-9:
            return 0.0, 0.0, 0.0, 0.0

        n = max(len(self.events), 1)
        heard_amp     = float(np.clip(math.sqrt(rms_accum / n), 0.0, 1.0))
        heard_freq    = float(np.clip(
            math.log(max(freq_accum / total_strength, FREQ_MIN) / FREQ_MIN)
            / math.log(FREQ_MAX / FREQ_MIN),
            0.0, 1.0,
        ))
        heard_timbre  = float(np.clip(timbre_accum  / total_strength, 0.0, 1.0))
        heard_formant = float(np.clip(formant_accum / total_strength, 0.0, 1.0))

        return heard_amp, heard_freq, heard_timbre, heard_formant


# ══════════════════════════════════════════════════════════════════════════════
#  SOUND SENSOR
# ══════════════════════════════════════════════════════════════════════════════

class BinauralCochleaSensor:
    """
    Dual-ear filter-bank sensor.
    Returns 8 inputs: [Left_B1, Left_B2, Left_B3, Left_B4, Right_B1, Right_B2, Right_B3, Right_B4]
    """
    def __init__(self, num_bands: int = 4, smoothing: float = 0.9):
        self._alpha = smoothing
        self.num_bands = num_bands
        self._state = np.zeros(num_bands * 2, dtype=np.float64)

    def observe(
        self,
        field: SoundField,
        lx_left: float, ly_left: float,
        lx_right: float, ly_right: float,
        exclude_emitter_id: Optional[int] = None,
    ) -> List[float]:
        
        # Sample the cochlea at both physical ear locations
        left_ear = field.sample_cochlea(lx_left, ly_left, self.num_bands, exclude_emitter_id)
        right_ear = field.sample_cochlea(lx_right, ly_right, self.num_bands, exclude_emitter_id)
        
        raw = np.array(left_ear + right_ear, dtype=np.float64)
        
        # Exponential smoothing (acts as the temporal envelope tracking of the ear)
        self._state = self._alpha * raw + (1.0 - self._alpha) * self._state
        return [float(v) for v in self._state]

# ══════════════════════════════════════════════════════════════════════════════
#  SOUND AGENT ENV  —  drop-in replacement for AgentEnv
# ══════════════════════════════════════════════════════════════════════════════

class SoundAgentEnv:
    """
    Wraps the original AgentEnv behaviour with sound I/O.

    n_inputs  = 6 + SOUND_INPUTS  = 10
    n_outputs = 2 + SOUND_OUTPUTS =  6

    Update run.py:
        cfg.n_inputs  = 10
        cfg.n_outputs =  6
    """

    WORLD_W = 800
    WORLD_H = 600

    def __init__(
        self,
        rng:       np.random.Generator,
        goals:     list,
        obstacles: list,
        field:     SoundField,
        genome_id: int,
        hear_self: bool = True,
    ):
        from collections import deque
        self.rng       = rng
        self.goals     = goals
        self.obstacles = obstacles
        self.field     = field
        self.hear_self = hear_self

        self.emitter = VocalResonator(emitter_id=genome_id)
        self.sensor = BinauralCochleaSensor(num_bands=4)
        self.heading = 0.0  # Angle the agent is facing
        self.head_radius = 8.0 # Distance from center to ear

        self.x = self.y = 400.0
        self.trail_x = deque(maxlen=80)
        self.trail_y = deque(maxlen=80)
        self.reward  = 0.0
        self.done    = False
        self._t      = 0.0
        self._dt     = 1.0 / 60.0
        self._last_heard: List[float] = [0.0] * SOUND_INPUTS

    def reset(self) -> List[float]:
        self.x = float(self.rng.uniform(80, self.WORLD_W - 80))
        self.y = float(self.rng.uniform(80, self.WORLD_H - 80))
        self.trail_x.clear(); self.trail_x.append(self.x)
        self.trail_y.clear(); self.trail_y.append(self.y)
        self.reward = 0.0
        self.done   = False
        self._t     = 0.0
        self._last_heard = [0.0] * SOUND_INPUTS
        return self._obs()

    def step(self, action: List[float]) -> Tuple[List[float], float, bool]:
        import pygame
        dt = self._dt
        self._t += dt

        # movement — first 2 outputs
        speed = 4.0
        dx = (action[0] - 0.5) * 2.0 * speed
        dy = (action[1] - 0.5) * 2.0 * speed if len(action) > 1 else 0.0

        nx = float(np.clip(self.x + dx, 0, self.WORLD_W))
        ny = float(np.clip(self.y + dy, 0, self.WORLD_H))
        pt = pygame.Rect(int(nx) - 3, int(ny) - 3, 6, 6)
        if not any(pt.colliderect(o) for o in self.obstacles):
            self.x, self.y = nx, ny
        self.trail_x.append(self.x)
        self.trail_y.append(self.y)

        # sound — outputs 2,3,4,5
        # if len(action) >= 6:
        ev = self.emitter.update(action[2:], self.x, self.y, self._t, dt)
        if ev is not None:
            self.field.emit(ev)
            if ENABLE_AUDIO:
                _play(ev)

        # 3. Calculate Physical Ear Positions
        if dx != 0 or dy != 0:
            self.heading = math.atan2(dy, dx)

        self.left_ear = (
            self.x + math.cos(self.heading - math.pi/2) * self.head_radius,
            self.y + math.sin(self.heading - math.pi/2) * self.head_radius
        )
        self.right_ear = (
            self.x + math.cos(self.heading + math.pi/2) * self.head_radius,
            self.y + math.sin(self.heading + math.pi/2) * self.head_radius
        )

        # 4. Perceive through Cochlea
        excl = None if self.hear_self else self.emitter.emitter_id
        self._last_heard = self.sensor.observe(
            self.field, 
            *self.left_ear, *self.right_ear, 
            excl
        )

        # reward
        self.reward = self._base_reward()
        self.reward += float(np.nan_to_num(self._last_heard[0])) * 0.05

        return self._obs(), self.reward, self.done

    def _base_reward(self) -> float:
        if self.goals:
            dists = [math.hypot(self.x - gx, self.y - gy) for gx, gy in self.goals]
            r = 1.0 / (1.0 + min(dists))
        else:
            start_x = list(self.trail_x)[0]
            start_y = list(self.trail_y)[0]
            r = math.hypot(self.x - start_x, self.y - start_y) / self.WORLD_W
        if self.obstacles:
            dists = [math.hypot(self.x - o.centerx, self.y - o.centery)
                     for o in self.obstacles]
            r = max(0.0, r - 1.0 / (1.0 + min(dists)))
        return float(r)

    def _obs(self) -> List[float]:
        nx = self.x / self.WORLD_W
        ny = self.y / self.WORLD_H
        base = [nx, ny, 0.0, 0.0, 0.0, 0.0]
        if self.goals:
            dists = [(math.hypot(self.x - gx, self.y - gy), gx, gy)
                     for gx, gy in self.goals]
            _, gx, gy = min(dists)
            base[2] = (gx - self.x) / self.WORLD_W
            base[3] = (gy - self.y) / self.WORLD_H
        if self.obstacles:
            dists = [(math.hypot(self.x - o.centerx, self.y - o.centery),
                      o.centerx, o.centery)
                     for o in self.obstacles]
            _, ox, oy = min(dists)
            base[4] = (ox - self.x) / self.WORLD_W
            base[5] = (oy - self.y) / self.WORLD_H
        
        return base + self._last_heard


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISER OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_sound_field(
    surface,
    field:    SoundField,
    scale_x:  float = 1.0,
    scale_y:  float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    max_rings: int  = 4,
    alpha:    int   = 40,
):
    """
    Draw expanding wave rings for each active event.
    Colour encodes timbre: teal = tonal/voiced, orange = noisy/unvoiced.
    Ring spacing encodes frequency: high-freq = tightly packed rings.
    Call inside _draw_world(), after trails, before agents.
    """
    import pygame

    t    = field.sim_time
    surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

    for ev in field.events:
        age = t - ev.time_born
        if age <= 0 or age > ev.ttl:
            continue

        r_lead = age * WAVE_SPEED
        fade   = (1.0 - age / ev.ttl) * ev.amplitude
        sx     = int(ev.x * scale_x + offset_x)
        sy     = int(ev.y * scale_y + offset_y)

        # colour: lerp teal → orange by timbre
        teal   = (80,  200, 220)
        orange = (255, 160,  60)
        col    = tuple(int(teal[i] + ev.timbre * (orange[i] - teal[i])) for i in range(3))

        # ring spacing = half-wavelength in screen pixels
        wavelength_world = WAVE_SPEED / max(ev.frequency, 1.0)
        ring_gap         = wavelength_world * min(scale_x, scale_y) * 0.5

        for i in range(max_rings):
            r = r_lead - i * max(ring_gap, 1.0)
            if r <= 0:
                break
            r_screen   = int(r * min(scale_x, scale_y))
            ring_alpha = int(alpha * fade * (1.0 - i / max_rings))
            if r_screen < 1 or ring_alpha < 2:
                continue
            pygame.draw.circle(
                surf, (*col, ring_alpha), (sx, sy), r_screen,
                width=max(1, int(2 * fade)),
            )

    surface.blit(surf, (0, 0))


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL AUDIO SYNTHESIS  (scipy)
# ══════════════════════════════════════════════════════════════════════════════

def _play(ev: SoundEvent, sample_rate: int = 22050, duration: float = 0.07):
    """Synthesise and play one SoundEvent through pygame.mixer."""
    try:
        import pygame
        from scipy import signal as sp
        n   = int(duration * sample_rate)
        t   = np.linspace(0, duration, n, endpoint=False)
        saw = sp.sawtooth(2 * np.pi * ev.frequency * t)
        nse = np.random.default_rng().standard_normal(n)
        src = (1.0 - ev.timbre) * saw + ev.timbre * nse
        f_c = 300.0 + ev.formant * 2400.0
        nyq = sample_rate / 2.0
        lo  = max(20.0, f_c - 200) / nyq
        hi  = min(0.999, (f_c + 200) / nyq)
        if lo < hi:
            b, a = sp.butter(2, [lo, hi], btype="band")
            src  = sp.filtfilt(b, a, src)
        env = np.ones(n)
        atk = int(n * 0.05); env[:atk]  = np.linspace(0, 1, atk)
        dec = int(n * 0.15); env[-dec:] = np.linspace(1, 0, dec)
        src *= env * ev.amplitude
        peak = np.abs(src).max()
        if peak > 1e-6:
            src /= peak
        pygame.sndarray.make_sound((src * 32767).astype(np.int16)).play()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"SOUND_OUTPUTS : {SOUND_OUTPUTS}   (append to movement outputs)")
    print(f"SOUND_INPUTS  : {SOUND_INPUTS}    (append to position inputs)")
    print(f"cfg.n_outputs = 2 + {SOUND_OUTPUTS} = {2 + SOUND_OUTPUTS}")
    print(f"cfg.n_inputs  = 6 + {SOUND_INPUTS}  = {6 + SOUND_INPUTS}")

    field  = SoundField()
    em     = VocalResonator(emitter_id=1)
    sensor = BinauralCochleaSensor()
    rng    = np.random.default_rng(0)
    dt     = 1 / 60

    for step in range(30):
        fake    = list(rng.uniform(0, 1, 6))
        fake[3] = 0.9   # loud amplitude
        ev = em.update(fake[2:], x=400.0, y=300.0, sim_time=step * dt, dt=dt)
        if ev is not None:
            field.emit(ev)
            print(f"  step {step:2d}: emit  f={ev.frequency:6.1f}Hz "
                  f"amp={ev.amplitude:.2f} timbre={ev.timbre:.2f} "
                  f"formant={ev.formant:.2f}")
        field.step(dt)
        obs = sensor.observe(field, 550.0, 300.0, exclude_emitter_id=1)
        if obs[0] > 0.01:
            print(f"          heard amp={obs[0]:.3f} freq={obs[1]:.3f} "
                  f"timbre={obs[2]:.3f} formant={obs[3]:.3f}")

    print("\nSelf-test passed.")
