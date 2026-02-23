"""
Типы частиц, вспомогательные функции и dataclass Particle.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from .constants import KB


# ─── Тип частицы ──────────────────────────────────────────────────────────────


class ParticleType(Enum):
    BLOCKED = "blocked"  # момент заморожен, интегрируется через LLG
    SP = "sp"  # суперпарамагнитный, флипы через прыжковый шум
    AUTO = "auto"  # определяется автоматически


# ─── Вспомогательные функции ──────────────────────────────────────────────────


def vnorm(v):
    mag = np.linalg.norm(v)
    return v / mag if mag > 0 else v.copy()


def cube(x):
    return x * x * x


def sphere_volume(r):
    return (4.0 / 3.0) * math.pi * cube(r)


def neel_time(Ku, r, T, tau0=1e-9):
    """τ_N = τ₀ · exp(Ku·V / (kB·T))"""
    if T <= 0 or Ku <= 0:
        return math.inf
    V = sphere_volume(r)
    sigma = Ku * V / (KB * T)
    if sigma > 700:
        return math.inf
    return tau0 * math.exp(sigma)


def blocking_radius(Ku, T, tau_obs=100.0, tau0=1e-9):
    """r_crit: радиус, при котором τ_N = τ_obs"""
    if Ku <= 0 or T <= 0:
        return 0.0
    ln_factor = math.log(tau_obs / tau0)
    return (3.0 * KB * T * ln_factor / (4.0 * math.pi * Ku)) ** (1.0 / 3.0)


# ─── Класс частицы ────────────────────────────────────────────────────────────


@dataclass
class Particle:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    r_h: float = 0.0
    msat: float = 0.0
    eta: float = 0.0
    ptype: ParticleType = ParticleType.AUTO

    m: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    u_anis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    u2_anis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    c1_anis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    c2_anis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    c3_anis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    biasfield: np.ndarray = field(default_factory=lambda: np.zeros(3))
    demagnetising_field: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tempfield: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dmdt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    previousm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tempm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk3: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk4: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk5: np.ndarray = field(default_factory=lambda: np.zeros(3))
    temp_prefactor: float = 0.0
    randomvprefact: float = 0.0
    switch_time: float = 0.0
    fixed: bool = False

    def volume(self):
        return sphere_volume(self.r)

    def dist(self, x, y, z):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2)

    def calc_temp_prefactor(self, alpha, temp):
        from .constants import GAMMA0

        if self.msat > 0 and temp > 0:
            self.temp_prefactor = math.sqrt(
                (2.0 * KB * alpha * temp) / (GAMMA0 * self.msat * self.volume())
            )

    def calc_randomv_prefact(self, temp):
        if self.eta > 0 and temp > 0:
            vol_h = sphere_volume(self.r_h)
            self.randomvprefact = math.sqrt(
                (2.0 * KB * temp) / (6.0 * self.eta * vol_h)
            )

    def neel_relaxation_time(self, Ku, T, tau0=1e-9):
        return neel_time(Ku, self.r, T, tau0)

    def sigma(self, Ku, T):
        if T <= 0 or Ku <= 0:
            return math.inf
        return Ku * self.volume() / (KB * T)

    def __repr__(self):
        return (
            f"Particle(r={self.r * 1e9:.1f}nm, "
            f"type={self.ptype.value}, m={np.round(self.m, 3)})"
        )
