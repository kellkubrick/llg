"""
Распределение по размерам частиц.
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class SizeDistribution:
    """
    Логнормальное распределение радиусов:
        P(r) ∝ (1/r) · exp(-(ln r - ln r_median)² / (2·ln²σ_geo))

    Параметры:
        r_median  — медианный радиус, м
        sigma_geo — геометрическое стандартное отклонение (1.1–1.5 типично)
    """

    r_median: float
    sigma_geo: float = 1.3

    def sample(self, n, rng):
        mu = math.log(self.r_median)
        sig = math.log(self.sigma_geo)
        return np.exp(rng.normal(mu, sig, n))

    def pdf(self, r_values):
        mu = math.log(self.r_median)
        sig = math.log(self.sigma_geo)
        ln_r = np.log(r_values)
        return np.exp(-0.5 * ((ln_r - mu) / sig) ** 2) / (
            r_values * sig * math.sqrt(2 * math.pi)
        )

    @property
    def mean_r(self):
        sig = math.log(self.sigma_geo)
        return self.r_median * math.exp(sig**2 / 2.0)

    @property
    def mode_r(self):
        sig = math.log(self.sigma_geo)
        return self.r_median * math.exp(-(sig**2))

    def __repr__(self):
        return (
            f"SizeDistribution(r_med={self.r_median * 1e9:.1f}nm, "
            f"σ_geo={self.sigma_geo:.2f}, <r>={self.mean_r * 1e9:.1f}nm)"
        )
