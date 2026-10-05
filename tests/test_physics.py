import math

import jax.numpy as jnp
import numpy as np

from core.kharitonskii import kharitonskii_mean_var_oz
from core.llg import llg_rhs, unit
from core.models import ThermalMode
from core.thermal import KB_CGS, thermal_std


def test_unit_normalizes_each_vector() -> None:
    vectors = jnp.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]])
    normalized = np.asarray(unit(vectors))
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_llg_rhs_is_zero_for_parallel_field() -> None:
    magnetization = jnp.asarray([[0.0, 0.0, 1.0]])
    field = jnp.asarray([[0.0, 0.0, 20.0]])
    np.testing.assert_allclose(llg_rhs(magnetization, field, 1.76e7, 0.1), 0.0)


def test_thermal_std_matches_fdt_formula() -> None:
    alpha = 0.1
    temperature = 300.0
    gamma = 1.76e7
    magnetization = 480.0
    volume = math.pi / 6.0 * (10.0e-7) ** 3
    dt = 1.0e-12
    expected = math.sqrt(
        2.0 * alpha * KB_CGS * temperature
        / (gamma * magnetization * volume * dt)
    )
    actual = thermal_std(
        alpha,
        temperature,
        gamma,
        magnetization,
        volume,
        dt,
        int(ThermalMode.ON),
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)
    assert float(
        thermal_std(
            alpha,
            temperature,
            gamma,
            magnetization,
            volume,
            dt,
            int(ThermalMode.OFF),
        )
    ) == 0.0


def test_kharitonskii_oz_moments_match_documented_formula() -> None:
    c = 0.06
    saturation = 480.0
    particle_diameter = 10.0e-7
    height = 1.0e-4
    radius = 0.75e-4
    zeta = 0.4
    mean, variance = kharitonskii_mean_var_oz(
        c, saturation, particle_diameter, height, radius, zeta
    )

    cos_theta = 1.0 / math.sqrt(1.0 + (2.0 * radius / height) ** 2)
    scale = 4.0 * math.pi / 3.0
    expected_mean = -2.0 * scale * c * saturation * (1.0 - 1.5 * cos_theta) * zeta
    expected_variance = (
        scale**2
        * c
        * saturation**2
        / 10.0
        * (1.0 - 3.75 * (particle_diameter / height) ** 3)
    )
    np.testing.assert_allclose(mean, expected_mean, rtol=1.0e-6)
    np.testing.assert_allclose(variance, expected_variance, rtol=1.0e-6)
