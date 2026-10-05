import jax.numpy as jnp

from .models import ThermalMode

KB_CGS = 1.380649e-16  # erg/K


def thermal_std(alpha, temperature, gamma, saturation_magnetization, volume, dt, mode):
    amplitude = jnp.sqrt(
        jnp.maximum(
            0.0,
            (2.0 * alpha * KB_CGS * temperature)
            / (gamma * saturation_magnetization * volume * dt),
        )
    )
    return jnp.where(
        (mode == int(ThermalMode.ON)) & (temperature > 0.0),
        amplitude,
        0.0,
    )
