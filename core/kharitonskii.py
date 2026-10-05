import jax.numpy as jnp


def kharitonskii_mean_var_oz(
    concentration,
    spontaneous_magnetization,
    particle_diameter_cm,
    cylinder_height_cm,
    cylinder_radius_cm,
    zeta,
):
    cos_theta_max = 1.0 / jnp.sqrt(
        1.0 + (2.0 * cylinder_radius_cm / cylinder_height_cm) ** 2
    )
    geometry_factor = 1.0 - 1.5 * cos_theta_max
    size_ratio = particle_diameter_cm / cylinder_height_cm
    scale = 4.0 * jnp.pi / 3.0

    mean = (
        -2.0
        * scale
        * concentration
        * spontaneous_magnetization
        * geometry_factor
        * zeta
    )
    variance = (
        scale**2
        * concentration
        * spontaneous_magnetization**2
        / 10.0
        * (1.0 - 3.75 * size_ratio**3)
    )
    return mean, jnp.maximum(variance, 0.0)
