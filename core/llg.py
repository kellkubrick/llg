import jax.numpy as jnp


def unit(m):
    norm = jnp.linalg.norm(m, axis=-1, keepdims=True)
    return m / jnp.maximum(norm, 1.0e-30)


def llg_rhs(m, field, gamma, alpha):
    mxh = jnp.cross(m, field)
    mxmxh = jnp.cross(m, mxh)
    prefactor = -gamma / (1.0 + alpha * alpha)
    return prefactor * (mxh + alpha * mxmxh)
