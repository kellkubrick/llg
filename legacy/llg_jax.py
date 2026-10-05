import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial

kB = 1.380649e-16  # erg/K


def unit(m):
    norm = jnp.linalg.norm(m, axis=-1, keepdims=True)
    return m / jnp.maximum(norm, 1e-30)


# ============================================================
# Kharitonskii Table 1, Oz projection
# ============================================================

def kharitonskii_mean_var_Oz(c, Is, d0_cm, cyl_d_cm, cyl_R_cm, zeta):
    cos_th = 1.0 / jnp.sqrt(1.0 + (2.0 * cyl_R_cm / cyl_d_cm) ** 2)

    g = 1.0 - 1.5 * cos_th
    q = d0_cm / cyl_d_cm
    a = 4.0 * jnp.pi / 3.0

    mean = -2.0 * a * c * Is * g * zeta

    var = (a * a * c * Is * Is / 10.0) * (1.0 - (15.0 / 4.0) * q**3)

    return mean, jnp.maximum(var, 0.0)


# ============================================================
# LLG
# ============================================================

def llg_rhs(m, H, gamma, alpha):
    mxH = jnp.cross(m, H)
    mxmxH = jnp.cross(m, mxH)

    pref = -gamma / (1.0 + alpha * alpha)

    return pref * (
        mxH
        + alpha * mxmxH
    )


def thermal_std(alpha, T, gamma, Ms, V, dt, thermal_mode):
    amp = jnp.sqrt(
        jnp.maximum(
            0.0,
            (2.0 * alpha * kB * T)
            / (gamma * Ms * V * dt)
        )
    )

    return jnp.where(
        (thermal_mode == 1) & (T > 0.0),
        amp,
        0.0,
    )


# ============================================================
# INITIAL STATE
# ============================================================

@partial(jax.jit, static_argnames=("N", "axis_mode"))
def init_ensemble(key, N, d_nm, p_up, axis_mode):
    d_cm = d_nm * 1e-7
    V = (jnp.pi / 6.0) * d_cm**3

    k_axis, k_sign = random.split(key)

    # easy axes
    e = lax.cond(
        axis_mode == 0,

        # all easy axes || Oz
        lambda _:
            jnp.zeros((N, 3), dtype=jnp.float32).at[:, 2].set(1.0),

        # random easy axes
        lambda _:
            unit(random.normal(k_axis, (N, 3), dtype=jnp.float32)),

        operand=None,
    )

    sign = jnp.where(random.uniform(k_sign, (N,)) < p_up, 1.0, -1.0)

    m0 = unit(sign[:, None] * e)

    return d_cm, V, e, m0


# ============================================================
# RELAXATION CORE
#
# dipolar_mode:
#   0 -> no dipolar field
#   1 -> Kharitonskii mean field
#   2 -> Gaussian closure: mean + sigma*xi
#
# For strict Kharitonskii-Oz interpretation:
#   axis_mode = 0
# ============================================================

@partial(jax.jit, static_argnames=("N", "n_steps", "dipolar_mode", "thermal_mode", "axis_mode"))
def relaxation_core(key,
    # computationa
    N,
    n_steps,

    # material
    Ms,
    K,
    alpha,
    gamma,

    # temperature
    T,
    thermal_mode,

    # particle
    d_nm,

    # physical particle concentration
    c,

    # sample geometry
    cyl_d_cm,
    cyl_R_cm,

    # interaction
    dipolar_mode,

    # external field
    H0,

    # integration
    dt,

    # initial state
    p_up,

    # easy-axis configuration
    axis_mode,
):
    k_init, k_xi, k_loop = random.split(key, 3)

    d_cm, V, e, m0 = init_ensemble(
        k_init,
        N,
        d_nm,
        p_up,
        axis_mode,
    )

    Hk = 2.0 * K / Ms

    # quenched random variable:
    # fixed for each particle through whole trajectory
    xi = random.normal(
        k_xi,
        (N,),
        dtype=jnp.float32,
    )

    sigma_th = thermal_std(
        alpha,
        T,
        gamma,
        Ms,
        V,
        dt,
        thermal_mode,
    )

    # Kharitonskii variance does not depend on zeta
    _, var_rf = kharitonskii_mean_var_Oz(
        c,
        Ms,
        d_cm,
        cyl_d_cm,
        cyl_R_cm,
        0.0,
    )

    sigma_rf = jnp.sqrt(var_rf)

    def make_Hrf(mean_rf):
        return lax.cond(
            dipolar_mode == 0,
            lambda _: jnp.zeros((N,), dtype=jnp.float32),
            lambda _: lax.cond(
                dipolar_mode == 1,
                lambda __:
                    jnp.full(
                        (N,),
                        mean_rf,
                        dtype=jnp.float32,
                    ),
                lambda __:
                    mean_rf + sigma_rf * xi,
                operand=None,
            ),
            operand=None,
        )

    def H_det(m_now, Hrf_z):
        mdote = jnp.sum(
            m_now * e,
            axis=1,
        )

        Hani = (
            Hk * mdote
        )[:, None] * e

        Hext = (
            jnp.zeros_like(m_now)
            .at[:, 2]
            .set(H0)
        )

        Hrf = (
            jnp.zeros_like(m_now)
            .at[:, 2]
            .set(Hrf_z)
        )

        return Hani + Hext + Hrf

    # --------------------------------------------------------
    # one stochastic Heun step
    # --------------------------------------------------------

    def step_fn(carry, _):
        m, key = carry

        key, k_th = random.split(key)

        # same thermal realization for predictor/corrector
        H_th = (
            random.normal(
                k_th,
                (N, 3),
                dtype=jnp.float32,
            )
            * sigma_th
        )

        # ---------- predictor ----------

        zeta1 = jnp.mean(m[:, 2])

        mean_rf1, _ = kharitonskii_mean_var_Oz(
            c,
            Ms,
            d_cm,
            cyl_d_cm,
            cyl_R_cm,
            zeta1,
        )

        Hrf1 = make_Hrf(mean_rf1)

        H1 = H_det(m, Hrf1) + H_th

        k1 = llg_rhs(
            m,
            H1,
            gamma,
            alpha,
        )

        mp = unit(
            m + dt * k1
        )

        # ---------- corrector ----------
        # mean dipolar field recomputed self-consistently

        zeta2 = jnp.mean(mp[:, 2])

        mean_rf2, _ = kharitonskii_mean_var_Oz(
            c,
            Ms,
            d_cm,
            cyl_d_cm,
            cyl_R_cm,
            zeta2,
        )

        Hrf2 = make_Hrf(mean_rf2)

        H2 = H_det(mp, Hrf2) + H_th

        k2 = llg_rhs(
            mp,
            H2,
            gamma,
            alpha,
        )

        m_new = unit(
            m
            + 0.5 * dt * (k1 + k2)
        )

        # observables at the new time
        mz_new = jnp.mean(
            m_new[:, 2]
        )

        mean_rf_new, _ = kharitonskii_mean_var_Oz(
            c,
            Ms,
            d_cm,
            cyl_d_cm,
            cyl_R_cm,
            mz_new,
        )

        return (
            m_new,
            key,
        ), (
            mz_new,
            mean_rf_new,
        )

    # initial observables
    mz0 = jnp.mean(m0[:, 2])

    mean_rf0, _ = kharitonskii_mean_var_Oz(
        c,
        Ms,
        d_cm,
        cyl_d_cm,
        cyl_R_cm,
        mz0,
    )

    (_, _), (mz_steps, mean_rf_steps) = lax.scan(
        step_fn,
        (m0, k_loop),
        xs=None,
        length=n_steps,
    )

    # Include t = 0 explicitly
    mz_t = jnp.concatenate([
        mz0[None],
        mz_steps,
    ])

    mean_rf_t = jnp.concatenate([
        mean_rf0[None],
        mean_rf_steps,
    ])

    t = (
        jnp.arange(
            n_steps + 1,
            dtype=jnp.float32,
        )
        * dt
    )

    return {
        "t": t,
        "mz": mz_t,
        "mean_rf": mean_rf_t,
        "sigma_rf": sigma_rf,
        "Hk": Hk,
        "c": c,
    }
