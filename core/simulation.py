from __future__ import annotations

from functools import lru_cache, partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from .kharitonskii import kharitonskii_mean_var_oz
from .llg import llg_rhs, unit
from .models import (
    AxisMode,
    DipolarMode,
    ParameterValidationError,
    SimulationParams,
    SimulationResult,
    ThermalMode,
    ValidationMessage,
    ValidationReport,
    validate_params,
)
from .thermal import thermal_std


def init_ensemble(key, N, d_nm, p_up, axis_mode):
    particle_diameter_cm = d_nm * 1.0e-7
    volume = (jnp.pi / 6.0) * particle_diameter_cm**3
    axis_key, sign_key = random.split(key)

    def aligned_axes(_):
        return jnp.zeros((N, 3), dtype=jnp.float32).at[:, 2].set(1.0)

    def random_axes(_):
        return unit(random.normal(axis_key, (N, 3), dtype=jnp.float32))

    axes = lax.cond(
        axis_mode == int(AxisMode.ALIGNED),
        aligned_axes,
        random_axes,
        operand=None,
    )
    sign = jnp.where(
        random.uniform(sign_key, (N,), dtype=jnp.float32) < p_up,
        1.0,
        -1.0,
    )
    magnetization = unit(sign[:, None] * axes)
    return particle_diameter_cm, volume, axes, magnetization


def _relaxation_impl(
    key,
    N,
    n_steps,
    Ms,
    K,
    alpha,
    gamma,
    T,
    thermal_mode,
    d_nm,
    c,
    cyl_d_cm,
    cyl_R_cm,
    dipolar_mode,
    H0,
    dt,
    p_up,
    axis_mode,
):
    init_key, disorder_key, loop_key = random.split(key, 3)
    particle_diameter_cm, volume, axes, initial_m = init_ensemble(
        init_key, N, d_nm, p_up, axis_mode
    )
    anisotropy_field_strength = 2.0 * K / Ms

    disorder = lax.cond(
        dipolar_mode == int(DipolarMode.GAUSSIAN),
        lambda _: random.normal(disorder_key, (N,), dtype=jnp.float32),
        lambda _: jnp.zeros((N,), dtype=jnp.float32),
        operand=None,
    )
    sigma_thermal = thermal_std(
        alpha, T, gamma, Ms, volume, dt, thermal_mode
    )
    _, variance_random_field = kharitonskii_mean_var_oz(
        c,
        Ms,
        particle_diameter_cm,
        cyl_d_cm,
        cyl_R_cm,
        0.0,
    )
    sigma_random_field = jnp.where(
        dipolar_mode == int(DipolarMode.GAUSSIAN),
        jnp.sqrt(variance_random_field),
        0.0,
    )

    def applied_mean(mean):
        return jnp.where(dipolar_mode == int(DipolarMode.OFF), 0.0, mean)

    def random_field_z(mean):
        mean_applied = applied_mean(mean)
        return jnp.where(
            dipolar_mode == int(DipolarMode.GAUSSIAN),
            mean_applied + sigma_random_field * disorder,
            mean_applied,
        )

    def deterministic_field(magnetization, random_field):
        projection = jnp.sum(magnetization * axes, axis=1)
        anisotropy = (anisotropy_field_strength * projection)[:, None] * axes
        longitudinal = H0 + random_field
        return anisotropy + jnp.zeros_like(magnetization).at[:, 2].set(longitudinal)

    def step(carry, _):
        magnetization, step_key = carry
        next_key, thermal_key = random.split(step_key)
        thermal_field = lax.cond(
            thermal_mode == int(ThermalMode.ON),
            lambda _: random.normal(thermal_key, (N, 3), dtype=jnp.float32)
            * sigma_thermal,
            lambda _: jnp.zeros((N, 3), dtype=jnp.float32),
            operand=None,
        )

        zeta_predictor = jnp.mean(magnetization[:, 2])
        mean_predictor, _ = kharitonskii_mean_var_oz(
            c,
            Ms,
            particle_diameter_cm,
            cyl_d_cm,
            cyl_R_cm,
            zeta_predictor,
        )
        field_predictor = (
            deterministic_field(magnetization, random_field_z(mean_predictor))
            + thermal_field
        )
        slope_predictor = llg_rhs(magnetization, field_predictor, gamma, alpha)
        predicted_m = unit(magnetization + dt * slope_predictor)

        zeta_corrector = jnp.mean(predicted_m[:, 2])
        mean_corrector, _ = kharitonskii_mean_var_oz(
            c,
            Ms,
            particle_diameter_cm,
            cyl_d_cm,
            cyl_R_cm,
            zeta_corrector,
        )
        field_corrector = (
            deterministic_field(predicted_m, random_field_z(mean_corrector))
            + thermal_field
        )
        slope_corrector = llg_rhs(predicted_m, field_corrector, gamma, alpha)
        new_m = unit(
            magnetization + 0.5 * dt * (slope_predictor + slope_corrector)
        )

        new_mz = jnp.mean(new_m[:, 2])
        new_mean, _ = kharitonskii_mean_var_oz(
            c,
            Ms,
            particle_diameter_cm,
            cyl_d_cm,
            cyl_R_cm,
            new_mz,
        )
        return (new_m, next_key), (new_mz, applied_mean(new_mean))

    initial_mz = jnp.mean(initial_m[:, 2])
    initial_mean, _ = kharitonskii_mean_var_oz(
        c,
        Ms,
        particle_diameter_cm,
        cyl_d_cm,
        cyl_R_cm,
        initial_mz,
    )
    (_, _), (mz_steps, mean_steps) = lax.scan(
        step,
        (initial_m, loop_key),
        xs=None,
        length=n_steps,
    )

    return {
        "t": jnp.arange(n_steps + 1, dtype=jnp.float32) * dt,
        "mz": jnp.concatenate((initial_mz[None], mz_steps)),
        "mean_rf": jnp.concatenate((applied_mean(initial_mean)[None], mean_steps)),
        "sigma_rf": sigma_random_field,
        "Hk": anisotropy_field_strength,
        "c": c,
    }


relaxation_core = partial(
    jax.jit,
    static_argnames=("N", "n_steps"),
)(_relaxation_impl)


@lru_cache(maxsize=8)
def _kernel_for_shape(N: int, n_steps: int):
    @jax.jit
    def kernel(
        key,
        Ms,
        K,
        alpha,
        gamma,
        T,
        thermal_mode,
        d_nm,
        c,
        cyl_d_cm,
        cyl_R_cm,
        dipolar_mode,
        H0,
        dt,
        p_up,
        axis_mode,
    ):
        return _relaxation_impl(
            key,
            N,
            n_steps,
            Ms,
            K,
            alpha,
            gamma,
            T,
            thermal_mode,
            d_nm,
            c,
            cyl_d_cm,
            cyl_R_cm,
            dipolar_mode,
            H0,
            dt,
            p_up,
            axis_mode,
        )

    return kernel


def _dynamic_arguments(params: SimulationParams, seed: int):
    effective_p_up = (
        0.5 if AxisMode(params.axis_mode) is AxisMode.RANDOM else params.p_up
    )
    return (
        random.PRNGKey(seed),
        jnp.asarray(params.Ms, dtype=jnp.float32),
        jnp.asarray(params.K, dtype=jnp.float32),
        jnp.asarray(params.alpha, dtype=jnp.float32),
        jnp.asarray(params.gamma, dtype=jnp.float32),
        jnp.asarray(params.T, dtype=jnp.float32),
        jnp.asarray(int(params.thermal_mode), dtype=jnp.int32),
        jnp.asarray(params.d_nm, dtype=jnp.float32),
        jnp.asarray(params.c, dtype=jnp.float32),
        jnp.asarray(params.cyl_d_cm, dtype=jnp.float32),
        jnp.asarray(params.cyl_R_cm, dtype=jnp.float32),
        jnp.asarray(int(params.dipolar_mode), dtype=jnp.int32),
        jnp.asarray(params.H0, dtype=jnp.float32),
        jnp.asarray(params.dt, dtype=jnp.float32),
        jnp.asarray(effective_p_up, dtype=jnp.float32),
        jnp.asarray(int(params.axis_mode), dtype=jnp.int32),
    )


def run_relaxation(params: SimulationParams, seed: int = 1) -> SimulationResult:
    report = validate_params(params)
    if not report.is_valid:
        raise ParameterValidationError(report)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ParameterValidationError(
            ValidationReport(
                errors=(
                    ValidationMessage("seed", "seed must be a non-negative integer"),
                )
            )
        )

    total_start = perf_counter()
    arguments = _dynamic_arguments(params, seed)
    kernel = _kernel_for_shape(params.N, params.n_steps)

    compile_start = perf_counter()
    compiled = kernel.lower(*arguments).compile()
    compile_seconds = perf_counter() - compile_start

    execution_start = perf_counter()
    device_result = compiled(*arguments)
    jax.block_until_ready(device_result)
    execution_seconds = perf_counter() - execution_start

    transfer_start = perf_counter()
    host_result = jax.device_get(device_result)
    transfer_seconds = perf_counter() - transfer_start

    device = jax.devices()[0]
    return SimulationResult(
        t=np.asarray(host_result["t"]),
        mz=np.asarray(host_result["mz"]),
        mean_rf=np.asarray(host_result["mean_rf"]),
        sigma_rf=float(np.asarray(host_result["sigma_rf"])),
        Hk=float(np.asarray(host_result["Hk"])),
        c=float(np.asarray(host_result["c"])),
        seed=seed,
        backend=jax.default_backend(),
        device=str(device),
        compile_seconds=compile_seconds,
        execution_seconds=execution_seconds,
        transfer_seconds=transfer_seconds,
        total_seconds=perf_counter() - total_start,
    )
