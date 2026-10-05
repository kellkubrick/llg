from dataclasses import replace

import numpy as np
import pytest

from core import run_relaxation
from core.models import DipolarMode, SimulationParams, ThermalMode


def small_params(**changes) -> SimulationParams:
    defaults = SimulationParams(
        N=32,
        n_steps=4,
        p_up=0.75,
        cyl_R_cm=0.75e-4,
    )
    return replace(defaults, **changes)


def test_result_contains_true_initial_state() -> None:
    params = small_params(
        dipolar_mode=DipolarMode.OFF,
        thermal_mode=ThermalMode.OFF,
    )
    result = run_relaxation(params, seed=7)
    assert len(result.t) == params.n_steps + 1
    assert len(result.mz) == params.n_steps + 1
    assert result.t[0] == 0.0
    assert np.all(result.mean_rf == 0.0)
    assert result.sigma_rf == 0.0


def test_same_seed_and_parameters_are_deterministic() -> None:
    params = small_params()
    first = run_relaxation(params, seed=11)
    second = run_relaxation(params, seed=11)
    np.testing.assert_array_equal(first.mz, second.mz)
    np.testing.assert_array_equal(first.mean_rf, second.mean_rf)


@pytest.mark.parametrize(
    ("mode", "expect_sigma"),
    [
        (DipolarMode.OFF, False),
        (DipolarMode.MEAN_FIELD, False),
        (DipolarMode.GAUSSIAN, True),
    ],
)
def test_dipolar_modes_report_applied_fields(
    mode: DipolarMode, expect_sigma: bool
) -> None:
    result = run_relaxation(small_params(dipolar_mode=mode), seed=3)
    assert (result.sigma_rf > 0.0) is expect_sigma
    if mode is DipolarMode.OFF:
        assert np.all(result.mean_rf == 0.0)
    else:
        assert np.any(result.mean_rf != 0.0)


def test_thermal_switch_changes_stochastic_trajectory() -> None:
    off = run_relaxation(
        small_params(thermal_mode=ThermalMode.OFF),
        seed=9,
    )
    on = run_relaxation(
        small_params(thermal_mode=ThermalMode.ON),
        seed=9,
    )
    assert not np.array_equal(off.mz, on.mz)
