from dataclasses import replace

import pytest

from core.models import (
    AxisMode,
    DipolarMode,
    SimulationParams,
    validate_params,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("N", 0),
        ("n_steps", 0),
        ("dt", 0.0),
        ("Ms", 0.0),
        ("K", -1.0),
        ("alpha", -0.1),
        ("gamma", 0.0),
        ("T", -1.0),
        ("d_nm", 0.0),
        ("c", 1.1),
        ("cyl_d_cm", 0.0),
        ("cyl_R_cm", 0.0),
        ("p_up", -0.1),
        ("axis_mode", 3),
        ("dipolar_mode", 3),
        ("thermal_mode", 3),
    ],
)
def test_invalid_parameter_is_rejected(field: str, value: object) -> None:
    params = replace(SimulationParams(), **{field: value})
    report = validate_params(params)
    assert not report.is_valid
    assert field in {item.field for item in report.errors}


def test_random_axes_with_dipolar_closure_is_warning_not_error() -> None:
    params = replace(
        SimulationParams(),
        axis_mode=AxisMode.RANDOM,
        dipolar_mode=DipolarMode.GAUSSIAN,
    )
    report = validate_params(params)
    assert report.is_valid
    assert any("phenomenological" in item.message for item in report.warnings)


def test_geometry_domain_warning() -> None:
    params = replace(SimulationParams(), cyl_d_cm=2.0, cyl_R_cm=0.5)
    report = validate_params(params)
    assert report.is_valid
    assert any("2R > d" in item.message for item in report.warnings)
