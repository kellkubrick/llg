from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import numpy as np
from numpy.typing import NDArray


class DipolarMode(IntEnum):
    OFF = 0
    MEAN_FIELD = 1
    GAUSSIAN = 2


class AxisMode(IntEnum):
    ALIGNED = 0
    RANDOM = 1


class ThermalMode(IntEnum):
    OFF = 0
    ON = 1


@dataclass(frozen=True, slots=True)
class SimulationParams:
    N: int = 100_000
    n_steps: int = 20_000
    Ms: float = 480.0
    K: float = 1.35e5
    alpha: float = 0.1
    gamma: float = 1.76e7
    T: float = 300.0
    thermal_mode: ThermalMode = ThermalMode.ON
    d_nm: float = 10.0
    c: float = 0.06
    cyl_d_cm: float = 1.0e-4
    cyl_R_cm: float = 0.5e-4
    dipolar_mode: DipolarMode = DipolarMode.GAUSSIAN
    H0: float = 20.0
    dt: float = 1.0e-12
    p_up: float = 0.5
    axis_mode: AxisMode = AxisMode.ALIGNED


@dataclass(frozen=True, slots=True)
class ValidationMessage:
    field: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[ValidationMessage, ...] = ()
    warnings: tuple[ValidationMessage, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.errors


class ParameterValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        details = "; ".join(item.message for item in report.errors)
        super().__init__(details)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    t: NDArray[np.floating]
    mz: NDArray[np.floating]
    mean_rf: NDArray[np.floating]
    sigma_rf: float
    Hk: float
    c: float
    seed: int
    backend: str
    device: str
    compile_seconds: float
    execution_seconds: float
    transfer_seconds: float
    total_seconds: float


def _valid_enum(value: object, enum_type: type[IntEnum]) -> bool:
    try:
        enum_type(value)
    except (TypeError, ValueError):
        return False
    return True


def validate_params(params: SimulationParams) -> ValidationReport:
    errors: list[ValidationMessage] = []
    warnings: list[ValidationMessage] = []

    def positive(field: str, value: float) -> None:
        if not isfinite(value) or value <= 0:
            errors.append(ValidationMessage(field, f"{field} must be greater than zero"))

    def nonnegative(field: str, value: float) -> None:
        if not isfinite(value) or value < 0:
            errors.append(ValidationMessage(field, f"{field} must be non-negative"))

    if isinstance(params.N, bool) or not isinstance(params.N, int) or params.N <= 0:
        errors.append(ValidationMessage("N", "N must be a positive integer"))
    if (
        isinstance(params.n_steps, bool)
        or not isinstance(params.n_steps, int)
        or params.n_steps <= 0
    ):
        errors.append(ValidationMessage("n_steps", "n_steps must be a positive integer"))

    positive("dt", params.dt)
    positive("Ms", params.Ms)
    nonnegative("K", params.K)
    nonnegative("alpha", params.alpha)
    positive("gamma", params.gamma)
    nonnegative("T", params.T)
    positive("d_nm", params.d_nm)
    positive("cyl_d_cm", params.cyl_d_cm)
    positive("cyl_R_cm", params.cyl_R_cm)

    if not isfinite(params.c) or not 0.0 <= params.c <= 1.0:
        errors.append(ValidationMessage("c", "c must be between 0 and 1"))
    if not isfinite(params.p_up) or not 0.0 <= params.p_up <= 1.0:
        errors.append(ValidationMessage("p_up", "p_up must be between 0 and 1"))

    if not _valid_enum(params.axis_mode, AxisMode):
        errors.append(ValidationMessage("axis_mode", "axis_mode must be 0 or 1"))
    if not _valid_enum(params.dipolar_mode, DipolarMode):
        errors.append(
            ValidationMessage("dipolar_mode", "dipolar_mode must be 0, 1, or 2")
        )
    if not _valid_enum(params.thermal_mode, ThermalMode):
        errors.append(
            ValidationMessage("thermal_mode", "thermal_mode must be 0 or 1")
        )

    if errors:
        return ValidationReport(tuple(errors), tuple(warnings))

    axis_mode = AxisMode(params.axis_mode)
    dipolar_mode = DipolarMode(params.dipolar_mode)
    if axis_mode is AxisMode.RANDOM and dipolar_mode is not DipolarMode.OFF:
        warnings.append(
            ValidationMessage(
                "axis_mode",
                "Random easy axes with the Kharitonskii Oz closure are a "
                "phenomenological longitudinal random-field approximation.",
            )
        )
    if axis_mode is AxisMode.RANDOM:
        warnings.append(
            ValidationMessage(
                "p_up",
                "For random easy axes, p_up does not control global Mz; the UI uses 0.5.",
            )
        )
    if 2.0 * params.cyl_R_cm <= params.cyl_d_cm:
        warnings.append(
            ValidationMessage(
                "cyl_R_cm",
                "The Kharitonskii table used by this model assumes 2R > d.",
            )
        )

    d0_cm = params.d_nm * 1.0e-7
    variance_factor = 1.0 - 3.75 * (d0_cm / params.cyl_d_cm) ** 3
    if variance_factor < 0.0:
        warnings.append(
            ValidationMessage(
                "cyl_d_cm",
                "The selected particle/sample geometry makes the approximate "
                "random-field variance negative; the kernel will use zero variance.",
            )
        )

    return ValidationReport(tuple(errors), tuple(warnings))
