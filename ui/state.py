from dataclasses import dataclass
from enum import StrEnum

from core.models import SimulationParams, SimulationResult, ValidationReport


class RunStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    READY = "ready"
    ERROR = "error"


@dataclass(slots=True)
class UIState:
    status: RunStatus = RunStatus.IDLE
    params: SimulationParams | None = None
    result: SimulationResult | None = None
    report: ValidationReport | None = None
    error: str | None = None
