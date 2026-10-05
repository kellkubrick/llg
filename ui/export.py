from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.models import SimulationParams, SimulationResult, ValidationMessage


@dataclass(frozen=True, slots=True)
class ExportArtifacts:
    directory: Path
    csv_path: Path
    metadata_path: Path


def export_result(
    result: SimulationResult,
    params: SimulationParams,
    warnings: tuple[ValidationMessage, ...] = (),
    root: Path = Path("results"),
) -> ExportArtifacts:
    timestamp = datetime.now(UTC)
    run_id = f"{timestamp:%Y%m%dT%H%M%S_%fZ}_seed{result.seed}"
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=False)

    csv_path = directory / "timeseries.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("t_s", "mz", "mean_rf_oe"))
        writer.writerows(zip(result.t, result.mz, result.mean_rf, strict=True))

    metadata = {
        "schema_version": 1,
        "created_at": timestamp.isoformat(),
        "seed": result.seed,
        "parameters_canonical": asdict(params),
        "scalars": {
            "sigma_rf_oe": result.sigma_rf,
            "Hk_oe": result.Hk,
            "c": result.c,
        },
        "runtime": {
            "backend": result.backend,
            "device": result.device,
            "compile_seconds": result.compile_seconds,
            "execution_seconds": result.execution_seconds,
            "transfer_seconds": result.transfer_seconds,
            "total_seconds": result.total_seconds,
        },
        "warnings": [asdict(item) for item in warnings],
    }
    metadata_path = directory / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return ExportArtifacts(directory, csv_path, metadata_path)
