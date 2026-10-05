import csv
import json
from pathlib import Path

import numpy as np

from core.models import SimulationParams, SimulationResult, ValidationMessage
from ui.export import export_result
from ui.plots import magnetization_figure, mean_field_figure


def example_result() -> SimulationResult:
    return SimulationResult(
        t=np.asarray([0.0, 1.0e-9]),
        mz=np.asarray([0.0, 0.25]),
        mean_rf=np.asarray([0.0, -2.0]),
        sigma_rf=4.0,
        Hk=562.5,
        c=0.06,
        seed=1,
        backend="cpu",
        device="TFRT_CPU_0",
        compile_seconds=0.1,
        execution_seconds=0.2,
        transfer_seconds=0.01,
        total_seconds=0.31,
    )


def test_export_writes_timeseries_and_metadata(tmp_path: Path) -> None:
    result = example_result()
    warning = ValidationMessage("axis_mode", "phenomenological approximation")
    artifacts = export_result(
        result,
        SimulationParams(),
        warnings=(warning,),
        root=tmp_path,
    )

    with artifacts.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["t_s", "mz", "mean_rf_oe"]
    assert len(rows) == 3

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["parameters_canonical"]["dt"] == SimulationParams().dt
    assert metadata["runtime"]["backend"] == "cpu"
    assert metadata["warnings"][0]["field"] == "axis_mode"


def test_plot_figures_use_nanoseconds_and_expected_series() -> None:
    result = example_result()
    magnetization = magnetization_figure(result)
    mean_field = mean_field_figure(result)
    np.testing.assert_allclose(magnetization.data[0].x, [0.0, 1.0])
    np.testing.assert_allclose(magnetization.data[0].y, result.mz)
    np.testing.assert_allclose(mean_field.data[0].y, result.mean_rf)
    assert magnetization.layout.xaxis.title.text == "Time (ns)"
