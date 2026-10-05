import plotly.graph_objects as go
from nicegui import ui

from core.models import SimulationResult


def _base_figure(title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        xaxis_title="Time (ns)",
        yaxis_title=y_title,
        margin={"l": 55, "r": 20, "t": 55, "b": 50},
        template="plotly_white",
        height=330,
    )
    return figure


def magnetization_figure(result: SimulationResult | None = None) -> go.Figure:
    figure = _base_figure("Ensemble magnetization", "⟨m_z⟩")
    if result is not None:
        figure.add_scatter(
            x=result.t * 1.0e9,
            y=result.mz,
            mode="lines",
            name="Mz",
        )
    return figure


def mean_field_figure(result: SimulationResult | None = None) -> go.Figure:
    figure = _base_figure("Applied mean random field", "mean_rf (Oe)")
    if result is not None:
        figure.add_scatter(
            x=result.t * 1.0e9,
            y=result.mean_rf,
            mode="lines",
            name="mean_rf",
        )
    return figure


class ResultPlots:
    def __init__(self) -> None:
        self.magnetization = ui.plotly(magnetization_figure()).classes("w-full")
        self.mean_field = ui.plotly(mean_field_figure()).classes("w-full")

    def update(self, result: SimulationResult) -> None:
        self.magnetization.figure = magnetization_figure(result)
        self.mean_field.figure = mean_field_figure(result)
        self.magnetization.update()
        self.mean_field.update()
