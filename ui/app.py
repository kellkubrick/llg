from nicegui import run, ui

from core import ParameterValidationError, run_relaxation, validate_params
from core.models import ValidationMessage, ValidationReport
from ui.controls import ControlPanel
from ui.export import export_result
from ui.plots import ResultPlots
from ui.state import RunStatus, UIState


def _render_messages(
    container,
    report: ValidationReport,
) -> None:
    container.clear()
    with container:
        for item in report.errors:
            ui.label(item.message).classes("text-negative")
        for item in report.warnings:
            ui.label(item.message).classes("text-warning")


@ui.page("/")
def index() -> None:
    state = UIState()

    with ui.header().classes("items-center"):
        ui.label("sLLG relaxation").classes("text-h5")
        ui.space()
        ui.label("CGS · JAX").classes("text-caption")

    with ui.row().classes("w-full items-start q-pa-md gap-4"):
        with ui.card().classes("w-full max-w-md q-pa-md"):
            ui.label("Simulation parameters").classes("text-h6")
            controls = ControlPanel()

        with ui.column().classes("grow min-w-0 gap-3"):
            with ui.card().classes("w-full q-pa-md"):
                with ui.row().classes("w-full items-center"):
                    status_badge = ui.badge(RunStatus.IDLE.value, color="grey")
                    status_label = ui.label("Ready")
                    spinner = ui.spinner(size="md")
                    spinner.visible = False
                    ui.space()
                    run_button = ui.button("Run", icon="play_arrow")
                    export_button = ui.button("Export", icon="download")
                    export_button.disable()

                messages = ui.column().classes("w-full gap-1")
                summary = ui.label(
                    "The first run for a new N and n_steps includes JIT compilation."
                ).classes("text-caption text-grey-7")

            with ui.card().classes("w-full q-pa-sm"):
                plots = ResultPlots()

    def set_status(status: RunStatus, text: str) -> None:
        state.status = status
        status_badge.set_text(status.value)
        colors = {
            RunStatus.IDLE: "grey",
            RunStatus.RUNNING: "primary",
            RunStatus.READY: "positive",
            RunStatus.ERROR: "negative",
        }
        status_badge.props(f"color={colors[status]}")
        status_label.set_text(text)

    def refresh_warnings() -> None:
        try:
            params, _ = controls.read()
        except (TypeError, ValueError):
            return
        report = validate_params(params)
        state.report = report
        _render_messages(messages, ValidationReport(warnings=report.warnings))

    async def handle_run() -> None:
        try:
            params, seed = controls.read()
        except (TypeError, ValueError) as exc:
            set_status(RunStatus.ERROR, "Invalid input")
            _render_messages(
                messages,
                ValidationReport(errors=(ValidationMessage("form", str(exc)),)),
            )
            return

        report = validate_params(params)
        state.params = params
        state.report = report
        _render_messages(messages, report)
        if not report.is_valid:
            set_status(RunStatus.ERROR, "Validation failed")
            return

        controls.set_disabled(True)
        run_button.disable()
        export_button.disable()
        spinner.visible = True
        set_status(RunStatus.RUNNING, "Calculating (JIT compilation may be included)…")
        state.error = None

        try:
            result = await run.io_bound(run_relaxation, params, seed)
            if result is None:
                raise RuntimeError("The calculation was interrupted before completion")
            set_status(RunStatus.RUNNING, "Rendering results…")
            state.result = result
            plots.update(result)
            summary.set_text(
                f"{result.backend.upper()} · {result.device} · "
                f"compile {result.compile_seconds:.3f} s · "
                f"execute {result.execution_seconds:.3f} s · "
                f"transfer {result.transfer_seconds:.3f} s · "
                f"σrf {result.sigma_rf:.3g} Oe · Hk {result.Hk:.3g} Oe"
            )
            set_status(RunStatus.READY, "Result ready")
            export_button.enable()
        except ParameterValidationError as exc:
            state.error = str(exc)
            state.report = exc.report
            _render_messages(messages, exc.report)
            set_status(RunStatus.ERROR, "Validation failed")
        except Exception as exc:  # noqa: BLE001 - UI boundary must keep the app alive
            state.error = str(exc)
            _render_messages(
                messages,
                ValidationReport(
                    errors=(ValidationMessage("runtime", f"Calculation failed: {exc}"),),
                    warnings=report.warnings,
                ),
            )
            set_status(RunStatus.ERROR, "Calculation failed")
        finally:
            spinner.visible = False
            controls.set_disabled(False)
            run_button.enable()

    async def handle_export() -> None:
        if state.result is None or state.params is None:
            return
        export_button.disable()
        try:
            warnings = state.report.warnings if state.report is not None else ()
            artifacts = await run.io_bound(
                export_result,
                state.result,
                state.params,
                warnings,
            )
            if artifacts is None:
                raise RuntimeError("Export was interrupted")
            ui.notify(f"Saved to {artifacts.directory}", type="positive")
        except Exception as exc:  # noqa: BLE001 - UI boundary must keep the app alive
            ui.notify(f"Export failed: {exc}", type="negative")
        finally:
            export_button.enable()

    run_button.on_click(handle_run)
    export_button.on_click(handle_export)
    controls.on_model_change(refresh_warnings)
    refresh_warnings()


def main() -> None:
    ui.run(title="sLLG relaxation", reload=False)
