from collections.abc import Callable

from nicegui import ui

from core.models import AxisMode, DipolarMode, SimulationParams, ThermalMode

DIPOLAR_OPTIONS = {
    int(DipolarMode.OFF): "No dipolar interaction",
    int(DipolarMode.MEAN_FIELD): "Kharitonskii mean-field closure",
    int(DipolarMode.GAUSSIAN): "Kharitonskii Gaussian random-field closure",
}
AXIS_OPTIONS = {
    int(AxisMode.ALIGNED): "Aligned easy axes",
    int(AxisMode.RANDOM): "Random easy axes",
}
THERMAL_OPTIONS = {
    int(ThermalMode.OFF): "Disabled",
    int(ThermalMode.ON): "Thermal sLLG",
}


class ControlPanel:
    def __init__(self) -> None:
        defaults = SimulationParams()
        self._elements = []

        self._heading("Computational")
        self.N = self._number("Ensemble size N", defaults.N, step=1, integer=True)
        self.n_steps = self._number(
            "Integration steps", defaults.n_steps, step=1, integer=True
        )
        self.dt_ps = self._number("Time step dt (ps)", defaults.dt * 1.0e12)
        self.seed = self._number("Random seed", 1, step=1, integer=True)

        self._heading("Material · CGS")
        self.Ms = self._number("Ms (emu/cm³)", defaults.Ms)
        self.K = self._number("K (erg/cm³)", defaults.K)
        self.alpha = self._number("Gilbert damping α", defaults.alpha)
        self.gamma = self._number("γ (rad/(s·Oe))", defaults.gamma)
        self.T = self._number("Temperature (K)", defaults.T)
        self.d_nm = self._number("Particle diameter (nm)", defaults.d_nm)

        self._heading("Sample and interactions")
        self.c = self._number("Physical concentration c", defaults.c)
        self.cyl_d_um = self._number(
            "Cylinder height d (µm)", defaults.cyl_d_cm * 1.0e4
        )
        self.cyl_R_um = self._number(
            "Cylinder radius R (µm)", defaults.cyl_R_cm * 1.0e4
        )
        self.dipolar_mode = self._select(
            "Dipolar model", DIPOLAR_OPTIONS, int(defaults.dipolar_mode)
        )

        self._heading("Protocol")
        self.H0 = self._number("External field H0 along Oz (Oe)", defaults.H0)
        self.thermal_mode = self._select(
            "Thermal mode", THERMAL_OPTIONS, int(defaults.thermal_mode)
        )
        self.axis_mode = self._select(
            "Easy-axis configuration", AXIS_OPTIONS, int(defaults.axis_mode)
        )
        self.p_up = self._number("Initial fraction p_up", defaults.p_up)
        self.axis_mode.on_value_change(lambda _: self._sync_axis_controls())
        self._sync_axis_controls()

    def _heading(self, text: str) -> None:
        ui.label(text).classes("text-subtitle1 text-weight-medium q-mt-sm")

    def _number(
        self,
        label: str,
        value: float,
        *,
        step: float | None = None,
        integer: bool = False,
    ):
        element = ui.number(label, value=value, format="%.0f" if integer else None)
        element.classes("w-full")
        if step is not None:
            element.props(f"step={step}")
        self._elements.append(element)
        return element

    def _select(self, label: str, options: dict[int, str], value: int):
        element = ui.select(options=options, value=value, label=label).classes("w-full")
        self._elements.append(element)
        return element

    @staticmethod
    def _float(element, label: str) -> float:
        if element.value is None:
            raise ValueError(f"{label} is required")
        return float(element.value)

    @classmethod
    def _integer(cls, element, label: str) -> int:
        value = cls._float(element, label)
        if not value.is_integer():
            raise ValueError(f"{label} must be an integer")
        return int(value)

    def read(self) -> tuple[SimulationParams, int]:
        axis_mode = AxisMode(int(self.axis_mode.value))
        p_up = 0.5 if axis_mode is AxisMode.RANDOM else self._float(self.p_up, "p_up")
        params = SimulationParams(
            N=self._integer(self.N, "N"),
            n_steps=self._integer(self.n_steps, "n_steps"),
            Ms=self._float(self.Ms, "Ms"),
            K=self._float(self.K, "K"),
            alpha=self._float(self.alpha, "alpha"),
            gamma=self._float(self.gamma, "gamma"),
            T=self._float(self.T, "T"),
            thermal_mode=ThermalMode(int(self.thermal_mode.value)),
            d_nm=self._float(self.d_nm, "d_nm"),
            c=self._float(self.c, "c"),
            cyl_d_cm=self._float(self.cyl_d_um, "cylinder height") * 1.0e-4,
            cyl_R_cm=self._float(self.cyl_R_um, "cylinder radius") * 1.0e-4,
            dipolar_mode=DipolarMode(int(self.dipolar_mode.value)),
            H0=self._float(self.H0, "H0"),
            dt=self._float(self.dt_ps, "dt") * 1.0e-12,
            p_up=p_up,
            axis_mode=axis_mode,
        )
        return params, self._integer(self.seed, "seed")

    def set_disabled(self, disabled: bool) -> None:
        for element in self._elements:
            element.disable() if disabled else element.enable()
        if not disabled:
            self._sync_axis_controls()

    def on_model_change(self, handler: Callable[[], None]) -> None:
        for element in (
            self.axis_mode,
            self.dipolar_mode,
            self.cyl_d_um,
            self.cyl_R_um,
            self.d_nm,
        ):
            element.on_value_change(lambda _, callback=handler: callback())

    def _sync_axis_controls(self) -> None:
        if int(self.axis_mode.value) == int(AxisMode.RANDOM):
            self.p_up.value = 0.5
            self.p_up.disable()
        else:
            self.p_up.enable()
