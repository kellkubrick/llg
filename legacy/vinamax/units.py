"""
Класс Units — переключатель системы единиц СИ / СГС.
"""


class Units:
    """
    Переключатель системы единиц: 'SI' или 'CGS'.

    Все вычисления внутри Simulation всегда ведутся в СИ.
    Units конвертирует входные данные пользователя → СИ и обратно.

    Пример:
        units = Units("CGS")
        ms_si = units.msat_to_si(480)      # 480 эму/см³  → 480 000 А/м
        ku_si = units.ku_to_si(1.1e5)      # 1.1e5 эрг/см³ → 1.1e4 Дж/м³
        B_si  = units.field_to_si(100)     # 100 Oe → 0.01 Тл
        r_si  = units.radius_to_si(10)     # 10 нм  → 10e-9 м
    """

    _MSAT = 1e3  # эму/см³  → А/м
    _KU = 0.1  # эрг/см³  → Дж/м³
    _FIELD = 1e-4  # Oe       → Тл
    _ENERGY = 1e-7  # эрг      → Дж
    _LENGTH = 1e-9  # нм       → м

    def __init__(self, system="SI"):
        s = system.upper()
        if s not in ("SI", "CGS"):
            raise ValueError(f"Система единиц: 'SI' или 'CGS', получено {system!r}")
        self.system = s
        self.is_cgs = s == "CGS"

    # ── конверсия ввода: пользователь → внутреннее СИ ──

    def msat_to_si(self, v: float) -> float:
        """Намагниченность насыщения: эму/см³ → А/м"""
        return v * self._MSAT if self.is_cgs else v

    def ku_to_si(self, v: float) -> float:
        """Константа анизотропии: эрг/см³ → Дж/м³"""
        return v * self._KU if self.is_cgs else v

    def field_to_si(self, v: float) -> float:
        """Магнитное поле: Oe → Тл"""
        return v * self._FIELD if self.is_cgs else v

    def energy_to_si(self, v: float) -> float:
        """Энергия: эрг → Дж"""
        return v * self._ENERGY if self.is_cgs else v

    def radius_to_si(self, v: float) -> float:
        """Радиус: нм → м (нм удобны в обеих системах)"""
        return v * self._LENGTH

    # ── конверсия вывода: СИ → пользователь ──

    def msat_from_si(self, v: float) -> float:
        return v / self._MSAT if self.is_cgs else v

    def ku_from_si(self, v: float) -> float:
        return v / self._KU if self.is_cgs else v

    def field_from_si(self, v: float) -> float:
        return v / self._FIELD if self.is_cgs else v

    def energy_from_si(self, v: float) -> float:
        return v / self._ENERGY if self.is_cgs else v

    def radius_from_si(self, v: float) -> float:
        return v / self._LENGTH

    # ── обёртки для B_ext: функция пользователя (Oe) → внутреннее (Тл) ──

    def wrap_B_ext(self, func):
        """
        Если СГС: функция пользователя возвращает Oe → конвертируем в Тл.
        Если СИ:  функция уже в Тл, не трогаем.
        """
        if not self.is_cgs:
            return func
        f = self._FIELD

        def _w(t):
            bx, by, bz = func(t)
            return bx * f, by * f, bz * f

        return _w

    def wrap_B_ext_space(self, func):
        if not self.is_cgs:
            return func
        f = self._FIELD

        def _w(t, x, y, z):
            bx, by, bz = func(t, x, y, z)
            return bx * f, by * f, bz * f

        return _w

    # ── метки единиц для вывода ──

    @property
    def lbl_msat(self):
        return "эму/см³" if self.is_cgs else "А/м"

    @property
    def lbl_ku(self):
        return "эрг/см³" if self.is_cgs else "Дж/м³"

    @property
    def lbl_field(self):
        return "Oe" if self.is_cgs else "Тл"

    @property
    def lbl_energy(self):
        return "эрг" if self.is_cgs else "Дж"

    @property
    def lbl_radius(self):
        return "нм"

    def __repr__(self):
        return f"Units(system={self.system!r})"
