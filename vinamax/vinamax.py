"""
vinamax.py — Python-порт симулятора магнитных наночастиц vinamax
Оригинал: Jonathan Leliaert (Go), https://github.com/JLeliaert/vinamax
Порт на Python + расширения: Claude (Anthropic)

Решает уравнение Ландау-Лифшица-Гильберта (LLG) для системы
магнитных наночастиц с учётом:
  - Поля Зеемана (внешнее поле)
  - Одноосной и кубической анизотропии
  - Демагнетизирующего диполь-дипольного взаимодействия
  - Теплового шума Брауна (LLG с флуктуациями)
  - Прыжкового шума Нееля (суперпарамагнитные частицы)
  - Логнормального распределения по размерам частиц
  - Смешанных ансамблей: заблокированные (Blocked) + суперпарамагнитные (SP)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from enum import Enum

# ─── Физические константы (СИ) ────────────────────────────────────────────────

GAMMA0 = 1.7595e11           # Гиромагнитное отношение, рад/(Т·с)
MU0    = 4 * math.pi * 1e-7  # Магнитная проницаемость вакуума, Тм/А
KB     = 1.3806488e-23       # Постоянная Больцмана, Дж/К

# ─── Физические константы (СГС) ───────────────────────────────────────────────

GAMMA0_CGS = 1.7595e7        # рад/(Эрстед·с)  [= GAMMA0 / 1e4]
KB_CGS     = 1.3806488e-16   # эрг/К

# ─── Таблица перевода ─────────────────────────────────────────────────────────
#
#  Величина        СГС              СИ         Множитель СГС→СИ
#  Намагниченность эму/см³          А/м         × 1e3
#  Анизотропия Ku  эрг/см³          Дж/м³       × 0.1
#  Поле            Эрстед (Oe)      Тл          × 1e-4
#  Энергия         эрг              Дж          × 1e-7
#  Радиус          нм               м           × 1e-9  (удобно в обоих)
#  α, T, t, m      б/р, К, с, б/р   те же       × 1


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

    _MSAT   = 1e3     # эму/см³  → А/м
    _KU     = 0.1     # эрг/см³  → Дж/м³
    _FIELD  = 1e-4    # Oe       → Тл
    _ENERGY = 1e-7    # эрг      → Дж
    _LENGTH = 1e-9    # нм       → м

    def __init__(self, system="SI"):
        s = system.upper()
        if s not in ("SI", "CGS"):
            raise ValueError(
                f"Система единиц: 'SI' или 'CGS', получено {system!r}"
            )
        self.system = s
        self.is_cgs = (s == "CGS")

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
    def lbl_msat(self):   return "эму/см³" if self.is_cgs else "А/м"
    @property
    def lbl_ku(self):     return "эрг/см³" if self.is_cgs else "Дж/м³"
    @property
    def lbl_field(self):  return "Oe"      if self.is_cgs else "Тл"
    @property
    def lbl_energy(self): return "эрг"     if self.is_cgs else "Дж"
    @property
    def lbl_radius(self): return "нм"

    def __repr__(self):
        return f"Units(system={self.system!r})"


# ─── Тип частицы ──────────────────────────────────────────────────────────────

class ParticleType(Enum):
    BLOCKED = "blocked"   # момент заморожен, интегрируется через LLG
    SP      = "sp"        # суперпарамагнитный, флипы через прыжковый шум
    AUTO    = "auto"      # определяется автоматически

# ─── Вспомогательные функции ──────────────────────────────────────────────────

def vnorm(v):
    mag = np.linalg.norm(v)
    return v / mag if mag > 0 else v.copy()

def cube(x):
    return x * x * x

def sphere_volume(r):
    return (4. / 3.) * math.pi * cube(r)

def neel_time(Ku, r, T, tau0=1e-9):
    """τ_N = τ₀ · exp(Ku·V / (kB·T))"""
    if T <= 0 or Ku <= 0:
        return math.inf
    V = sphere_volume(r)
    sigma = Ku * V / (KB * T)
    if sigma > 700:
        return math.inf
    return tau0 * math.exp(sigma)

def blocking_radius(Ku, T, tau_obs=100., tau0=1e-9):
    """r_crit: радиус, при котором τ_N = τ_obs"""
    if Ku <= 0 or T <= 0:
        return 0.
    ln_factor = math.log(tau_obs / tau0)
    return (3. * KB * T * ln_factor / (4. * math.pi * Ku)) ** (1. / 3.)

# ─── Класс частицы ────────────────────────────────────────────────────────────

@dataclass
class Particle:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    r_h: float = 0.0
    msat: float = 0.0
    eta: float = 0.0
    ptype: ParticleType = ParticleType.AUTO

    m: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    u_anis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    u2_anis: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    c1_anis: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    c2_anis: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    c3_anis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    biasfield: np.ndarray = field(default_factory=lambda: np.zeros(3))
    demagnetising_field: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heff: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tempfield: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dmdt: np.ndarray = field(default_factory=lambda: np.zeros(3))
    previousm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tempm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk3: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk4: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fehlk5: np.ndarray = field(default_factory=lambda: np.zeros(3))
    temp_prefactor: float = 0.0
    randomvprefact: float = 0.0
    switch_time: float = 0.0
    fixed: bool = False

    def volume(self):
        return sphere_volume(self.r)

    def dist(self, x, y, z):
        return math.sqrt((self.x-x)**2 + (self.y-y)**2 + (self.z-z)**2)

    def calc_temp_prefactor(self, alpha, temp):
        if self.msat > 0 and temp > 0:
            self.temp_prefactor = math.sqrt(
                (2. * KB * alpha * temp) / (GAMMA0 * self.msat * self.volume())
            )

    def calc_randomv_prefact(self, temp):
        if self.eta > 0 and temp > 0:
            vol_h = sphere_volume(self.r_h)
            self.randomvprefact = math.sqrt(
                (2. * KB * temp) / (6. * self.eta * vol_h)
            )

    def neel_relaxation_time(self, Ku, T, tau0=1e-9):
        return neel_time(Ku, self.r, T, tau0)

    def sigma(self, Ku, T):
        if T <= 0 or Ku <= 0:
            return math.inf
        return Ku * self.volume() / (KB * T)

    def __repr__(self):
        return (f"Particle(r={self.r*1e9:.1f}nm, "
                f"type={self.ptype.value}, m={np.round(self.m, 3)})")


# ─── Распределение по размерам ────────────────────────────────────────────────

@dataclass
class SizeDistribution:
    """
    Логнормальное распределение радиусов:
        P(r) ∝ (1/r) · exp(-(ln r - ln r_median)² / (2·ln²σ_geo))

    Параметры:
        r_median  — медианный радиус, м
        sigma_geo — геометрическое стандартное отклонение (1.1–1.5 типично)
    """
    r_median: float
    sigma_geo: float = 1.3

    def sample(self, n, rng):
        mu  = math.log(self.r_median)
        sig = math.log(self.sigma_geo)
        return np.exp(rng.normal(mu, sig, n))

    def pdf(self, r_values):
        mu  = math.log(self.r_median)
        sig = math.log(self.sigma_geo)
        ln_r = np.log(r_values)
        return np.exp(-0.5 * ((ln_r - mu) / sig) ** 2) / (r_values * sig * math.sqrt(2*math.pi))

    @property
    def mean_r(self):
        sig = math.log(self.sigma_geo)
        return self.r_median * math.exp(sig**2 / 2.)

    @property
    def mode_r(self):
        sig = math.log(self.sigma_geo)
        return self.r_median * math.exp(-sig**2)

    def __repr__(self):
        return (f"SizeDistribution(r_med={self.r_median*1e9:.1f}nm, "
                f"σ_geo={self.sigma_geo:.2f}, <r>={self.mean_r*1e9:.1f}nm)")


# ─── Основной класс симулятора ────────────────────────────────────────────────

class Simulation:
    """
    Симулятор системы магнитных наночастиц.

    Поддерживает:
    • Однородный размер или логнормальное распределение
    • Смешанный ансамбль Blocked (LLG + тепловой шум Брауна)
                        + SP (прыжки Нееля)
    • Автоматическую классификацию по критерию σ = KuV/(kBT)
    • Численные методы: euler, heun, rk4, dopri
    """

    def __init__(self, units="SI"):
        """
        Параметр units: 'SI' (по умолчанию) или 'CGS'.

        В режиме CGS:
          • msat()            принимает эму/см³
          • Ku1, Ku2, Kc1     задаются в эрг/см³
          • B_ext             возвращает Oe
          • particle_radius() принимает нм  (удобно в обоих)
          • E_total() и др.   возвращают эрг
        """
        self.units = Units(units)
        self.particles: List[Particle] = []

        self.Dt: float      = -1.0
        self.Mindt: float   = 1e-20
        self.Maxdt: float   = 1.0
        self.T: float       = 0.0
        self.Alpha: float   = -1.0
        self.Temp: float    = -1.0
        # Ku хранятся в СИ внутри; при CGS используйте set_ku() или свойство ku1
        self.Ku1: float     = 0.0  # внутреннее: Дж/м³
        self.Ku2: float     = 0.0
        self.Kc1: float     = 0.0
        self.Tau0: float    = 1e-9
        self.Errortolerance = 1e-5

        self.Demag            = True
        self.FMM              = False
        self.Adaptivestep     = False
        self.Brown            = False
        self.Jumpnoise        = False
        self.BrownianRotation = False
        self.solver           = "rk4"
        self.order            = 4
        self.viscosity        = 0.0

        # Порог для классификации: σ > порога → Blocked
        self.sigma_threshold  = 3.0

        self._gammaoveralpha  = 0.0
        self._maxtauwitht     = 0.0
        self._relax           = False
        self._outputinterval  = 0.0
        self._twrite          = 0.0
        self._outfile         = None
        self._n_blocked       = 0
        self._n_sp            = 0

        self._B_ext_raw       = lambda t: (0., 0., 0.)
        self._B_ext_space_raw = lambda t, x, y, z: (0., 0., 0.)
        self.B_ext       = lambda t: (0., 0., 0.)
        self.B_ext_space = lambda t, x, y, z: (0., 0., 0.)

        self._rng       = np.random.default_rng(42)
        self._rng_anis  = np.random.default_rng(42)

        self._world_cx   = 0.
        self._world_cy   = 0.
        self._world_cz   = 0.
        self._world_size = 1e-3
        self._const_radius   = 0.
        self._const_radius_h = 0.

    # ══════════════════════════════════════════════════════════════════════════
    # НАСТРОЙКА
    # ══════════════════════════════════════════════════════════════════════════

    def set_random_seed(self, seed):
        self._rng = np.random.default_rng(seed)

    def world(self, x_nm, y_nm, z_nm, size_nm):
        """Задать куб мира в нм (удобно для обеих систем единиц)."""
        self._world_cx   = self.units.radius_to_si(x_nm)
        self._world_cy   = self.units.radius_to_si(y_nm)
        self._world_cz   = self.units.radius_to_si(z_nm)
        self._world_size = self.units.radius_to_si(size_nm)

    def particle_radius(self, r_nm, r_h_nm=None):
        """Радиус частицы в нм (удобно для обеих систем единиц)."""
        self._const_radius   = self.units.radius_to_si(r_nm)
        rh = r_h_nm if r_h_nm is not None else r_nm
        self._const_radius_h = self.units.radius_to_si(rh)

    def set_viscosity(self, eta):
        self.viscosity = eta

    def msat(self, ms):
        """Намагниченность насыщения: А/м (СИ) или эму/см³ (СГС)."""
        ms_si = self.units.msat_to_si(ms)
        for p in self.particles:
            p.msat = ms_si

    def anisotropy_axis(self, ux, uy, uz):
        u = vnorm(np.array([ux, uy, uz]))
        for p in self.particles:
            p.u_anis = u.copy()

    def anisotropy_axis_random(self):
        """Случайные оси анизотропии — изотропный ансамбль."""
        for p in self.particles:
            phi   = self._rng.uniform(0, 2*math.pi)
            theta = math.acos(1 - 2*float(self._rng.uniform()))
            p.u_anis = vnorm(np.array([
                math.sin(theta)*math.cos(phi),
                math.sin(theta)*math.sin(phi),
                math.cos(theta)
            ]))

    def m_uniform(self, mx, my, mz):
        m = vnorm(np.array([mx, my, mz]))
        for p in self.particles:
            if not p.fixed:
                p.m = m.copy()

    def m_random(self):
        for p in self.particles:
            if not p.fixed:
                phi   = self._rng.uniform(0, 2*math.pi)
                theta = math.acos(1 - 2*float(self._rng.uniform()))
                p.m = vnorm(np.array([
                    math.sin(theta)*math.cos(phi),
                    math.sin(theta)*math.sin(phi),
                    math.cos(theta)
                ]))

    # ── Свойства для Ku в текущей системе единиц ──

    @property
    def ku1(self) -> float:
        """Ku1 в текущей системе единиц (Дж/м³ или эрг/см³)."""
        return self.units.ku_from_si(self.Ku1)

    @ku1.setter
    def ku1(self, v: float):
        """Задать Ku1 в текущей системе единиц."""
        self.Ku1 = self.units.ku_to_si(v)

    @property
    def ku2(self) -> float:
        return self.units.ku_from_si(self.Ku2)

    @ku2.setter
    def ku2(self, v: float):
        self.Ku2 = self.units.ku_to_si(v)

    @property
    def kc1(self) -> float:
        return self.units.ku_from_si(self.Kc1)

    @kc1.setter
    def kc1(self, v: float):
        self.Kc1 = self.units.ku_to_si(v)

    def set_solver(self, name):
        solvers = {"euler": 1, "heun": 2, "rk4": 4, "dopri": 5}
        if name not in solvers:
            raise ValueError(f"Неизвестный метод: {name}. Доступные: {list(solvers)}")
        self.solver = name
        self.order  = solvers[name]

    def output(self, interval, path="vinamax_out.txt"):
        self._outputinterval = interval
        self._twrite         = interval
        self._outfile        = open(path, "w")
        sys_label = self.units.system
        self._outfile.write(
            f"# Система единиц: {sys_label}\n"
            f"# t(с)\t<mx>\t<my>\t<mz>\tn_blocked\tn_sp\n"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ДОБАВЛЕНИЕ ЧАСТИЦ
    # ══════════════════════════════════════════════════════════════════════════

    def add_particle(self, x_nm, y_nm, z_nm, ptype=ParticleType.AUTO):
        """Координаты в нм (нм удобны для обеих систем единиц)."""
        p = Particle(
            x=self.units.radius_to_si(x_nm),
            y=self.units.radius_to_si(y_nm),
            z=self.units.radius_to_si(z_nm),
            r=self._const_radius,
            r_h=self._const_radius_h,
            ptype=ptype
        )
        if self.BrownianRotation:
            p.eta = self.viscosity
        self.particles.append(p)
        return p

    def add_particles_lognormal(self, n, r_median, sigma_geo=1.3,
                                 r_h_coating=0., ptype=ParticleType.AUTO,
                                 seed=None, max_attempts=5000):
        """
        Добавить N частиц с логнормальным распределением радиусов.

        Параметры
        ---------
        n           : количество частиц
        r_median    : медианный радиус, м  (напр. 10e-9 = 10 нм)
        sigma_geo   : геометрическое ст. откл. (1.0 = монодисперс, 1.3 типично)
        r_h_coating : толщина оболочки для гидродинамического радиуса, м
        ptype       : ParticleType.AUTO → авто-классификация по σ = KuV/(kBT)
        seed        : зерно ГСЧ для воспроизводимости
        max_attempts: лимит попыток размещения без перекрытий
        """
        # r_median и r_h_coating в нм → конвертируем в м
        r_median_si    = self.units.radius_to_si(r_median)
        r_h_coating_si = self.units.radius_to_si(r_h_coating) if r_h_coating > 0 else 0.
        dist = SizeDistribution(r_median_si, sigma_geo)
        rng_s = np.random.default_rng(
            seed if seed is not None else int(self._rng.integers(int(1e9)))
        )
        radii = dist.sample(n, rng_s)

        added = []
        attempts = 0
        i = 0
        half = self._world_size / 2.
        while i < n and attempts < max_attempts:
            r   = float(radii[i])
            r_h = r + r_h_coating_si
            x = float(self._rng.uniform(self._world_cx - half, self._world_cx + half))
            y = float(self._rng.uniform(self._world_cy - half, self._world_cy + half))
            z = float(self._rng.uniform(self._world_cz - half, self._world_cz + half))

            if self._has_overlap(x, y, z, r_h):
                attempts += 1
                continue

            p = Particle(x=x, y=y, z=z, r=r, r_h=r_h, ptype=ptype)
            if self.BrownianRotation:
                p.eta = self.viscosity
            self.particles.append(p)
            added.append(p)
            i += 1
            attempts = 0

        if len(added) < n:
            print(f"  Предупреждение: размещено {len(added)}/{n} частиц.")
        return added

    def _has_overlap(self, x, y, z, r_h):
        for p in self.particles:
            if p.dist(x, y, z) < (p.r_h + r_h):
                return True
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # КЛАССИФИКАЦИЯ BLOCKED / SP
    # ══════════════════════════════════════════════════════════════════════════

    def classify_particles(self, sigma_threshold=None, tau_obs=100.):
        """
        Разбить частицы на Blocked и SP по критерию σ = Ku·V / (kB·T).

            σ > sigma_threshold  →  Blocked
            σ ≤ sigma_threshold  →  SP

        Чем больше sigma_threshold, тем меньше частиц считаются заблокированными.
        Типичные значения: 2–5 (мягкий), 25 (строгий τ_obs-критерий).
        """
        if sigma_threshold is not None:
            self.sigma_threshold = sigma_threshold

        assert self.Temp > 0, "Задайте Temp > 0 перед classify_particles()"
        assert self.Ku1  > 0, "Задайте Ku1  > 0 перед classify_particles()"

        n_blocked = n_sp = 0
        for p in self.particles:
            if p.ptype != ParticleType.AUTO:
                n_blocked += (p.ptype == ParticleType.BLOCKED)
                n_sp      += (p.ptype == ParticleType.SP)
                continue
            if p.sigma(self.Ku1, self.Temp) > self.sigma_threshold:
                p.ptype = ParticleType.BLOCKED
                n_blocked += 1
            else:
                p.ptype = ParticleType.SP
                n_sp += 1

        self._n_blocked = n_blocked
        self._n_sp      = n_sp
        r_crit = blocking_radius(self.Ku1, self.Temp,
                                 tau_obs=tau_obs, tau0=self.Tau0)
        return {
            "n_total":        len(self.particles),
            "n_blocked":      n_blocked,
            "n_sp":           n_sp,
            "r_critical_nm":  r_crit * 1e9,
            "sigma_threshold": self.sigma_threshold,
        }

    def print_ensemble_info(self):
        radii = np.array([self.units.radius_from_si(p.r) for p in self.particles])
        if self.Ku1 > 0 and self.Temp > 0:
            r_c = blocking_radius(self.Ku1, self.Temp, tau0=self.Tau0)
            print(f"  Критический радиус блокировки: {self.units.radius_from_si(r_c):.2f} нм")
        N = len(self.particles)
        print(f"  Всего частиц: {N}")
        print(f"  Blocked: {self._n_blocked}  ({100*self._n_blocked/max(N,1):.1f}%)")
        print(f"  SP:      {self._n_sp}  ({100*self._n_sp/max(N,1):.1f}%)")
        if len(radii):
            print(f"  Размеры: min={radii.min():.1f}, "
                  f"med={np.median(radii):.1f}, "
                  f"max={radii.max():.1f} нм")

    # ══════════════════════════════════════════════════════════════════════════
    # ПРЫЖКОВЫЙ ШУМ НЕЕЛЯ — для SP частиц
    # ══════════════════════════════════════════════════════════════════════════

    def _reset_switch_time(self, p):
        tau_n = p.neel_relaxation_time(self.Ku1, self.Temp, self.Tau0)
        if tau_n == math.inf or tau_n <= 0:
            p.switch_time = math.inf
        else:
            p.switch_time = self.T + float(self._rng.exponential(tau_n))

    def _init_switch_times(self):
        for p in self.particles:
            if p.ptype == ParticleType.SP:
                self._reset_switch_time(p)

    def _check_switches(self):
        """Флип SP-частицы: m → ±u_anis с учётом поля Зеемана."""
        for p in self.particles:
            if p.ptype != ParticleType.SP:
                continue
            if self.T < p.switch_time:
                continue
            B   = self._zeeman(p)
            m_p =  p.u_anis
            m_m = -p.u_anis
            E_p = -p.msat * p.volume() * float(np.dot(m_p, B))
            E_m = -p.msat * p.volume() * float(np.dot(m_m, B))
            if self.Temp > 0:
                dE = E_p - E_m
                if dE < 0:
                    p.m = m_p.copy()
                else:
                    prob_p = 1. / (1. + math.exp(min(dE/(KB*self.Temp), 500)))
                    p.m = m_p.copy() if float(self._rng.random()) < prob_p else m_m.copy()
            else:
                p.m = m_p.copy() if E_p <= E_m else m_m.copy()
            self._reset_switch_time(p)

    # ══════════════════════════════════════════════════════════════════════════
    # ПОЛЯ
    # ══════════════════════════════════════════════════════════════════════════

    def _zeeman(self, p):
        bx, by, bz = self.B_ext(self.T)
        sx, sy, sz = self.B_ext_space(self.T, p.x, p.y, p.z)
        return np.array([bx+sx, by+sy, bz+sz]) + p.biasfield

    def _anis(self, p):
        if p.msat <= 0:
            return np.zeros(3)
        mdotu  = np.dot(p.m, p.u_anis)
        result = p.u_anis * (2.*self.Ku1*mdotu/p.msat)
        if self.Ku2 != 0:
            mdotu2 = np.dot(p.m, p.u2_anis)
            result += p.u2_anis * (2.*self.Ku2*mdotu2/p.msat)
        if self.Kc1 != 0:
            c1m = np.dot(p.m, p.c1_anis)
            c2m = np.dot(p.m, p.c2_anis)
            c3m = np.dot(p.m, p.c3_anis)
            cubic = (p.c1_anis*c1m*(c3m**2+c2m**2) +
                     p.c2_anis*c2m*(c3m**2+c1m**2) +
                     p.c3_anis*c3m*(c2m**2+c1m**2))
            result += cubic * (-2.*self.Kc1/p.msat)
        return result

    def _thermal_field(self, p):
        if not self.Brown or self.Temp <= 0 or p.temp_prefactor == 0:
            return np.zeros(3)
        return self._rng.standard_normal(3) * (p.temp_prefactor / math.sqrt(self.Dt))

    def _b_eff(self, p, temp):
        return p.demagnetising_field + self._anis(p) + self._zeeman(p) + temp

    # ══════════════════════════════════════════════════════════════════════════
    # ДЕМАГНЕТИЗИРУЮЩЕЕ ПОЛЕ  O(N²)
    # ══════════════════════════════════════════════════════════════════════════

    def _calculate_demag(self):
        for p in self.particles:
            p.demagnetising_field = np.zeros(3)
        N  = len(self.particles)
        pf = MU0 / 3.
        for i in range(N):
            for j in range(i+1, N):
                p1, p2 = self.particles[i], self.particles[j]
                msv1 = cube(p1.r)*p1.msat*pf
                msv2 = cube(p2.r)*p2.msat*pf
                rv   = np.array([p1.x-p2.x, p1.y-p2.y, p1.z-p2.z])
                r    = p1.dist(p2.x, p2.y, p2.z)
                r3, r5 = r**3, r**5
                dp1 = np.dot(p1.m, rv)
                dp2 = np.dot(p2.m, rv)
                p1.demagnetising_field += msv2*(3*dp2*rv/r5 - p2.m/r3)
                p2.demagnetising_field += msv1*(3*dp1*rv/r5 - p1.m/r3)

    # ══════════════════════════════════════════════════════════════════════════
    # LLG
    # ══════════════════════════════════════════════════════════════════════════

    def _tau(self, p, temp):
        p.heff = self._b_eff(p, temp)
        mxB    = np.cross(p.m, p.heff)
        amxmxB = np.cross(p.m, mxB) * self.Alpha
        return -(mxB + amxmxB) * self._gammaoveralpha

    def _no_precess(self, p):
        p.heff = self._b_eff(p, np.zeros(3))
        mxB    = np.cross(p.m, p.heff)
        return np.cross(p.m, mxB) * (-self.Alpha * self._gammaoveralpha)

    # ══════════════════════════════════════════════════════════════════════════
    # ЧИСЛЕННЫЕ МЕТОДЫ (только для Blocked частиц)
    # ══════════════════════════════════════════════════════════════════════════

    def _blocked(self):
        return [p for p in self.particles if p.ptype != ParticleType.SP]

    def _euler_step(self):
        for p in self._blocked():
            temp = self._thermal_field(p)
            tau  = self._tau(p, temp)
            if not p.fixed:
                p.m += tau * self.Dt
            p.m = vnorm(p.m)

    def _heun_step(self):
        blk  = self._blocked()
        ms0  = {id(p): p.m.copy() for p in blk}
        k1s  = {}
        tmps = {}
        for p in blk:
            temp        = self._thermal_field(p)
            tmps[id(p)] = temp
            k1          = self._tau(p, temp)
            k1s[id(p)]  = k1
            if not p.fixed:
                p.m = ms0[id(p)] + k1 * self.Dt
        self.T += self.Dt
        if self.Demag:
            self._calculate_demag()
        for p in blk:
            k2 = self._tau(p, tmps[id(p)])
            if not p.fixed:
                p.m = ms0[id(p)] + (k1s[id(p)] + k2) * 0.5 * self.Dt
            p.m = vnorm(p.m)
        self.T -= self.Dt

    def _rk4_step(self):
        blk  = self._blocked()
        ms0  = {id(p): p.m.copy() for p in blk}
        tmps = {}
        k1s  = {}
        for p in blk:
            temp        = self._thermal_field(p)
            tmps[id(p)] = temp
            k1          = self._tau(p, temp)
            k1s[id(p)]  = k1
            if not p.fixed:
                p.m = ms0[id(p)] + k1 * (self.Dt/2.)
        self.T += self.Dt/2.
        if self.Demag: self._calculate_demag()
        k2s = {}
        for p in blk:
            k2 = self._tau(p, tmps[id(p)])
            k2s[id(p)] = k2
            if not p.fixed:
                p.m = ms0[id(p)] + k2 * (self.Dt/2.)
        if self.Demag: self._calculate_demag()
        k3s = {}
        for p in blk:
            k3 = self._tau(p, tmps[id(p)])
            k3s[id(p)] = k3
            if not p.fixed:
                p.m = ms0[id(p)] + k3 * self.Dt
        self.T += self.Dt/2.
        if self.Demag: self._calculate_demag()
        for p in blk:
            k4 = self._tau(p, tmps[id(p)])
            dm = (k1s[id(p)]/6. + k2s[id(p)]/3. +
                  k3s[id(p)]/3. + k4/6.) * self.Dt
            if not p.fixed:
                p.m = ms0[id(p)] + dm
            p.m = vnorm(p.m)
        self.T -= self.Dt

    def _dopri_step(self):
        blk  = self._blocked()
        ms0  = {id(p): p.m.copy() for p in blk}
        tmps = {}
        k1s  = {}
        for p in blk:
            temp        = self._thermal_field(p) if not self._relax else np.zeros(3)
            tmps[id(p)] = temp
            p.previousm = p.m.copy()
            k           = self._tau(p, temp) if not self._relax else self._no_precess(p)
            k1s[id(p)]  = k
            if not p.fixed:
                p.m = ms0[id(p)] + k * (self.Dt/5.)
        self.T += self.Dt/5.
        if self.Demag: self._calculate_demag()
        k2s = {}
        for p in blk:
            k = self._tau(p, tmps[id(p)]) if not self._relax else self._no_precess(p)
            k2s[id(p)] = k
            if not p.fixed:
                p.m = ms0[id(p)] + (3/40.*k1s[id(p)] + 9/40.*k) * self.Dt
        self.T += self.Dt/10.
        if self.Demag: self._calculate_demag()
        k3s = {}
        for p in blk:
            k = self._tau(p, tmps[id(p)]) if not self._relax else self._no_precess(p)
            k3s[id(p)] = k
            if not p.fixed:
                p.m = ms0[id(p)] + (44/45.*k1s[id(p)] - 56/15.*k2s[id(p)] + 32/9.*k)*self.Dt
        self.T += self.Dt/2.
        if self.Demag: self._calculate_demag()
        k4s = {}
        for p in blk:
            k = self._tau(p, tmps[id(p)]) if not self._relax else self._no_precess(p)
            k4s[id(p)] = k
            if not p.fixed:
                p.m = ms0[id(p)] + (19372/6561.*k1s[id(p)] - 25360/2187.*k2s[id(p)]
                                     + 64448/6561.*k3s[id(p)] - 212/729.*k)*self.Dt
        self.T += (-4/5.+8/9.)*self.Dt
        if self.Demag: self._calculate_demag()
        k5s = {}
        for p in blk:
            k = self._tau(p, tmps[id(p)]) if not self._relax else self._no_precess(p)
            k5s[id(p)] = k
            if not p.fixed:
                p.m = ms0[id(p)] + (9017/3168.*k1s[id(p)] - 355/33.*k2s[id(p)]
                                     + 46732/5247.*k3s[id(p)] + 49/176.*k4s[id(p)]
                                     - 5103/18656.*k)*self.Dt
        self.T += self.Dt/9.
        if self.Demag: self._calculate_demag()
        max_err = 0.
        for p in blk:
            k = self._tau(p, tmps[id(p)]) if not self._relax else self._no_precess(p)
            dm5 = (35/384.*k1s[id(p)] + 500/1113.*k3s[id(p)]
                   + 125/192.*k4s[id(p)] - 2187/6784.*k5s[id(p)] + 11/84.*k)*self.Dt
            err = (71/57600.*k1s[id(p)] - 71/16695.*k3s[id(p)]
                   + 71/1920.*k4s[id(p)] - 17253/339200.*k5s[id(p)] + 22/525.*k)*self.Dt
            max_err = max(max_err, float(np.linalg.norm(err)))
            if not p.fixed:
                p.m = ms0[id(p)] + dm5
            p.m = vnorm(p.m)
        self._maxtauwitht = max_err
        self.T -= self.Dt

    def _undo_bad_step(self):
        for p in self._blocked():
            p.m = p.previousm.copy()

    # ══════════════════════════════════════════════════════════════════════════
    # ВЫВОД
    # ══════════════════════════════════════════════════════════════════════════

    def _average_moments(self):
        avg = np.zeros(3)
        total_vol = 0.
        for p in self.particles:
            v = p.volume()
            total_vol += v
            avg += p.m * v
        return avg / total_vol if total_vol > 0 else avg

    def _write(self, forced=False):
        if self._outfile is None:
            return
        if forced or (self._twrite >= self._outputinterval > 0):
            avg = self._average_moments()
            self._outfile.write(
                f"{self.T:.6e}\t{avg[0]:.8f}\t{avg[1]:.8f}\t{avg[2]:.8f}"
                f"\t{self._n_blocked}\t{self._n_sp}\n"
            )
            self._outfile.flush()
            self._twrite = 0.

    # ══════════════════════════════════════════════════════════════════════════
    # ЗАПУСК
    # ══════════════════════════════════════════════════════════════════════════

    def prepare(self):
        assert self.Dt > 0,       "Dt должен быть > 0"
        assert self.Alpha >= 0,   "Alpha должна быть ≥ 0"
        assert self.Temp >= 0,    "Temp должна быть ≥ 0"
        assert len(self.particles) > 0, "Нет частиц"

        self._gammaoveralpha = GAMMA0 / (1. + self.Alpha**2)

        # Авто-классификация
        if any(p.ptype == ParticleType.AUTO for p in self.particles):
            if self.Ku1 > 0 and self.Temp > 0:
                self.classify_particles()
            else:
                for p in self.particles:
                    if p.ptype == ParticleType.AUTO:
                        p.ptype = ParticleType.BLOCKED
                self._n_blocked = sum(1 for p in self.particles if p.ptype == ParticleType.BLOCKED)
                self._n_sp      = sum(1 for p in self.particles if p.ptype == ParticleType.SP)

        if self.Brown:
            for p in self._blocked():
                p.calc_temp_prefactor(self.Alpha, self.Temp)

        if self.Jumpnoise:
            self._init_switch_times()

    def run(self, duration):
        self.prepare()
        t_end = self.T + duration
        if self.Demag:
            self._calculate_demag()
        self._write(forced=True)

        while self.T + 1e-18 <= t_end:
            if self.Demag:
                self._calculate_demag()

            if self.solver == "euler":
                self._euler_step(); self.T += self.Dt
            elif self.solver == "heun":
                self._heun_step();  self.T += self.Dt
            elif self.solver == "rk4":
                self._rk4_step();   self.T += self.Dt
            elif self.solver == "dopri":
                if self.Adaptivestep and self.T + self.Dt > t_end + 1e-20:
                    self.Dt = max(t_end - self.T, self.Mindt)
                self._dopri_step(); self.T += self.Dt
                if self.Adaptivestep:
                    if self._maxtauwitht > self.Errortolerance:
                        self._undo_bad_step(); self.T -= self.Dt
                    new_dt = 0.95*self.Dt*(self.Errortolerance/max(self._maxtauwitht,1e-30))**(1./self.order)
                    self.Dt = max(self.Mindt, min(self.Maxdt, new_dt))
                    if not self._relax:
                        self._maxtauwitht = 1e-12
            else:
                raise ValueError(f"Неизвестный solver: {self.solver}")

            if self.Jumpnoise:
                self._check_switches()

            self._twrite += self.Dt
            self._write()

        if self._outfile:
            self._outfile.close()

    # ══════════════════════════════════════════════════════════════════════════
    # ЭНЕРГИИ И УТИЛИТЫ
    # ══════════════════════════════════════════════════════════════════════════

    def _E_zeeman_si(self):
        return sum(-p.msat*p.volume()*float(np.dot(p.m, self._zeeman(p))) for p in self.particles)

    def _E_anis_si(self):
        return sum(-0.5*p.msat*p.volume()*float(np.dot(p.m, self._anis(p))) for p in self.particles)

    def _E_demag_si(self):
        return sum(-0.5*p.msat*p.volume()*float(np.dot(p.m, p.demagnetising_field)) for p in self.particles)

    def E_zeeman(self):
        """Энергия Зеемана в текущей системе единиц (Дж или эрг)."""
        return self.units.energy_from_si(self._E_zeeman_si())

    def E_anis(self):
        """Энергия анизотропии в текущей системе единиц."""
        return self.units.energy_from_si(self._E_anis_si())

    def E_demag(self):
        """Демагнетизирующая энергия в текущей системе единиц."""
        return self.units.energy_from_si(self._E_demag_si())

    def E_total(self):
        """Полная энергия в текущей системе единиц (Дж или эрг)."""
        return self.units.energy_from_si(
            self._E_zeeman_si() + self._E_anis_si() + self._E_demag_si()
        )

    def give_mz(self):
        return float(self._average_moments()[2])

    def give_m(self):
        return self._average_moments()

    def size_histogram(self, bins=20):
        radii = np.array([p.r*1e9 for p in self.particles])
        counts, edges = np.histogram(radii, bins=bins)
        return counts, edges

    def __repr__(self):
        return (f"Simulation(N={len(self.particles)}, "
                f"Blocked={self._n_blocked}, SP={self._n_sp}, "
                f"T={self.T:.2e}s, solver={self.solver})")


# ═══════════════════════════════════════════════════════════════════════════════
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def example_single_particle():
    """Одна частица, прецессия в поле B=0.1 Тл."""
    print("=" * 55)
    print("Пример 1: одна частица, прецессия")
    print("=" * 55)
    sim = Simulation()
    sim.world(0, 0, 0, 10)    # нм
    sim.particle_radius(10)   # нм
    sim.add_particle(0., 0., 0., ptype=ParticleType.BLOCKED)
    sim.B_ext  = lambda t: (0., 0., 0.1)
    sim.Demag  = False
    sim.msat(860e3)
    sim.Dt     = 5e-13
    sim.T      = 0.
    sim.Temp   = 0.
    sim.Alpha  = 0.02
    sim.Ku1    = 0.
    sim.Brown  = False
    sim.anisotropy_axis(0, 1, 0)
    sim.m_uniform(1, 0, 0)
    sim.output(5e-12, "example1_out.txt")
    sim.set_solver("rk4")
    print(f"  γ·B / 2π = {0.1 * GAMMA0 / (2*math.pi) / 1e9:.2f} ГГц")
    sim.run(0.5e-9)
    print(f"  <mz> = {sim.give_mz():.4f}")
    print("  → example1_out.txt")
    return sim


def example_mixed_ensemble():
    """
    Смешанный ансамбль Fe₃O₄-частиц с логнормальным распределением.

    Малые частицы → SP  (прыжки Нееля)
    Крупные       → Blocked (LLG + тепловой шум Брауна)
    """
    print("=" * 55)
    print("Пример 2: смешанный ансамбль Blocked + SP")
    print("=" * 55)

    sim = Simulation()
    sim.set_random_seed(2024)
    sim.world(0, 0, 0, 600)      # нм

    # Параметры Fe₃O₄
    sim.Alpha  = 0.1
    sim.Temp   = 300.             # K
    sim.Ku1    = 1.1e4            # Дж/м³
    sim.Tau0   = 1e-9             # с
    sim.Dt     = 5e-12            # 5 пс
    sim.T      = 0.
    sim.Brown  = True
    sim.Jumpnoise = True
    sim.Demag  = True

    # Логнормальное распределение: r_median=8 нм, σ=1.4
    # → захватывает и SP (<r_crit), и Blocked (>r_crit) частицы
    r_crit = blocking_radius(sim.Ku1, sim.Temp, tau0=sim.Tau0)
    print(f"\n  r_crit = {sim.units.radius_from_si(r_crit):.2f} нм  (граница Blocked/SP)")

    particles = sim.add_particles_lognormal(
        n          = 40,
        r_median   = 8,       # нм
        sigma_geo  = 1.4,
        r_h_coating= 2,       # нм оболочка
        seed       = 42
    )

    sim.msat(480e3)
    sim.anisotropy_axis_random()   # изотропный ансамбль
    sim.m_random()

    sim.B_ext = lambda t: (0., 0., 5e-3)   # 5 мТл вдоль Z

    # Автоклассификация: σ > 3 → Blocked
    stats = sim.classify_particles(sigma_threshold=3.0)
    print(f"\n  Классификация (σ_threshold={stats['sigma_threshold']:.1f}):")
    sim.print_ensemble_info()

    sim.set_solver("rk4")
    sim.output(50e-12, "mixed_ensemble_out.txt")

    print(f"\n  Запуск 1 нс...")
    sim.run(1e-9)

    avg = sim.give_m()
    print(f"\n  Результаты:")
    print(f"  <mx> = {avg[0]:.4f}")
    print(f"  <my> = {avg[1]:.4f}")
    print(f"  <mz> = {avg[2]:.4f}  (поле вдоль Z)")
    print(f"  E_total = {sim.E_total():.3e} Дж")
    print("  → mixed_ensemble_out.txt")
    return sim


def example_size_distribution_info():
    """Аналитическая информация о распределении без симуляции."""
    print("=" * 55)
    print("Пример 3: анализ распределения размеров")
    print("=" * 55)

    dist = SizeDistribution(r_median=8e-9, sigma_geo=1.4)
    print(f"  {dist}")
    print(f"  Медиана:   {dist.r_median*1e9:.1f} нм")
    print(f"  Среднее:   {dist.mean_r*1e9:.1f} нм")
    print(f"  Мода:      {dist.mode_r*1e9:.1f} нм")

    Ku = 1.1e4
    print(f"\n  Критический радиус r_crit (Ku={Ku:.1e}):")
    for T in [200, 250, 300, 350, 400]:
        r_c  = blocking_radius(Ku, T)
        tau  = neel_time(Ku, r_c, T)
        print(f"    T={T} K → r_crit = {r_c*1e9:.2f} нм, τ_N = {tau:.0f} с")


def example_cgs():
    """
    Пример в единицах СГС.

    Параметры задаются в привычных для магнетизма единицах:
      - Msat:  эму/см³
      - Ku:    эрг/см³
      - B_ext: Эрстед (Oe)
      - r:     нм
      - E:     эрг
    """
    print("=" * 55)
    print("Пример 4: одна частица в единицах СГС")
    print("=" * 55)

    # ── создаём симулятор в СГС ──
    sim = Simulation(units="CGS")
    print(f"  Система единиц: {sim.units}")

    # ── мир и частица (координаты в нм) ──
    sim.world(0, 0, 0, 10)          # куб 10×10×10 нм
    sim.particle_radius(10)         # 10 нм

    sim.add_particle(0, 0, 0, ptype=ParticleType.BLOCKED)

    # ── физика ─────────────────────────────────────────────
    # Msat Fe3O4 ≈ 480 эму/см³  (в СГС)
    sim.msat(480)                   # эму/см³ → внутри: 480 000 А/м

    # Ku  ≈ 1.1e5 эрг/см³  (в СГС)
    sim.ku1 = 1.1e5                 # эрг/см³ → внутри: 1.1e4 Дж/м³

    # Внешнее поле 100 Oe вдоль Z  (в СГС)
    sim.B_ext = sim.units.wrap_B_ext(lambda t: (0., 0., 100.))

    # ── параметры решателя ──
    sim.Alpha  = 0.1
    sim.Temp   = 0.
    sim.Dt     = 1e-12             # с (одинаково в обоих системах)
    sim.T      = 0.
    sim.Demag  = False
    sim.Brown  = False

    sim.anisotropy_axis(0, 0, 1)
    sim.m_uniform(1, 0, 0)

    sim.output(5e-12, "cgs_example_out.txt")
    sim.set_solver("rk4")

    print(f"  Msat = 480 {sim.units.lbl_msat}  "
          f"→ внутри {sim.particles[0].msat:.0f} А/м")
    print(f"  Ku1  = 1.1e5 {sim.units.lbl_ku}  "
          f"→ внутри {sim.Ku1:.4e} Дж/м³")
    print(f"  B    = 100 {sim.units.lbl_field}"
          f"  → внутри {sim.units.field_to_si(100)*1e3:.1f} мТл")

    sim.run(0.5e-9)

    E_total = sim.E_total()
    print(f"\n  <mz> = {sim.give_mz():.4f}")
    print(f"  E_total = {E_total:.4e} {sim.units.lbl_energy}")
    print("  → cgs_example_out.txt")
    return sim


def example_cgs_mixed():
    """
    Смешанный ансамбль в СГС единицах.
    """
    print("=" * 55)
    print("Пример 5: смешанный ансамбль в единицах СГС")
    print("=" * 55)

    sim = Simulation(units="CGS")
    sim.set_random_seed(2024)

    # Куб 600×600×600 нм
    sim.world(0, 0, 0, 600)

    # Параметры Fe3O4 в СГС
    sim.Alpha     = 0.1
    sim.Temp      = 300.           # К (одинаково)
    sim.ku1       = 1.1e5          # эрг/см³
    sim.Tau0      = 1e-9           # с
    sim.Dt        = 5e-12          # с
    sim.T         = 0.
    sim.Brown     = True
    sim.Jumpnoise = True
    sim.Demag     = True

    # r_crit в нм (внутри всё в СИ)
    r_crit_si  = blocking_radius(sim.Ku1, sim.Temp, tau0=sim.Tau0)
    r_crit_nm  = sim.units.radius_from_si(r_crit_si)
    print(f"\n  r_crit = {r_crit_nm:.2f} нм  (граница Blocked/SP)")

    # Добавить 30 частиц: r_median=8 нм, σ=1.4, оболочка 2 нм
    sim.add_particles_lognormal(n=30, r_median=8, sigma_geo=1.4,
                                r_h_coating=2, seed=42)

    # Задать Msat в СГС
    sim.msat(480)                  # эму/см³

    sim.anisotropy_axis_random()
    sim.m_random()

    # Поле 50 Oe вдоль Z (в СГС)
    sim.B_ext = sim.units.wrap_B_ext(lambda t: (0., 0., 50.))

    stats = sim.classify_particles(sigma_threshold=3.0)
    print(f"\n  Классификация:")
    print(f"  Blocked: {stats['n_blocked']}, SP: {stats['n_sp']}")
    sim.print_ensemble_info()

    sim.set_solver("rk4")
    sim.output(50e-12, "cgs_mixed_out.txt")

    print(f"\n  Запуск 0.5 нс...")
    sim.run(0.5e-9)

    avg = sim.give_m()
    E   = sim.E_total()
    print(f"\n  <mx> = {avg[0]:.4f}")
    print(f"  <my> = {avg[1]:.4f}")
    print(f"  <mz> = {avg[2]:.4f}  (поле вдоль Z)")
    print(f"  E_total = {E:.4e} {sim.units.lbl_energy}")
    print("  → cgs_mixed_out.txt")
    return sim


if __name__ == "__main__":
    example_size_distribution_info()
    print()
    example_single_particle()
    print()
    example_mixed_ensemble()
    print()
    example_cgs()
    print()
    example_cgs_mixed()
