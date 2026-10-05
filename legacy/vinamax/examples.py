"""
Примеры использования vinamax.
"""

import math

from .constants import GAMMA0
from .simulation import Simulation
from .particle import ParticleType, blocking_radius, neel_time
from .distribution import SizeDistribution


def example_single_particle():
    """Одна частица, прецессия в поле B=0.1 Тл."""
    print("=" * 55)
    print("Пример 1: одна частица, прецессия")
    print("=" * 55)
    sim = Simulation()
    sim.world(0, 0, 0, 10)  # нм
    sim.particle_radius(10)  # нм
    sim.add_particle(0.0, 0.0, 0.0, ptype=ParticleType.BLOCKED)
    sim.B_ext = lambda t: (0.0, 0.0, 0.1)
    sim.Demag = False
    sim.msat(860e3)
    sim.Dt = 5e-13
    sim.T = 0.0
    sim.Temp = 0.0
    sim.Alpha = 0.02
    sim.Ku1 = 0.0
    sim.Brown = False
    sim.anisotropy_axis(0, 1, 0)
    sim.m_uniform(1, 0, 0)
    sim.output(5e-12, "example1_out.txt")
    sim.set_solver("rk4")
    print(f"  γ·B / 2π = {0.1 * GAMMA0 / (2 * math.pi) / 1e9:.2f} ГГц")
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
    sim.world(0, 0, 0, 600)  # нм

    # Параметры Fe₃O₄
    sim.Alpha = 0.1
    sim.Temp = 300.0  # K
    sim.Ku1 = 1.1e4  # Дж/м³
    sim.Tau0 = 1e-9  # с
    sim.Dt = 5e-12  # 5 пс
    sim.T = 0.0
    sim.Brown = True
    sim.Jumpnoise = True
    sim.Demag = True

    # Логнормальное распределение: r_median=8 нм, σ=1.4
    # → захватывает и SP (<r_crit), и Blocked (>r_crit) частицы
    r_crit = blocking_radius(sim.Ku1, sim.Temp, tau0=sim.Tau0)
    print(
        f"\n  r_crit = {sim.units.radius_from_si(r_crit):.2f} нм  (граница Blocked/SP)"
    )

    particles = sim.add_particles_lognormal(
        n=40,
        r_median=8,  # нм
        sigma_geo=1.4,
        r_h_coating=2,  # нм оболочка
        seed=42,
    )

    sim.msat(480e3)
    sim.anisotropy_axis_random()  # изотропный ансамбль
    sim.m_random()

    sim.B_ext = lambda t: (0.0, 0.0, 5e-3)  # 5 мТл вдоль Z

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
    print(f"  Медиана:   {dist.r_median * 1e9:.1f} нм")
    print(f"  Среднее:   {dist.mean_r * 1e9:.1f} нм")
    print(f"  Мода:      {dist.mode_r * 1e9:.1f} нм")

    Ku = 1.1e4
    print(f"\n  Критический радиус r_crit (Ku={Ku:.1e}):")
    for T in [200, 250, 300, 350, 400]:
        r_c = blocking_radius(Ku, T)
        tau = neel_time(Ku, r_c, T)
        print(f"    T={T} K → r_crit = {r_c * 1e9:.2f} нм, τ_N = {tau:.0f} с")


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
    sim.world(0, 0, 0, 10)  # куб 10×10×10 нм
    sim.particle_radius(10)  # 10 нм

    sim.add_particle(0, 0, 0, ptype=ParticleType.BLOCKED)

    # ── физика ─────────────────────────────────────────────
    # Msat Fe3O4 ≈ 480 эму/см³  (в СГС)
    sim.msat(480)  # эму/см³ → внутри: 480 000 А/м

    # Ku  ≈ 1.1e5 эрг/см³  (в СГС)
    sim.ku1 = 1.1e5  # эрг/см³ → внутри: 1.1e4 Дж/м³

    # Внешнее поле 100 Oe вдоль Z  (в СГС)
    sim.B_ext = sim.units.wrap_B_ext(lambda t: (0.0, 0.0, 100.0))

    # ── параметры решателя ──
    sim.Alpha = 0.1
    sim.Temp = 0.0
    sim.Dt = 1e-12  # с (одинаково в обоих системах)
    sim.T = 0.0
    sim.Demag = False
    sim.Brown = False

    sim.anisotropy_axis(0, 0, 1)
    sim.m_uniform(1, 0, 0)

    sim.output(5e-12, "cgs_example_out.txt")
    sim.set_solver("rk4")

    print(
        f"  Msat = 480 {sim.units.lbl_msat}  → внутри {sim.particles[0].msat:.0f} А/м"
    )
    print(f"  Ku1  = 1.1e5 {sim.units.lbl_ku}  → внутри {sim.Ku1:.4e} Дж/м³")
    print(
        f"  B    = 100 {sim.units.lbl_field}"
        f"  → внутри {sim.units.field_to_si(100) * 1e3:.1f} мТл"
    )

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
    sim.Alpha = 0.1
    sim.Temp = 300.0  # К (одинаково)
    sim.ku1 = 1.1e5  # эрг/см³
    sim.Tau0 = 1e-9  # с
    sim.Dt = 5e-12  # с
    sim.T = 0.0
    sim.Brown = True
    sim.Jumpnoise = True
    sim.Demag = True

    # r_crit в нм (внутри всё в СИ)
    r_crit_si = blocking_radius(sim.Ku1, sim.Temp, tau0=sim.Tau0)
    r_crit_nm = sim.units.radius_from_si(r_crit_si)
    print(f"\n  r_crit = {r_crit_nm:.2f} нм  (граница Blocked/SP)")

    # Добавить 30 частиц: r_median=8 нм, σ=1.4, оболочка 2 нм
    sim.add_particles_lognormal(n=30, r_median=8, sigma_geo=1.4, r_h_coating=2, seed=42)

    # Задать Msat в СГС
    sim.msat(480)  # эму/см³

    sim.anisotropy_axis_random()
    sim.m_random()

    # Поле 50 Oe вдоль Z (в СГС)
    sim.B_ext = sim.units.wrap_B_ext(lambda t: (0.0, 0.0, 50.0))

    stats = sim.classify_particles(sigma_threshold=3.0)
    print(f"\n  Классификация:")
    print(f"  Blocked: {stats['n_blocked']}, SP: {stats['n_sp']}")
    sim.print_ensemble_info()

    sim.set_solver("rk4")
    sim.output(50e-12, "cgs_mixed_out.txt")

    print(f"\n  Запуск 0.5 нс...")
    sim.run(0.5e-9)

    avg = sim.give_m()
    E = sim.E_total()
    print(f"\n  <mx> = {avg[0]:.4f}")
    print(f"  <my> = {avg[1]:.4f}")
    print(f"  <mz> = {avg[2]:.4f}  (поле вдоль Z)")
    print(f"  E_total = {E:.4e} {sim.units.lbl_energy}")
    print("  → cgs_mixed_out.txt")
    return sim


def main():
    example_size_distribution_info()
    print()
    example_single_particle()
    print()
    example_mixed_ensemble()
    print()
    example_cgs()
    print()
    example_cgs_mixed()
