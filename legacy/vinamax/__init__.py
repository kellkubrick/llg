"""
vinamax — Python-порт симулятора магнитных наночастиц vinamax
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

from .constants import GAMMA0, MU0, KB, GAMMA0_CGS, KB_CGS
from .units import Units
from .particle import (
    ParticleType,
    Particle,
    vnorm,
    cube,
    sphere_volume,
    neel_time,
    blocking_radius,
)
from .distribution import SizeDistribution
from .simulation import Simulation

__all__ = [
    # Константы
    "GAMMA0",
    "MU0",
    "KB",
    "GAMMA0_CGS",
    "KB_CGS",
    # Единицы
    "Units",
    # Частицы и утилиты
    "ParticleType",
    "Particle",
    "vnorm",
    "cube",
    "sphere_volume",
    "neel_time",
    "blocking_radius",
    # Распределение
    "SizeDistribution",
    # Симулятор
    "Simulation",
]
