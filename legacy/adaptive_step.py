import numpy as np
import matplotlib.pyplot as plt

# Параметры
Is = 480
K = 135000
r = 5e-7
Hz = 0.5
C = 0.3
dN_N = 0.05
costheta = 0.707
h0 = 0
w = 0

alpha = 0.02
gamma = 5.3e7

coef = gamma / (1 + alpha ** 2)
M0 = (Is * 4 * np.pi * r ** 3) / 3
Havz = -8 * np.pi * C * Is * dN_N * (1 - 1.5 * costheta) / 3

# Начальные условия
mx = 0.6
my = 0.8
mz = 0

# Адаптивный шаг (RKF45)
tolerance = 1e-6
dt_min = 1e-10
dt_max = 1e-9
t = 0
t_final = 1e-8  # 1 мкс


# Функция правых частей уравнения ЛЛГ
def rhs(t, mx, my, mz):
    H_eff = 2 * K * M0 * mz + Hz + Havz + h0 * np.cos(w * t)
    dmxdth = -coef * (my + alpha * mx * mz) * H_eff
    dmydth = coef * (mx - alpha * mz * my) * H_eff
    dmzdth = coef * alpha * (mx ** 2 + my ** 2) * H_eff
    return np.array([dmxdth, dmydth, dmzdth])


# Списки для сохранения данных
time = []
mx_values = []
my_values = []
mz_values = []

dt = dt_max  # Начальный шаг
while t < t_final:
    # Сохраняем текущее состояние
    current_mx, current_my, current_mz = mx, my, mz

    # Вычисление коэффициентов RKF45
    k1 = rhs(t, mx, my, mz)
    k2 = rhs(t + dt / 4,
             mx + dt / 4 * k1[0],
             my + dt / 4 * k1[1],
             mz + dt / 4 * k1[2])

    k3 = rhs(t + 3 * dt / 8,
             mx + 3 * dt / 32 * k1[0] + 9 * dt / 32 * k2[0],
             my + 3 * dt / 32 * k1[1] + 9 * dt / 32 * k2[1],
             mz + 3 * dt / 32 * k1[2] + 9 * dt / 32 * k2[2])

    k4 = rhs(t + 12 * dt / 13,
             mx + 1932 * dt / 2197 * k1[0] - 7200 * dt / 2197 * k2[0] + 7296 * dt / 2197 * k3[0],
             my + 1932 * dt / 2197 * k1[1] - 7200 * dt / 2197 * k2[1] + 7296 * dt / 2197 * k3[1],
             mz + 1932 * dt / 2197 * k1[2] - 7200 * dt / 2197 * k2[2] + 7296 * dt / 2197 * k3[2])

    k5 = rhs(t + dt,
             mx + 439 * dt / 216 * k1[0] - 8 * k2[0] + 3680 * dt / 513 * k3[0] - 845 * dt / 4104 * k4[0],
             my + 439 * dt / 216 * k1[1] - 8 * k2[1] + 3680 * dt / 513 * k3[1] - 845 * dt / 4104 * k4[1],
             mz + 439 * dt / 216 * k1[2] - 8 * k2[2] + 3680 * dt / 513 * k3[2] - 845 * dt / 4104 * k4[2])

    k6 = rhs(t + dt / 2,
             mx - 8 * dt / 27 * k1[0] + 2 * k2[0] - 3544 * dt / 2565 * k3[0] + 1859 * dt / 4104 * k4[0] - 11 * dt / 40 *
             k5[0],
             my - 8 * dt / 27 * k1[1] + 2 * k2[1] - 3544 * dt / 2565 * k3[1] + 1859 * dt / 4104 * k4[1] - 11 * dt / 40 *
             k5[1],
             mz - 8 * dt / 27 * k1[2] + 2 * k2[2] - 3544 * dt / 2565 * k3[2] + 1859 * dt / 4104 * k4[2] - 11 * dt / 40 *
             k5[2])

    # Решения 4-го и 5-го порядков
    mx4 = mx + dt * (25 / 216 * k1[0] + 1408 / 2565 * k3[0] + 2197 / 4104 * k4[0] - 1 / 5 * k5[0])
    my4 = my + dt * (25 / 216 * k1[1] + 1408 / 2565 * k3[1] + 2197 / 4104 * k4[1] - 1 / 5 * k5[1])
    mz4 = mz + dt * (25 / 216 * k1[2] + 1408 / 2565 * k3[2] + 2197 / 4104 * k4[2] - 1 / 5 * k5[2])

    mx5 = mx + dt * (16 / 135 * k1[0] + 6656 / 12825 * k3[0] + 28561 / 56430 * k4[0] - 9 / 50 * k5[0] + 2 / 55 * k6[0])
    my5 = my + dt * (16 / 135 * k1[1] + 6656 / 12825 * k3[1] + 28561 / 56430 * k4[1] - 9 / 50 * k5[1] + 2 / 55 * k6[1])
    mz5 = mz + dt * (16 / 135 * k1[2] + 6656 / 12825 * k3[2] + 28561 / 56430 * k4[2] - 9 / 50 * k5[2] + 2 / 55 * k6[2])

    # Оценка ошибки
    error = np.sqrt((mx4 - mx5) ** 2 + (my4 - my5) ** 2 + (mz4 - mz5) ** 2) / dt

    if error > tolerance:
        dt = max(0.9 * dt * (tolerance / error) ** 0.25, dt_min)
    else:
        # Принимаем шаг
        mx, my, mz = mx4, my4, mz4
        t += dt

        # Нормировка вектора
        norm = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        if norm > 1e-9:
            mx /= norm
            my /= norm
            mz /= norm

        # Сохранение данных
        time.append(t)
        mx_values.append(mx)
        my_values.append(my)
        mz_values.append(mz)

        # Увеличиваем шаг
        dt = min(1.1 * dt * (tolerance / error) ** 0.25, dt_max)

# Визуализация
time = [1e9 * t for t in time]
plt.plot(time, mx_values, label='mx')
plt.plot(time, my_values, label='my')
plt.plot(time, mz_values, label='mz')
plt.xlabel('Time, ns')
plt.ylabel('m')
plt.legend()
plt.show()