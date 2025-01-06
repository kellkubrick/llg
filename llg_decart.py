import numpy as np
import matplotlib.pyplot as plt


# Задаем параметры
Is = 480
K = 135000
r = 5*1e-7
Hz = 0.5
C = 0.3
dN_N = 0.05
costheta = 0.707

alpha = 0.01
gamma = -5.3*1e7


coef = gamma/(1+alpha**2)
M0 = (Is * 4 * np.pi * r**3)/3
Havz = -8*np.pi*C*Is*dN_N*(1-1.5*costheta)/3

# Задаем начальные условия
mx = 0.6
my = 0
mz = 0.8

# Задаем временной интервал и шаг
t = 0
dt = 1e-12
steps=1000

# Создаем списки для сохранения значений популяции и времени
time = []
mx_values = []
my_values = []
mz_values = []

for step in range(steps):
    # Подсчет инкрементов для x и y
    dmx1 = dt * (-coef*(my+alpha*mx*mz)*(2*K*M0*mz + Hz + Havz))
    dmy1 = dt * (coef*(mx-alpha*mz*my)*(2*K*M0*mz + Hz + Havz))
    dmz1 = dt * (coef * (alpha * (mx**2 + my**2)) * (2 * K * M0 * mz + Hz + Havz))

    dmx2 = dt * (-coef*((my+dmy1/2)+alpha*(mx+dmx1/2)*(mz+dmz1/2))*(2*K*M0*(mz+dmz1/2) + Hz + Havz))
    dmy2 = dt * (coef*((mx+dmx1/2)-alpha*(mz+dmz1/2)*(my+dmy1/2))*(2*K*M0*(mz+dmz1/2) + Hz + Havz))
    dmz2 = dt * (coef * (alpha * ((mx+dmx1/2)**2 + (my+dmy1/2)**2)) * (2 * K * M0 * (mz+dmz1/2) + Hz + Havz))

    dmx3 = dt * (-coef*((my+dmy2/2)+alpha*(mx+dmx2/2)*(mz+dmz2/2))*(2*K*M0*(mz+dmz2/2) + Hz + Havz))
    dmy3 = dt * (coef*((mx+dmx2/2)-alpha*(mz+dmz2/2)*(my+dmy2/2))*(2*K*M0*(mz+dmz2/2) + Hz + Havz))
    dmz3 = dt * (coef * (alpha * ((mx+dmx2/2)**2 + (my+dmy2/2)**2)) * (2 * K * M0 * (mz+dmz2/2) + Hz + Havz))

    dmx4 = dt * (-coef*((my+dmy3)+alpha*(mx+dmx3)*(mz+dmz3))*(2*K*M0*(mz+dmz3) + Hz + Havz))
    dmy4 = dt * (coef*((mx+dmx3)-alpha*(mz+dmz3)*(my+dmy3))*(2*K*M0*(mz+dmz3) + Hz + Havz))
    dmz4 = dt * (coef * (alpha * ((mx+dmx3)**2 + (my+dmy3)**2)) * (2 * K * M0 * (mz+dmz3) + Hz + Havz))


    # Обновляем значения x и y
    mx += (dmx1 + 2 * dmx2 + 2 * dmx3 + dmx4) / 6
    my += (dmy1 + 2 * dmy2 + 2 * dmy3 + dmy4) / 6
    my += (dmz1 + 2 * dmz2 + 2 * dmz3 + dmz4) / 6

    # Сохраняем текущие значения
    time.append(t)
    mx_values.append(mx)
    my_values.append(my)
    mz_values.append(mz)
    # Обновляем время
    t += dt

# Рисуем график численности популяций
plt.plot(time, mx_values, label='mx')
plt.plot(time, my_values, label='my')
plt.plot(time, mz_values, label='mz')
plt.xlabel('Time')
plt.ylabel('m')
plt.legend()
plt.show()
print(mx_values)