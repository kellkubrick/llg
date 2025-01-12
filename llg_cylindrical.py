import numpy as np
import matplotlib.pyplot as plt


# Задаем параметры
Is = 480
K = 135000
r = 5*1e-7
Hz = 100
C = 0.3
dN_N = 0.05
costheta = 0.707

alpha = 0.01
gamma = 5.3*1e7


coef = gamma/(1+alpha**2)
M0 = (Is * 4 * np.pi * r**3)/3
Havz = -8*np.pi*C*Is*dN_N*(1-1.5*costheta)/3

# Задаем начальные условия
theta = np.pi/3
phi = np.pi/3

# Задаем временной интервал и шаг
t = 0
dt = 1e-10
steps=1000

# Создаем списки для сохранения значений популяции и времени
time = []
theta_values = []
phi_values = []

for step in range(steps):
    # Подсчет инкрементов для x и y
    dtheta1 = dt * (-coef*alpha*(2 * K * M0 * np.sin(2*theta) + np.sin(theta)*(Hz+Havz)))

    dtheta2 = dt * (-coef*alpha*(2 * K * M0 * np.sin(2*(theta+dtheta1/2)) + np.sin(theta+dtheta1/2)*(Hz+Havz)))

    dtheta3 = dt * (-coef*alpha*(2 * K * M0 * np.sin(2*(theta+dtheta2/2)) + np.sin(theta+dtheta2/2)*(Hz+Havz)))

    dtheta4 = dt * (-coef*alpha*(2 * K * M0 * np.sin(2*(theta+dtheta3)) + np.sin(theta+dtheta3)*(Hz+Havz)))

    # Обновляем значения x и y
    theta += (dtheta1 + 2 * dtheta2 + 2 * dtheta3 + dtheta4) / 6

    # Сохраняем текущие значения
    time.append(t)
    theta_values.append(theta)
    # Обновляем время
    t += dt

# Рисуем график численности популяций
plt.plot(time, theta_values, label='theta')
plt.xlabel('Time')
plt.ylabel('Theta')
plt.legend()
plt.show()
print(theta_values)
print(theta)
