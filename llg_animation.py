import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
Is = 480
K = 135000
r = 5*1e-7
Hz = 0
C = 0.3
dN_N = 0.05
costheta = 0.707
h0 = 20
w = 1e2

alpha = 0.02
gamma = 5.3*1e7


coef = gamma/(1+alpha**2)
M0 = (Is * 4 * np.pi * r**3)/3
Havz = -8*np.pi*C*Is*dN_N*(1-1.5*costheta)/3

# Задаем начальные условия
mx = 0.6
my = 0.8
mz = 0

# Задаем временной интервал и шаг
t = 0
dt = 1e-10
steps=1000

# Создаем списки для сохранения значений популяции и времени
time = []
mx_values = []
my_values = []
mz_values = []

for step in range(steps):
    # Подсчет инкрементов для x и y
    dmx1 = dt * (-coef*(my+alpha*mx*mz)*(2*K*M0*mz + Hz + Havz + h0*np.cos(w*t)))
    dmy1 = dt * (coef*(mx-alpha*mz*my)*(2*K*M0*mz + Hz + Havz + h0*np.cos(w*t)))
    dmz1 = dt * (coef * (alpha * (mx**2 + my**2)) * (2 * K * M0 * mz + Hz + Havz + h0*np.cos(w*t)))

    dmx2 = dt * (-coef*((my+dmy1/2)+alpha*(mx+dmx1/2)*(mz+dmz1/2))*(2*K*M0*(mz+dmz1/2) + Hz + Havz + h0*np.cos(w*(t+dt/2))))
    dmy2 = dt * (coef*((mx+dmx1/2)-alpha*(mz+dmz1/2)*(my+dmy1/2))*(2*K*M0*(mz+dmz1/2) + Hz + Havz + h0*np.cos(w*(t+dt/2))))
    dmz2 = dt * (coef * (alpha * ((mx+dmx1/2)**2 + (my+dmy1/2)**2)) * (2 * K * M0 * (mz+dmz1/2) + Hz + Havz + h0*np.cos(w*(t+dt/2))))

    dmx3 = dt * (-coef*((my+dmy2/2)+alpha*(mx+dmx2/2)*(mz+dmz2/2))*(2*K*M0*(mz+dmz2/2) + Hz + Havz + h0*np.cos(w*(t+dt/2))))
    dmy3 = dt * (coef*((mx+dmx2/2)-alpha*(mz+dmz2/2)*(my+dmy2/2))*(2*K*M0*(mz+dmz2/2) + Hz + Havz + h0*np.cos(w*(t+dt/2))))
    dmz3 = dt * (coef * (alpha * ((mx+dmx2/2)**2 + (my+dmy2/2)**2)) * (2 * K * M0 * (mz+dmz2/2) + Hz + Havz + h0*np.cos(w*(t+dt/2))))

    dmx4 = dt * (-coef*((my+dmy3)+alpha*(mx+dmx3)*(mz+dmz3))*(2*K*M0*(mz+dmz3) + Hz + Havz + h0*np.cos(w*(t+dt))))
    dmy4 = dt * (coef*((mx+dmx3)-alpha*(mz+dmz3)*(my+dmy3))*(2*K*M0*(mz+dmz3) + Hz + Havz + h0*np.cos(w*(t+dt))))
    dmz4 = dt * (coef * (alpha * ((mx+dmx3)**2 + (my+dmy3)**2)) * (2 * K * M0 * (mz+dmz3) + Hz + Havz + h0*np.cos(w*(t+dt))))


    # Обновляем значения x и y
    mx += (dmx1 + 2 * dmx2 + 2 * dmx3 + dmx4) / 6
    my += (dmy1 + 2 * dmy2 + 2 * dmy3 + dmy4) / 6
    mz += (dmz1 + 2 * dmz2 + 2 * dmz3 + dmz4) / 6

    # Сохраняем текущие значения
    time.append(t)
    mx_values.append(mx)
    my_values.append(my)
    mz_values.append(mz)
    # Обновляем время
    t += dt
# Ваши данные
# Предположим, что у вас есть массивы x, y, z и time
x = np.array(mx_values)
y = np.array(my_values)
z = np.array(mz_values)
time = np.array(time)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Инициализация вектора и стрелки
line, = ax.plot([0, x[0]], [0, y[0]], [0, z[0]], color='r', lw=2)
arrow = ax.quiver(0, 0, 0, x[0], y[0], z[0], color='b', length=1, normalize=True)

# Для следа будем сохранять предыдущие точки
trace, = ax.plot([], [], [], color='g', lw=1, alpha=0.5)
trace_x, trace_y, trace_z = [], [], []

# Установка пределов осей
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

# Функция инициализации анимации
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    trace.set_data([], [])
    trace.set_3d_properties([])
    return line, trace, arrow

# Функция анимации
def animate(i):
    # Обновляем вектор
    line.set_data([0, x[i]], [0, y[i]])
    line.set_3d_properties([0, z[i]])

    # Обновляем стрелку
    #arrow.remove()  # Удаляем старую стрелку
    arrow = ax.quiver(0, 0, 0, x[i], y[i], z[i], color='b', length=1, normalize=True)

    # Добавляем текущую точку к следу
    trace_x.append(x[i])
    trace_y.append(y[i])
    trace_z.append(z[i])
    trace.set_data(trace_x, trace_y)
    trace.set_3d_properties(trace_z)

    return line, trace, arrow

# Создание анимации
ani = animation.FuncAnimation(fig, animate, frames=len(time), init_func=init, blit=True, interval=10)

# Показать анимацию
plt.show()