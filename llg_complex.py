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
h0 = 0
w = 0

alpha = 0.02
#gamma = 5.3*1e7
gamma = 8.79*1e6


coef = -gamma/(1+alpha**2)
M0 = Is

Havz = -8*np.pi*C*Is*dN_N*(1-1.5*costheta)/3

# Задаем начальные условия
mx = np.sqrt(2)/2
my = 0
mz = np.sqrt(2)/2

# Задаем временной интервал и шаг
t = 0
dt = 1e-10
steps=300


# Создаем списки для сохранения значений популяции и времени
time = []
mx_values = []
my_values = []
mz_values = []



def h_anis(m):
    return 2*K*m/Is

def h_ext():
    return Hz

def h_dd():
    return -8*np.pi*C*Is*dN_N*(1-1.5*costheta)/3



def h_eff_x(mx, my, mz):
    return (h_anis(mx)/2) + (h_anis(mz)/2)
def h_eff_y(mx, my, mz):
    return 0*((h_anis(mx)/2) + (h_anis(my)/2))
def h_eff_z(mx,my,mz):
    return h_ext() + h_dd() + (h_anis(mx)/2) + (h_anis(mz)/2)


for step in range(steps):
    dmx1 = dt * coef*( (my + alpha*mx*mz)*h_eff_z(mx,my,mz) - (mz - alpha*my*mx)*h_eff_y(mx,my,mz) - alpha*(my**2 + mz**2)*h_eff_x(mx,my,mz) )
    dmy1 = dt * coef*( (mz + alpha*my*mx)*h_eff_x(mx,my,mz) - (mx - alpha*mz*my)*h_eff_z(mx,my,mz) - alpha*(mz**2 + mx**2)*h_eff_y(mx,my,mz) )
    dmz1 = dt * coef*( (mx + alpha*mz*my)*h_eff_y(mx,my,mz) - (my - alpha*mx*mz)*h_eff_x(mx,my,mz) - alpha*(mx**2 + my**2)*h_eff_z(mx,my,mz) )

    dmx2 = dt * coef*( ((my+dmy1/2) + alpha*(mx+dmx1/2)*(mz+dmz1/2))*h_eff_z(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) - ((mz+dmz1/2) - alpha*(my+dmy1/2)*(mx+dmx1/2))*h_eff_y(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) - alpha*((my+dmy1/2)**2 + (mz+dmz1/2)**2)*h_eff_x(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) )
    dmy2 = dt * coef*( ((mz+dmz1/2) + alpha*(my+dmy1/2)*(mx+dmx1/2))*h_eff_x(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) - ((mx+dmx1/2) - alpha*(mz+dmz1/2)*(my+dmy1/2))*h_eff_z(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) - alpha*((mz+dmz1/2)**2 + (mx+dmx1/2)**2)*h_eff_y(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) )
    dmz2 = dt * coef*( ((mx+dmx1/2) + alpha*(mz+dmz1/2)*(my+dmy1/2))*h_eff_y(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) - ((my+dmy1/2) - alpha*(mx+dmx1/2)*(mz+dmz1/2))*h_eff_x(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) - alpha*((mx+dmx1/2)**2 + (my+dmy1/2)**2)*h_eff_z(mx+dmx1/2, my+dmy1/2, mz+dmz1/2) )

    dmx3 = dt * coef*( ((my+dmy2/2) + alpha*(mx+dmx2/2)*(mz+dmz2/2))*h_eff_z(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) - ((mz+dmz2/2) - alpha*(my+dmy2/2)*(mx+dmx2/2))*h_eff_y(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) - alpha*((my+dmy2/2)**2 + (mz+dmz2/2)**2)*h_eff_x(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) )
    dmy3 = dt * coef*( ((mz+dmz2/2) + alpha*(my+dmy2/2)*(mx+dmx2/2))*h_eff_x(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) - ((mx+dmx2/2) - alpha*(mz+dmz2/2)*(my+dmy2/2))*h_eff_z(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) - alpha*((mz+dmz2/2)**2 + (mx+dmx2/2)**2)*h_eff_y(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) )
    dmz3 = dt * coef*( ((mx+dmx2/2) + alpha*(mz+dmz2/2)*(my+dmy2/2))*h_eff_y(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) - ((my+dmy2/2) - alpha*(mx+dmx2/2)*(mz+dmz2/2))*h_eff_x(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) - alpha*((mx+dmx2/2)**2 + (my+dmy2/2)**2)*h_eff_z(mx+dmx2/2, my+dmy2/2, mz+dmz2/2) )

    dmx4 = dt * coef*( ((my+dmy3) + alpha*(mx+dmx3)*(mz+dmz3))*h_eff_z(mx+dmx3,my+dmy3,mz+dmz3) - ((mz+dmz3) - alpha*(my+dmy3)*(mx+dmx3))*h_eff_y(mx+dmx3,my+dmy3,mz+dmz3) - alpha*((my+dmy3)**2 + (mz+dmz3)**2)*h_eff_x(mx+dmx3,my+dmy3,mz+dmz3) )
    dmy4 = dt * coef*( ((mz+dmz3) + alpha*(my+dmy3)*(mx+dmx3))*h_eff_x(mx+dmx3,my+dmy3,mz+dmz3) - ((mx+dmx3) - alpha*(mz+dmz3)*(my+dmy3))*h_eff_z(mx+dmx3,my+dmy3,mz+dmz3) - alpha*((mz+dmz3)**2 + (mx+dmx3)**2)*h_eff_y(mx+dmx3,my+dmy3,mz+dmz3) )
    dmz4 = dt * coef*( ((mx+dmx3) + alpha*(mz+dmz3)*(my+dmy3))*h_eff_y(mx+dmx3,my+dmy3,mz+dmz3) - ((my+dmy3) - alpha*(mx+dmx3)*(mz+dmz3))*h_eff_x(mx+dmx3,my+dmy3,mz+dmz3) - alpha*((mx+dmx3)**2 + (my+dmy3)**2)*h_eff_z(mx+dmx3,my+dmy3,mz+dmz3) )

    # Обновляем значения x и y
    mx += (dmx1 + 2 * dmx2 + 2 * dmx3 + dmx4) / 6
    my += (dmy1 + 2 * dmy2 + 2 * dmy3 + dmy4) / 6
    mz += (dmz1 + 2 * dmz2 + 2 * dmz3 + dmz4) / 6

    norm = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    if norm > 1e-9:  # Избегаем деления на ноль
        mx /= norm
        my /= norm
        mz /= norm

    # Сохраняем текущие значения
    time.append(t)
    mx_values.append(mx)
    my_values.append(my)
    mz_values.append(mz)
    # Обновляем время
    t += dt

time=[1e9*t for t in time]
# Рисуем график численности популяций
plt.plot(time, mx_values, label='mx')
plt.plot(time, my_values, label='my')
plt.plot(time, mz_values, label='mz')
plt.xlabel('Time, ns')
plt.ylabel('m')
plt.legend()
plt.show()
