from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# --- Параметры системы ---
# Гиромагнитное отношение, коэффициент демпфирования
gamma = SX.sym('gamma')
alpha = SX.sym('alpha')

# Параметры размагничивания и внешнее поле
Is = SX.sym('Is')
K = SX.sym('K')
dN_N = SX.sym('dN_N')
c = SX.sym('c')  # Константа для H_ez
costheta = SX.sym('costheta')
H0 = SX.sym('H0', 3)  # Внешнее поле [H0x, H0y, H0z]

# --- Вычисление эффективного поля H_e ---
def compute_H_e(m, H0):
    H_ex = 2*K*m[0]/Is
    H_ey = 0
    H_ez = H0[2] - 8*np.pi*c*Is*dN_N*(1-1.5*costheta)/3
    return vertcat(H_ex, H_ey, H_ez)

# --- Определение ОДУ ---
m = SX.sym('m', 3)
H_e = compute_H_e(m, H0)
H_ex, H_ey, H_ez = H_e[0], H_e[1], H_e[2]

denominator = 1 / (1 + alpha**2)

# Уравнение для dm_x/dt
term_x = (m[1] + alpha * m[0] * m[2]) * H_ez \
         - (m[2] - alpha * m[1] * m[0]) * H_ey \
         - alpha * (m[1]**2 + m[2]**2) * H_ex
dmx_dt = -gamma * denominator * term_x

# Уравнение для dm_y/dt
term_y = (m[2] + alpha * m[1] * m[0]) * H_ex \
         - (m[0] - alpha * m[2] * m[1]) * H_ez \
         - alpha * (m[2]**2 + m[0]**2) * H_ey
dmy_dt = -gamma * denominator * term_y

# Уравнение для dm_z/dt
term_z = (m[0] + alpha * m[2] * m[1]) * H_ey \
         - (m[1] - alpha * m[0] * m[2]) * H_ex \
         - alpha * (m[0]**2 + m[1]**2) * H_ez
dmz_dt = -gamma * denominator * term_z

# Функция правой части ОДУ
ode = vertcat(dmx_dt, dmy_dt, dmz_dt)
params = vertcat(gamma, alpha, Is, K, dN_N, costheta, c, H0)
f = Function('f', [m, params], [ode])

# --- Настройка неявного интегратора ---
tf = 1.0  # Временной интервал
n = 50     # Количество шагов
d = 4      # Степень полинома коллокации

# Коллокационные точки (Лежандр)
tau_root = [0] + collocation_points(d, 'legendre')

# Матрицы коэффициентов C и D
C = np.zeros((d+1, d+1))
D = np.zeros(d+1)
tau = SX.sym('tau')

for j in range(d+1):
    L = 1
    for r in range(d+1):
        if r != j:
            L *= (tau - tau_root[r]) / (tau_root[j] - tau_root[r])
    lfcn = Function('lfcn', [tau], [L])
    D[j] = lfcn(1.0)
    tfcn = Function('tfcn', [tau], [tangent(L, tau)])
    for r in range(d+1):
        C[j, r] = tfcn(tau_root[r])

# Символические переменные
X0 = MX.sym('X0', 3)
P = MX.sym('P', 10)  # Параметры: gamma, alpha, Nx, Ny, Nz, M0, H0
V = MX.sym('V', d*3)

# Состояния в коллокационных точках
X = [X0] + vertsplit(V, [r*3 for r in range(d+1)])

# Уравнения коллокации
V_eq = []
for j in range(1, d+1):
    xp_j = sum(C[r, j] * X[r] for r in range(d+1))
    # ИСПРАВЛЕНИЕ: Передача параметров в правильном порядке
    f_j = f(X[j], P)
    V_eq.append((tf/n)*f_j - xp_j)

V_eq = vertcat(*V_eq)
vfcn = Function('vfcn', [V, X0, P], [V_eq]).expand()
ifcn = rootfinder('ifcn', 'newton', vfcn)
V_sol = ifcn(MX(), X0, P)

# Сборка конечного состояния
X = [X0 if r == 0 else V_sol[(r-1)*3:r*3] for r in range(d+1)]
XF = sum(D[r] * X[r] for r in range(d+1))
F = Function('F', [X0, P], [XF])

# Итоговый интегратор с сохранением истории состояний
X_total = X0
states = [X_total.full().flatten()]  # Сохраняем как плоский массив

for _ in range(n):
    X_total = F(X_total, P)
    states.append(X_total.full().flatten())  # Преобразуем в numpy и "разглаживаем"

# Преобразование списка в двумерный массив
all_states = np.array(states)

# Создание функции с выходом всех состояний
irk_integrator = Function(
    'irk_integrator',
    [X0, P],
    [all_states],
    ['x0', 'p'],
    ['all_states']
)
# --- Тестирование ---
# Начальные условия (m_x, m_y, m_z)
m0 = np.array([1.0, 0.0, 0.0])

# Параметры:
# [gamma, alpha, Nx, Ny, Nz, M0, H0]
params = np.array([
5.3*1e7, # gamma
0.01, # alpha
480.0, # Is
135000, # K
0.05, # dN_N
0.707, # costheta
0.3, #c
0.0,
0.0,
0.0# c
#[0.0, 0.0, 0.5] # H0
])

# --- Тестирование ---
res = irk_integrator(x0=m0, p=params)
all_states = res['all_states'].full()  # Преобразование в numpy-массив

# Временные метки
time = np.linspace(0, tf, n+1)
mx = all_states[:, 0]
my = all_states[:, 1]  # Теперь индекс 1 допустим
mz = all_states[:, 2]

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(time, mx, label='m_x')
plt.plot(time, my, label='m_y')
plt.plot(time, mz, label='m_z')
plt.xlabel('Время (с)')
plt.ylabel('Намагниченность')
plt.title('Динамика вектора намагниченности')
plt.legend()
plt.grid(True)
plt.show()