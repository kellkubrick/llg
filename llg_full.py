import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 5.3e7  # Gyromagnetic ratio (m/A·s)
alpha = 0.01    # Gilbert damping constant
Ms = 8.0e5      # Saturation magnetization (A/m)
dt = 1e-12      # Time step (s)
steps = 1000    # Number of time steps

# Initial magnetization
M = np.array([0.8, 0.6, 0.0]) * Ms

# Effective field function (example: external field only)
def effective_field(M):
    H_ext = np.array([0.0, 0.0, 1.0e5])  # External field (A/m)
    return H_ext

# LLG equation
def llg_equation(M, t):
    H_eff = effective_field(M)
    dMdt = -gamma * np.cross(M, H_eff) + (alpha / Ms) * np.cross(M, np.cross(M, H_eff))
    return dMdt

# Runge-Kutta 4th order
def runge_kutta(M, t, dt):
    k1 = dt * llg_equation(M, t)
    k2 = dt * llg_equation(M + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * llg_equation(M + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * llg_equation(M + k3, t + dt)
    M_new = M + (k1 + 2*k2 + 2*k3 + k4) / 6
    return M_new
M_values = []
time = []
# Time evolution
for step in range(steps):
    M = runge_kutta(M, step * dt, dt)
    M = Ms * M / np.linalg.norm(M)  # Normalize
    M_values.append(M)
    time.append(step)
    print(f"Step {step}: M = {M}")

plt.plot(time, M_values, label='m')
plt.xlabel('Time')
plt.ylabel('m')
plt.legend()
plt.show()