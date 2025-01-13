import numpy as np
import matplotlib.pyplot as plt

# Parameters
Is = 480
K = 135000
r = 5*1e-7
H = 50
C = 0.3
dN_N = 0.05
costheta = 0.707
h0 = 20
w = 1e3

alpha = 0.02
gamma = 5.3*1e7


coef = gamma/(1+alpha**2)
Ms = (Is * 4 * np.pi * r**3)/3
Havz = -8*np.pi*C*Is*dN_N*(1-1.5*costheta)/3

# Initial magnetization
M = np.array([0.8, 0.0, 0.6]) * Ms

# Effective field function (example: external field only)
def effective_field(M,t):
    Hz = 2*K*M[2] + H + Havz + h0*np.cos(w*t)
    H_ext = np.array([0.0, 0.0, Hz])  # External field (A/m)
    return H_ext

# LLG equation
def llg_equation(M, t):
    H_eff = effective_field(M,t)
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

t = 0
dt = 1e-11
steps=1000
M_values = []
time = []
# Time evolution
for step in range(steps):
    M = runge_kutta(M, t, dt)
    # M = Ms * M / np.linalg.norm(M)  # Normalize
    M_values.append(M)
    time.append(t)
    print(f"Step {step}: M = {M}")
    t += dt

plt.plot(time, M_values, label='m')
plt.xlabel('Time')
plt.ylabel('m')
plt.legend()
plt.show()