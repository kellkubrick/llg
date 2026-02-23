# ============================================================
# SCRIPT 1: sLLG (Heun) + self-consistent random-field (Strategy A)
# Constant external field along +Z, CGS units
# Output: time traces of <m_x>, <m_y>, <m_z>, mixture params p(t), s(t)
# ============================================================

import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random, lax

print("JAX devices:", jax.devices())


# -------------------- CONFIG --------------------
class Config:
    # Ensemble
    N_PARTICLES = 1
    CONCENTRATION = 0.01  # volume fraction c (dilute)

    # Material (CGS)
    MS = 480.0  # emu/cm^3 ~ G
    K_ANIS = 1.35e5  # erg/cm^3
    TEMP = 0.0  # K
    ALPHA = 0.1
    GAMMA = 1.76e7  # s^-1 Oe^-1

    # Size distribution (lognormal diameter)
    D_MEAN = 10.0e-7  # cm (10 nm)
    D_LOGSIGMA = 0.25

    # Integration
    DT = 1.0e-12  # s
    N_STEPS = 50_000

    # External field
    H_EXT_VAL = 50.0  # Oe along +z

    # Sample geometry (cylinder), Table-1 approximation 2R>d
    CYL_HEIGHT = 2.0e-4  # cm
    CYL_RADIUS = 5.0e-4  # cm


cfg = Config()

# -------------------- CONSTANTS --------------------
kB_erg = 1.380649e-16  # erg/K


# -------------------- HELPERS --------------------
def normalize(m):
    return m / jnp.linalg.norm(m, axis=1, keepdims=True)


def random_unit_vectors(key, n):
    k1, k2 = random.split(key)
    u = random.uniform(k1, (n,), minval=0.0, maxval=1.0)
    v = random.uniform(k2, (n,), minval=0.0, maxval=1.0)
    theta = jnp.arccos(1.0 - 2.0 * u)
    phi = 2.0 * jnp.pi * v
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack([x, y, z], axis=1)


def lognormal_diameters(key, n, d_mean, log_sigma):
    mu = jnp.log(d_mean) - 0.5 * (log_sigma**2)
    return jnp.exp(mu + log_sigma * random.normal(key, (n,)))


# -------------------- KHARITONSKII MOMENTS (Oz, cylinder, 2R>d) --------------------
def cylinder_cos_theta_max(d, R):
    return d / jnp.sqrt(d * d + (2.0 * R) * (2.0 * R))


def kharitonskii_moments_Oz(c, Is, d0, d, R, zeta):
    cos_th = cylinder_cos_theta_max(d, R)
    A = 4.0 * jnp.pi / 3.0

    mean_H = -(8.0 * jnp.pi / 3.0) * c * Is * (1.0 - 1.5 * cos_th) * zeta

    ratio = d0 / d
    var_H = (A**2) * c * (Is**2) * (1.0 / 10.0) * (1.0 - (15.0 / 4.0) * (ratio**3))
    var_H = jnp.maximum(var_H, 1e-30)

    mu3 = (A**3) * c * (Is**3) * (1.0 / 280.0) * (1.0 - 67.0 * (ratio**6)) * zeta
    mu4 = (A**4) * c * (Is**4) * (1.0 / 1120.0) * (1.0 - 264.0 * (ratio**9))

    return mean_H, var_H, mu3, mu4


# -------------------- STRATEGY A: 2-GAUSSIAN MIXTURE MATCHING --------------------
def cbrt(x):
    return jnp.sign(x) * (jnp.abs(x) + 1e-30) ** (1.0 / 3.0)


def mixture_params_from_moments(mean, var, mu3, mu4, n_bisect=40):
    skew_scale = jnp.abs(mu3) / (var**1.5 + 1e-30)

    def gaussian_fallback():
        return 0.5, mean, mean, jnp.sqrt(var)

    def mu4_pred_for_p(p):
        q = 1.0 - p
        denom = p * (q - p)  # p*(1-2p)
        denom = jnp.where(jnp.abs(denom) < 1e-12, jnp.sign(denom) * 1e-12, denom)

        delta1 = cbrt(mu3 * (q * q) / denom)
        delta2 = -(p / q) * delta1

        A = (p / q) * (delta1**2)
        s2 = var - A
        s2 = jnp.where(s2 > 0.0, s2, -1.0)

        mu4_mix = p * (delta1**4) + q * (delta2**4) + 6.0 * s2 * A + 3.0 * (s2**2)
        mu4_mix = jnp.where(s2 > 0.0, mu4_mix, 1e30)
        return mu4_mix

    def fit():
        p_lo, p_hi = 1e-3, 0.499
        f_lo = mu4_pred_for_p(p_lo) - mu4
        f_hi = mu4_pred_for_p(p_hi) - mu4
        no_bracket = (f_lo * f_hi > 0.0) | (~jnp.isfinite(f_lo)) | (~jnp.isfinite(f_hi))

        def do_bisect():
            def body_fn(state):
                p_a, p_b, f_a, f_b = state
                p_m = 0.5 * (p_a + p_b)
                f_m = mu4_pred_for_p(p_m) - mu4
                same = f_a * f_m > 0.0
                p_a2 = jnp.where(same, p_m, p_a)
                f_a2 = jnp.where(same, f_m, f_a)
                p_b2 = jnp.where(same, p_b, p_m)
                f_b2 = jnp.where(same, f_b, f_m)
                return (p_a2, p_b2, f_a2, f_b2)

            state = (p_lo, p_hi, f_lo, f_hi)
            state = lax.fori_loop(0, n_bisect, lambda i, st: body_fn(st), state)
            p_star = 0.5 * (state[0] + state[1])

            p = p_star
            q = 1.0 - p
            denom = p * (q - p)
            denom = jnp.where(jnp.abs(denom) < 1e-12, jnp.sign(denom) * 1e-12, denom)

            delta1 = cbrt(mu3 * (q * q) / denom)
            delta2 = -(p / q) * delta1
            A = (p / q) * (delta1**2)
            s2 = jnp.maximum(var - A, 1e-30)

            return p, mean + delta1, mean + delta2, jnp.sqrt(s2)

        return lax.cond(no_bracket, gaussian_fallback, do_bisect)

    return lax.cond(skew_scale < 1e-4, gaussian_fallback, fit)


def sample_mixture(key, p, mu1, mu2, s, shape):
    k1, k2 = random.split(key)
    choose = random.uniform(k1, shape) < p
    z = random.normal(k2, shape)
    return jnp.where(choose, mu1 + s * z, mu2 + s * z)


# -------------------- sLLG (Heun) --------------------
def anisotropy_field(m, u, Hk):
    mu = jnp.sum(m * u, axis=1, keepdims=True)
    return Hk * mu * u


def llg_rhs(m, H, gamma, alpha):
    mxH = jnp.cross(m, H)
    mxmxH = jnp.cross(m, mxH)
    pref = -gamma / (1.0 + alpha * alpha)
    return pref * (mxH + alpha * mxmxH)


# Capture scalars (static for JIT)
N = int(cfg.N_PARTICLES)
c = float(cfg.CONCENTRATION)
Is = float(cfg.MS)
Ms = float(cfg.MS)
K = float(cfg.K_ANIS)
T = float(cfg.TEMP)
alpha = float(cfg.ALPHA)
gamma = float(cfg.GAMMA)

d_mean = float(cfg.D_MEAN)
d_logsigma = float(cfg.D_LOGSIGMA)

d0 = float(cfg.D_MEAN)
d = float(cfg.CYL_HEIGHT)
R = float(cfg.CYL_RADIUS)

DT = float(cfg.DT)
N_STEPS = int(cfg.N_STEPS)
H_EXT = float(cfg.H_EXT_VAL)


def heun_step(key, m, u, diam, Hz_ext, Hz_int):
    V = (jnp.pi / 6.0) * (diam**3)
    Hk = (2.0 * K) / Ms

    # Thermal field std (convention-dependent)
    Hth_std = jnp.sqrt((2.0 * alpha * kB_erg * T) / (gamma * Ms * V * DT + 1e-30))
    Hth_std = Hth_std.reshape((-1, 1))

    k_th, _ = random.split(key)
    H_th = Hth_std * random.normal(k_th, (m.shape[0], 3))

    H_ext = jnp.array([0.0, 0.0, Hz_ext])
    H_int = jnp.stack([jnp.zeros_like(Hz_int), jnp.zeros_like(Hz_int), Hz_int], axis=1)

    H_an = anisotropy_field(m, u, Hk)
    H_eff = H_an + H_int + H_ext + H_th
    k1 = llg_rhs(m, H_eff, gamma, alpha)
    m_pred = normalize(m + DT * k1)

    H_an2 = anisotropy_field(m_pred, u, Hk)
    H_eff2 = H_an2 + H_int + H_ext + H_th
    k2 = llg_rhs(m_pred, H_eff2, gamma, alpha)

    return normalize(m + 0.5 * DT * (k1 + k2))


@jax.jit
def run_const_field(key):
    # init
    k_u, k_d, k_m, key = random.split(key, 4)
    u = random_unit_vectors(k_u, N)
    diam = lognormal_diameters(k_d, N, d_mean, d_logsigma)
    m = normalize(random_unit_vectors(k_m, N))

    def step_fn(carry, idx):
        key, m = carry

        zeta = jnp.mean(m[:, 2])
        meanH, varH, mu3, mu4 = kharitonskii_moments_Oz(c, Is, d0, d, R, zeta)
        p_mix, mu1, mu2, s = mixture_params_from_moments(meanH, varH, mu3, mu4)

        k_mix, k_step, key2 = random.split(key, 3)
        Hz_int = sample_mixture(k_mix, p_mix, mu1, mu2, s, (N,))

        m_new = heun_step(k_step, m, u, diam, H_EXT, Hz_int)

        out = jnp.array(
            [
                jnp.mean(m_new[:, 0]),
                jnp.mean(m_new[:, 1]),
                jnp.mean(m_new[:, 2]),
                p_mix,
                s,
            ],
            dtype=jnp.float32,
        )
        return (key2, m_new), out

    (key_f, m_f), traj = lax.scan(step_fn, (key, m), jnp.arange(N_STEPS))
    return traj  # (N_STEPS, 5)


# -------------------- RUN --------------------
key = random.PRNGKey(0)
t0 = time.time()
traj = run_const_field(key)
traj_host = np.array(traj)
print("Done. wall time:", time.time() - t0, "s", "traj shape:", traj_host.shape)

Mx, My, Mz, p_hist, s_hist = traj_host.T
t = np.arange(N_STEPS) * DT


plt.figure()
plt.plot(t * 1e9, Mz)
plt.xlabel("t (ns)")
plt.ylabel("<m_z>")
plt.title("Constant field: <m_z>(t)")

plt.figure()
plt.plot(t * 1e9, p_hist)
plt.xlabel("t (ns)")
plt.ylabel("p(t)")
plt.title("Mixture weight p(t)")

plt.figure()
plt.plot(t * 1e9, s_hist)
plt.xlabel("t (ns)")
plt.ylabel("s(t) [Oe]")
plt.title("Mixture std s(t)")

plt.show()

import matplotlib.gridspec as gridspec

Mx, My, Mz, p_hist, s_hist = traj_host.T
t = np.arange(int(cfg.N_STEPS)) * float(cfg.DT)
t_ns = t * 1e9
# Создаем большую фигуру
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

# ГРАФИК 1: Компоненты намагниченности
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t_ns, Mx, label="$\langle m_x \\rangle$", color="#E63946", alpha=0.7)
ax1.plot(t_ns, My, label="$\langle m_y \\rangle$", color="#2A9D8F", alpha=0.7)
ax1.plot(t_ns, Mz, label="$\langle m_z \\rangle$", color="#457B9D", lw=2)
ax1.set_title("Динамика намагниченности (Ансамбль)", fontweight="bold")
ax1.set_xlabel("Время (нс)")
ax1.set_ylabel("m")
ax1.legend()
ax1.grid(True, alpha=0.2)

# ГРАФИК 2: Фазовый портрет (показывает путь релаксации)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(Mx, Mz, color="#6A0572", lw=0.6, alpha=0.6)
ax2.set_title(
    "Траектория $\langle m_x \\rangle$ vs $\langle m_z \\rangle$", fontweight="bold"
)
ax2.set_xlabel("$m_x$")
ax2.set_ylabel("$m_z$")
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
ax2.add_artist(plt.Circle((0, 0), 1, color="gray", fill=False, ls="--"))  # Сфера m=1
ax2.grid(True, alpha=0.2)

# ГРАФИК 3: Параметр смеси p(t) (Strategy A)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t_ns, p_hist, color="#F4A261", lw=2)
ax3.set_title("Вес смеси $p(t)$ (Асимметрия полей)", fontweight="bold")
ax3.set_xlabel("Время (нс)")
ax3.set_ylabel("p")
ax3.grid(True, alpha=0.2)

# ГРАФИК 4: Дисперсия локальных полей s(t)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_ns, s_hist, color="#264653", lw=2)
ax4.set_title("Ширина распределения полей $s(t)$ [Oe]", fontweight="bold")
ax4.set_xlabel("Время (нс)")
ax4.set_ylabel("$\sigma$ (Э)")
ax4.grid(True, alpha=0.2)

plt.suptitle(
    f"JAX Simulation Dashboard | N={cfg.N_PARTICLES} | c={cfg.CONCENTRATION}",
    fontsize=14,
    fontweight="bold",
)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 1. Настройка для одной частицы
N_TEST = 1
m_init = np.array([[1.0, 0.0, 0.0]])  # Начинаем с оси X
u_axis = np.array([[0.0, 0.0, 1.0]])  # Анизотропия вдоль Z
H_ext = 500.0  # Сильное поле для четкой прецессии


# Функции для расчета (упрощенные для теста)
def llg_simple(m, H_eff, gamma, alpha):
    mxH = np.cross(m, H_eff)
    mxmxH = np.cross(m, mxH)
    return -(gamma / (1 + alpha**2)) * (mxH + alpha * mxmxH)


# Параметры из картинки
dt = 1e-12
steps = 20000
alpha = 0.05  # Малое затухание для долгой прецессии
gamma = 1.76e7
Hk = 200.0

history = []
m = m_init[0]

for i in range(steps):
    H_eff = np.array([0, 0, H_ext]) + Hk * (m @ u_axis[0]) * u_axis[0]

    # Шаг Хойна
    k1 = llg_simple(m, H_eff, gamma, alpha)
    m_pred = m + k1 * dt
    m_pred /= np.linalg.norm(m_pred)

    k2 = llg_simple(m_pred, H_eff, gamma, alpha)
    m = m + 0.5 * (k1 + k2) * dt
    m /= np.linalg.norm(m)

    history.append(m.copy())

history = np.array(history)
t_ns = np.arange(steps) * dt * 1e9

# ВИЗУАЛИЗАЦИЯ (как на твоем референсе)
plt.figure(figsize=(10, 4))
plt.plot(t_ns, history[:, 0], label="mx", color="#1f77b4")
plt.plot(t_ns, history[:, 1], label="my", color="#ff7f0e")
plt.plot(t_ns, history[:, 2], label="mz", color="#2ca02c", lw=2)

plt.title("Прецессия намагниченности (LLG)", fontsize=12)
plt.xlabel("Время (нс)")
plt.ylabel("m")
plt.grid(True, alpha=0.7)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
