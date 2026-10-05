import jax
import jax.numpy as jnp
from jax import random, jit
import matplotlib.pyplot as plt
import time

# ==============================================================================
# 1. ГЛОБАЛЬНЫЕ КОНСТАНТЫ
# ==============================================================================
N_PARTICLES = 400000
MS = 480.0
KU = 1.35e5
ALPHA = 0.1
TEMP = 300.0
KB = 1.38e-16
GAMMA = 1.76e7
DT = 1.0e-12
N_STEPS = 10000
SAVE_EVERY = 50

CONC = 0.05
R_SAMPLE = 5e-4
D_SAMPLE = 1e-4

# ==============================================================================
# 2. ФИЗИЧЕСКИЕ ЯДРА
# ==============================================================================


@jit
def get_kharitonskii_field(zeta_v, r_moms, key):
    theta_max = jnp.arctan2(2.0 * R_SAMPLE, D_SAMPLE)
    eta = 1.0 - 1.5 * jnp.cos(theta_max)
    A1 = 4.0 * jnp.pi / 3.0

    r3, r6, r9 = r_moms

    mean_h = -2.0 * A1 * CONC * MS * eta * zeta_v
    sig2 = (A1**2) * CONC * (MS**2) * 0.1 * jnp.maximum(1.0 - 3.75 * r3, 0.05)
    mu3 = (A1**3) * CONC * (MS**3) * (1 / 280) * (1.0 - 67.0 * r6) * zeta_v
    mu4 = (A1**4) * CONC * (MS**4) * (41 / 1120) * (1.0 - 264.0 * r9)

    z = random.normal(key, (N_PARTICLES,))
    z = z - jnp.mean(z)

    sig = jnp.sqrt(jnp.maximum(sig2, 1e-12))
    k3 = mu3 / (sig**3 + 1e-20)
    k4 = (mu4 / (sig**4 + 1e-20)) - 3.0

    h_rand = z + (k3 / 6.0) * (z**2 - 1.0) + (k4 / 24.0) * (z**3 - 3.0 * z)

    return jnp.clip(mean_h + h_rand * sig, -6000.0, 6000.0)


@jit
def sllg_step(m, axes, vols, r_moms, key):
    gp = -GAMMA / (1.0 + ALPHA**2)
    k_dd, k_th = random.split(key)

    zeta_v = jnp.sum(m[:, 2] * vols) / jnp.sum(vols)

    h_dd_z = get_kharitonskii_field(zeta_v, r_moms, k_dd)

    sig_th = jnp.sqrt((2 * ALPHA * KB * TEMP) / (GAMMA * MS * vols * DT))
    h_th = random.normal(k_th, (N_PARTICLES, 3)) * sig_th[:, None]

    h_ext = jnp.array([0.0, 0.0, 40.0])

    def calc_h(mag):
        h_an = (2 * KU / MS) * jnp.sum(mag * axes, axis=1, keepdims=True) * axes
        h_tot = h_an + h_ext + h_th
        return h_tot.at[:, 2].add(h_dd_z)

    def rhs(mag, field):
        mxH = jnp.cross(mag, field)
        return gp * (mxH + ALPHA * jnp.cross(mag, mxH))

    # Predictor
    h1 = calc_h(m)
    k1 = rhs(m, h1)
    m_pred = m + k1 * DT
    m_pred /= jnp.linalg.norm(m_pred, axis=-1, keepdims=True)

    # Corrector
    h2 = calc_h(m_pred)
    k2 = rhs(m_pred, h2)
    m_new = m + 0.5 * (k1 + k2) * DT

    return m_new / jnp.linalg.norm(m_new, axis=-1, keepdims=True), zeta_v


# ==============================================================================
# 3. ОСНОВНОЙ ЦИКЛ
# ==============================================================================


def run():
    key = random.PRNGKey(int(time.time()))
    k_diam, k_axes, k_sim = random.split(key, 3)

    # ---- Логнормальное распределение (исправлено) ----
    mu = jnp.log(12e-7)
    sigma = 0.2
    normal_samples = random.normal(k_diam, (N_PARTICLES,))
    diams = jnp.exp(mu + sigma * normal_samples)
    vols = (jnp.pi / 6.0) * diams**3

    # ---- Случайные оси (исправлены ключи) ----
    k_axes1, k_axes2 = random.split(k_axes)
    phi = random.uniform(k_axes1, (N_PARTICLES,)) * 2 * jnp.pi
    ct = random.uniform(k_axes2, (N_PARTICLES,)) * 2 - 1
    st = jnp.sqrt(1.0 - ct**2)
    axes = jnp.stack([st * jnp.cos(phi), st * jnp.sin(phi), ct], axis=1)

    # ---- Моменты ----
    r_moms = jnp.array(
        [
            jnp.mean(diams**3) / D_SAMPLE**3,
            jnp.mean(diams**6) / D_SAMPLE**6,
            jnp.mean(diams**9) / D_SAMPLE**9,
        ]
    )

    m = axes
    history = []

    print(f"JAX Simulation: {N_PARTICLES} particles...")
    t0 = time.time()

    for i in range(N_STEPS):
        k_sim, subkey = random.split(k_sim)
        m, zv = sllg_step(m, axes, vols, r_moms, subkey)

        if i % SAVE_EVERY == 0:
            history.append(float(zv))

    print(f"Done in {time.time() - t0:.2f} s")
    # Разделяем частицы на группы по объему (например, 5 групп)
    v_sorted_idx = jnp.argsort(vols)  # Сортируем индексы по объему
    groups = jnp.array_split(v_sorted_idx, 5)

    plt.figure(figsize=(10, 6))
    for i, group_idx in enumerate(groups):
        # Считаем среднюю намагниченность только для этой группы частиц
        m_group = m[group_idx, 2].mean()
        v_mean = vols[group_idx].mean() * 1e21  # в нм^3
        plt.bar(f"Группа {i + 1}\n({v_mean:.0f} нм³)", m_group)

    plt.title("Намагниченность групп частиц разного объема")
    plt.ylabel("Mz")
    plt.show()
    return history, diams


# ==============================================================================
# 4. ЗАПУСК
# ==============================================================================

mz_data, d_dist = run()

# ==============================================================================
# 5. ГРАФИКИ
# ==============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Динамика
ax1.plot(
    jnp.arange(len(mz_data)) * SAVE_EVERY * DT * 1e9, mz_data, color="firebrick", lw=2
)
ax1.set_title(r"Relaxation $\langle M_z \rangle_V$")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Normalized Mz")
ax1.grid(True, alpha=0.3)

# Распределение
ax2.hist(d_dist * 1e7, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
ax2.set_title("Particle Size Distribution (nm)")
ax2.set_xlabel("Diameter (nm)")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
