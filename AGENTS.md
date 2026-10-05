# AGENTS.md

## Project purpose

This repository implements a GPU-oriented stochastic Landau–Lifshitz–Gilbert (sLLG) solver for ensembles of single-domain magnetic nanoparticles in CGS units.

The current physics model combines:

- uniaxial macrospin LLG dynamics;
- thermal fluctuations through an FDT-consistent stochastic field;
- an external field applied along `Oz`;
- an optional dipolar closure based on the first two moments of the `Oz` random-field distribution from Kharitonskii et al.;
- JAX/JIT execution intended for large ensembles and later use behind a UI.

The computational core is the priority. UI, plotting, file export, benchmarking, and validation utilities must remain outside the physics kernel unless explicitly requested.

## Repository boundaries: `legacy/` is off-limits

The repository-level `legacy/` directory is archived historical material and is **not part of the active codebase**.

Agents must not access `legacy/` unless the user explicitly asks for a task involving that directory. This prohibition includes:

- do not open or read files under `legacy/`;
- do not search, grep, index, summarize, or inspect `legacy/`;
- do not import, execute, test, lint, format, or modify code from `legacy/`;
- do not use `legacy/` as a reference for current APIs, physics, defaults, architecture, or implementation choices;
- do not copy code or configuration from `legacy/` into the active project;
- do not run recursive repository commands that descend into `legacy/`.

When using repository-wide tools, explicitly exclude the directory. Examples:

```text
rg ... --glob '!legacy/**'
find . -path './legacy' -prune -o ...
pytest --ignore=legacy
```

If a requested task appears to require information from `legacy/`, stop and ask for explicit permission before accessing it. Do not enter the directory merely to determine whether it is relevant.

This rule overrides general instructions to inspect the repository broadly.

---

## Non-negotiable physics conventions

### Units

Use CGS consistently unless a task explicitly introduces a conversion layer.

Expected quantities:

- `Ms`: emu/cm^3
- `K`: erg/cm^3
- magnetic fields: Oe
- particle/sample lengths inside the solver: cm
- `gamma`: rad/(s Oe)
- `dt`: s
- `kB = 1.380649e-16` erg/K

Do not silently mix SI and CGS.

### Macrospin model

Each particle is represented by a unit magnetization vector `m_i`.

Uniaxial anisotropy field:

```text
H_ani,i = Hk (m_i · e_i) e_i
Hk = 2 K / Ms
```

The deterministic LLG equation is implemented in Gilbert-equivalent form:

```text
dm/dt = -gamma/(1 + alpha^2) [m × H + alpha m × (m × H)]
```

The magnetization vector must remain normalized after integration steps.

### Thermal field

For stochastic LLG, the thermal field amplitude is

```text
sigma_th = sqrt(2 alpha kB T / (gamma Ms V dt))
```

The same thermal realization must be used in the predictor and corrector stages of a single Heun step.

Do not generate independent thermal noise for the predictor and corrector unless the numerical interpretation is intentionally changed and clearly documented.

### Dipolar closure

The current dipolar approximation uses the first two moments of the Kharitonskii `Oz` field distribution:

```text
mean_rf = -(8 pi / 3) c Is g zeta

g = 1 - (3/2) cos(theta_max)
cos(theta_max) = 1 / sqrt(1 + (2 R / d)^2)

var_rf = (4 pi / 3)^2 * c Is^2 / 10 * [1 - (15/4) (d0/d)^3]
```

where:

- `c` is the physical volume concentration of magnetic particles;
- `Is` is currently identified with `Ms`;
- `d0` is the particle diameter used by the closure;
- `d` and `R` are the cylinder height and radius;
- `zeta` is the current normalized ensemble magnetization along `Oz`.

### Physical concentration is independent of simulation size

`N` is the numerical ensemble size.

`c` is a physical material/sample parameter.

Never compute physical concentration from `N` unless the simulation explicitly represents a fixed physical sample with a fully specified volume and particle count.

For the UI-oriented solver, treat `N` and `c` as independent inputs.

### Meaning of `zeta`

For the aligned-easy-axis model,

```text
zeta(t) = <m_z(t)>
```

is used as the dynamical extension of the two-state quantity `alpha - beta` appearing in the Kharitonskii treatment.

This is a model closure, not an exact many-body identity.

### Scope of the Kharitonskii `Oz` closure

The strict interpretation of the current `Oz` closure applies to moments oriented parallel/antiparallel to the chosen `Oz` direction.

Therefore:

- `axis_mode = 0` with dipolar modes is the physically intended configuration;
- `axis_mode = 1` with random easy axes may be used for pure sLLG calculations;
- do not present `axis_mode = 1` + Kharitonskii `Oz` closure as a quantitatively justified realization of the paper unless a new orientational averaging/covariance derivation has been added.

If such a combination is retained for exploratory use, label it as a phenomenological longitudinal random-field approximation.

### Dipolar modes

Preferred naming:

```text
dipolar_mode = 0  -> off
                 1 -> mean field only
                 2 -> Gaussian closure: mean + sigma * xi
```

For mode 2:

```text
H_dd,i,z(t) = mean_rf[zeta(t)] + sigma_rf * xi_i
```

where `xi_i` is sampled once per particle and kept fixed during the trajectory.

Interpret `xi_i` as quenched spatial disorder, not thermal white noise.

Do not resample `xi_i` every integration step unless implementing a different physical model.

### Longitudinal approximation

The current closure acts only along `Oz`:

```text
H_dd = (0, 0, H_dd,z)
```

Do not silently reinterpret this as a full vector dipolar interaction.

A future full-vector closure must be introduced as a separate model/API rather than changing the meaning of the existing one.

---

## Numerical integration contract

### Heun predictor-corrector

For each time step:

1. compute `zeta_1` from the current state;
2. compute the current mean dipolar field;
3. build the deterministic field for the predictor;
4. generate one thermal field realization;
5. form the predictor state;
6. recompute `zeta_2` from the predictor;
7. recompute the mean dipolar field for the corrector;
8. use the same thermal realization in the corrector;
9. normalize the final magnetization vectors.

The quenched random contribution `sigma_rf * xi` stays unchanged between predictor and corrector.

### Time axis

Returned observables should include the true initial state at `t = 0`.

Preferred contract:

```text
len(t) == n_steps + 1
len(mz) == n_steps + 1

t[0] = 0
mz[0] = <m_z(0)>
```

Do not label the first post-step state as `t = 0`.

### Random-number handling

Use explicit JAX PRNG keys.

Requirements:

- no hidden global RNG state;
- deterministic results for the same key and parameters;
- split/fold keys explicitly;
- avoid reusing a key for distinct stochastic draws.

### JAX rules

The core should remain compatible with JIT and GPU execution.

Prefer:

- `jax.jit` for compute kernels;
- `lax.scan` for long time evolution;
- static arguments only for values that actually determine control flow or array shapes;
- JAX arrays inside compiled functions;
- no Python-side mutation inside JIT kernels.

Avoid unnecessary host/device transfers.

Do not call NumPy or Matplotlib from inside JIT-compiled functions.

---

## Core architecture

Keep the solver layered.

### Layer 1 — pure physics helpers

Examples:

```text
unit
llg_rhs
thermal_std
kharitonskii_mean_var_Oz
```

These functions should have no UI, plotting, printing, or file I/O.

### Layer 2 — ensemble initialization

Responsible for:

- particle volume;
- easy-axis distribution;
- initial magnetization state;
- deterministic reproducibility from a PRNG key.

For the first production version, monodisperse particles are preferred.

Polydispersity should be added deliberately because it changes how ensemble magnetization and the dipolar closure should be interpreted.

### Layer 3 — compiled simulation kernel

The main relaxation kernel should accept all physical and numerical parameters explicitly and return JAX arrays/scalars only.

Recommended conceptual API:

```python
relaxation_core(
    key,
    N,
    n_steps,
    Ms,
    K,
    alpha,
    gamma,
    T,
    thermal_mode,
    d_nm,
    c,
    cyl_d_cm,
    cyl_R_cm,
    dipolar_mode,
    H0,
    dt,
    p_up,
    axis_mode,
)
```

Recommended outputs:

```text
t
mz
mean_rf
sigma_rf
Hk
c
```

Only return the full final ensemble `m_fin` when downstream functionality actually needs it, because it increases device/host memory pressure.

### Layer 4 — Python/UI adapter

The future UI layer may:

- validate user input;
- convert nm/µm/mm to cm;
- convert returned time to ns/µs;
- convert JAX arrays to NumPy;
- run parameter sweeps;
- plot curves;
- export CSV/PNG/JSON;
- display warnings about model applicability.

None of this belongs in the JIT core.

---

## UI-facing parameter semantics

The UI must distinguish physical parameters from computational parameters.

### Computational parameters

Examples:

```text
N
n_steps
dt
seed
```

Changing `N` should improve/reduce sampling quality without changing the physical concentration `c`.

### Material parameters

```text
Ms
K
alpha
gamma
T
d_nm
```

### Sample/interactions

```text
c
cyl_d
cyl_R
dipolar_mode
```

### Protocol

```text
H0
initial_state / p_up
axis_mode
thermal_mode
```

### Initial-condition warning

`p_up` has a clear physical meaning for aligned easy axes.

For isotropically random easy axes, assigning `m = ±e` gives an approximately isotropic ensemble and `p_up` is not a useful control of global `Mz`.

For a future UI, prefer explicit initial-state presets such as:

```text
demagnetized
saturated +z
saturated -z
along easy axes
custom
```

Do not overload `p_up` with meanings it does not physically have.

---

## Model naming in user-visible text

Use precise names.

Preferred:

- `No dipolar interaction`
- `Kharitonskii mean-field closure`
- `Kharitonskii Gaussian random-field closure`
- `Thermal sLLG`
- `Aligned easy axes`
- `Random easy axes`

Avoid calling Gaussian mode simply "mean field" because it contains a random contribution with nonzero variance.

Avoid calling the longitudinal closure "exact dipole-dipole interaction".

---

## Do not silently change the physics

Agents must not introduce any of the following without explicit approval:

- SI/CGS unit changes;
- full vector dipolar fields in place of the longitudinal closure;
- time-resampled dipolar randomness;
- Néel–Arrhenius switching on top of thermal sLLG;
- particle-size distributions;
- inter-particle positions;
- explicit pairwise dipolar sums;
- higher Gram–Charlier moments;
- concentration derived from numerical `N`;
- a different stochastic calculus/integrator interpretation.

If any of these are requested, implement them as clearly named alternatives rather than silently modifying the existing model.

---

## Performance guidance

The target use case may involve `N ~ 10^5` or larger and many time steps.

Agents should:

- keep the time loop inside `lax.scan`;
- avoid saving full `N x 3` trajectories;
- save ensemble observables per step instead;
- avoid Python loops over particles;
- avoid repeated recompilation caused by unnecessary static arguments;
- avoid host conversions during the simulation;
- keep arrays in `float32` unless higher precision is explicitly needed.

For UI responsiveness, compilation time and execution time should be treated separately.

---

## Error handling and validation

Validation belongs in the Python/UI wrapper, not inside the deepest JIT kernel.

At minimum validate:

```text
N > 0
n_steps > 0
dt > 0
Ms > 0
K >= 0
alpha >= 0
gamma > 0
T >= 0
d_nm > 0
0 <= c <= 1
cyl_d_cm > 0
cyl_R_cm > 0
0 <= p_up <= 1
axis_mode in {0, 1}
dipolar_mode in {0, 1, 2}
thermal_mode in {0, 1}
```

Also warn if the geometry or parameter range leaves the stated domain of the closure approximation.

Do not silently clamp physically invalid user inputs except where a mathematical safety guard is already part of the formula (for example preventing division by zero or taking `sqrt` of a tiny negative value caused by roundoff).

---

## Source-of-truth priorities

When modifying the model, use the following order of authority:

1. equations and assumptions documented in this repository;
2. the cited Kharitonskii et al. paper for the random-field moments;
3. standard stochastic LLG conventions already adopted by the solver;
4. only then general micromagnetic knowledge.

Do not "improve" the model by importing assumptions from another formalism without making the change explicit.

---

## Agent workflow

Before editing the solver:

1. identify whether the requested change is numerical, physical, API-related, or UI-related;
2. preserve the unit system and parameter semantics;
3. check whether the change alters the physical model;
4. keep physics changes separate from presentation changes;
5. keep public parameter names stable where possible;
6. document any intentional incompatibility.

When proposing changes, explain them in terms of:

```text
physics meaning
numerical consequence
performance consequence
UI/API consequence
```

For small refactors that do not change the model, preserve numerical behavior for the same seed and parameters whenever practical.

---

## Current recommended production scope

For the first UI-backed release, prefer the smallest defensible model:

```text
monodisperse particles
thermal sLLG
fixed external field along Oz
aligned easy axes for Kharitonskii dipolar closure
dipolar modes: off / mean / Gaussian
independent physical concentration c
self-consistent mean field within Heun
quenched Gaussian spatial disorder
explicit t = 0 output
JAX/JIT GPU kernel
```

Random easy axes may remain available for no-dipole sLLG simulations.

More advanced physics should be added incrementally and under separate model names.

---

## References used by the current model

Primary random-field closure reference:

P. V. Kharitonskii, E. A. Setrov, A. Yu. Ralin, *Modeling of hysteresis characteristics of a dilute magnetic with dipole-dipole interaction of particles*, Materials Physics and Mechanics 52(2), 142–150 (2024).

The current solver uses only the first two `Oz` moments as a Gaussian closure. It does not currently implement the full Gram–Charlier/Bell-polynomial distribution from the paper.
