# dengue-wolbachia

Epidemiological model of dengue with biological control via *Wolbachia*: a
system of 8 ordinary differential equations coupling wild mosquitoes,
Wolbachia-carrying mosquitoes, and humans with SIRS dynamics.

This project simulates **a single case**: the baseline (no control) versus
a single release of wMel mosquitoes (population replacement) at $t=0$,
to show how *Wolbachia* acts as a biological control for dengue.

> **Note on the parameters.** All numerical values in
> `config/config.yaml` are **PROVISIONAL** (order-of-magnitude figures from
> the general literature on *Aedes aegypti*, with the time unit in
> days) and are explicitly marked as such in the file.
> Replace them with the values from your bibliographic reference before using
> the results in a report.

## Installation

Requires Python ≥ 3.10.

```bash
cd dengue-wolbachia
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

No dependencies beyond `numpy`, `scipy`, `matplotlib`, `pyyaml`, and
`pytest` (declared in `pyproject.toml`).

## Mathematical formulation

### State variables

| Variable | Meaning |
|---|---|
| $N_{FS}$ | wild females susceptible to dengue |
| $N_{FI}$ | wild females infected with dengue |
| $N_M$    | wild males |
| $W_F$    | Wolbachia-carrying females |
| $W_M$    | Wolbachia-carrying males |
| $S$      | susceptible humans |
| $I$      | infected humans |
| $R$      | recovered humans (temporary immunity) |

with the auxiliary quantities

$$
P = N_{FS}+N_{FI}+N_M+W_F+W_M, \qquad
H = S+I+R, \qquad
CI = \frac{N_M}{W_M+N_M}.
$$

### System of ODEs

$$
\begin{aligned}
\dot N_{FS} &= \rho_N f (N_{FI}+N_{FS})\,CI - \alpha_N N_{FS} - \beta_N N_{FS} P - \mu_N N_{FS} I \\
\dot N_{FI} &= \mu_N N_{FS} I - \alpha_N N_{FI} - \beta_N N_{FI} P \\
\dot N_M &= \rho_N (1-f)(N_{FI}+N_{FS})\,CI - \alpha_N N_M - \beta_N N_M P \\
\dot W_F &= \rho_W q\, W_F - \alpha_W W_F - \beta_W W_F P \\
\dot W_M &= \rho_W (1-q)\, W_F - \alpha_W W_M - \beta_W W_M P \\
\dot S &= \gamma R - \mu_H S N_{FI} \\
\dot I &= \mu_H S N_{FI} - \alpha_H I \\
\dot R &= \alpha_H I - \gamma R
\end{aligned}
$$

implemented in [`src/dengue_wolbachia/model.py`](src/dengue_wolbachia/model.py)
as the pure function `rhs(t, y, params)`, with no global state, ready to use
as the `args` of `scipy.integrate.solve_ivp`.

### Deliberate assumptions (see also the comments in `model.py`)

1. **Viral blocking.** $W_F$ does not appear in $\dot I$: Wolbachia-carrying
   females do not transmit dengue, so there is no $W_{FI}$
   compartment.
2. **Cytoplasmic incompatibility (CI).** $CI=N_M/(W_M+N_M)$ is the
   fraction of compatible matings. If $W_M \gg N_M$, $CI\to 0$ and wild
   natality collapses — this is the mechanism by which a sufficiently
   large release makes *Wolbachia* displace the wild population.
3. **Maternal inheritance.** $\dot W_F$ and $\dot W_M$ depend only on $W_F$, with
   no $CI$ factor: *Wolbachia* is inherited exclusively from the mother. That's
   why there is no equation that converts a wild mosquito into a carrier — the
   only way carriers can appear is by being born to a carrier mother, or by
   being released from outside the system.
4. **Shared density-dependent competition.** The $\beta X P$ terms
   are additional mortality proportional to the total density $P$: wild and
   carrier mosquitoes share the same pool of larval resources.
5. **Closed human population.** $\dot S+\dot I+\dot R=0$ by construction
   (no human natality/mortality in the model).

### Numerical details

- $CI$ is regularized as `CI = N_M / (W_M + N_M + eps)` with `eps=1e-12`.
- Integration via `scipy.integrate.solve_ivp`, method `LSODA` or `Radau`
  (never `odeint`): the system is stiff because mosquito rates
  (day timescale) coexist with the loss of immunity (month timescale).
  `rtol=1e-8`, `atol=1e-10` by default, configurable in `config/config.yaml`.
- Positivity: negative values on the order of $10^{-12}$ (integrator
  noise) are clipped to 0; appreciable negative values (> `1e-6`) make the
  integration fail with an explicit `ValueError` instead of being hidden.

### The control: a single release at $t=0$

The *Wolbachia* release happens exactly once, before the simulation
starts, so it is modeled directly as part of the initial
condition (the released amount is added to $W_F(0)$/$W_M(0)$) — there is no
need to split the integration into segments or to add any release term
inside the equations.

### Basic reproduction number ($R_0$)

$$R_0=\sqrt{\dfrac{\mu_H S^{*}\,\mu_N N_{FS}^{*}}{\alpha_H(\alpha_N+\beta_N P^{*})}},$$

evaluated at the disease-free wild equilibrium (baseline, no
control), via the next-generation matrix over $(I,N_{FI})$. Implemented
in [`equilibria.py`](src/dengue_wolbachia/equilibria.py).

## Project structure

```
dengue-wolbachia/
├── config/config.yaml       # all parameters + initial condition
├── src/dengue_wolbachia/
│   ├── model.py              # system derivatives (rhs)
│   ├── parameters.py         # Parameters, SimulationConfig, load_config
│   ├── simulate.py           # integration (solve_ivp) + CSV export
│   ├── equilibria.py         # analytic wild equilibrium and R0
│   └── plots.py               # figures (plain matplotlib, labels in English)
├── scripts/
│   └── run_baseline_vs_control.py   # the only script: runs the full case
├── tests/                    # pytest
└── outputs/{data,figures}/   # generated CSV and PNG/PDF
```

## Usage

```bash
python scripts/run_baseline_vs_control.py
```

This runs the two simulations (baseline and control), prints $R_0$ and the
final infected values to the console, and exports:

- `outputs/data/{baseline,control}.csv` — full trajectories of the 9 variables (includes `C`, cumulative incidence)
- `outputs/data/summary.csv` — $R_0$, final state, and cumulative/averted cases (the only place where $R_0$ is saved, not just printed to console)
- `outputs/figures/comparison.{png,pdf}` — $I(t)$: baseline vs. control, on a single axis
- `outputs/figures/baseline_timeseries.{png,pdf}` — stacked panels of the baseline
- `outputs/figures/control_timeseries.{png,pdf}` — stacked panels of the control case

Experiment parameters adjustable from the command line:

```bash
python scripts/run_baseline_vs_control.py --release-fraction 0.5 --t-final 1000
```

- `--release-fraction`: fraction of the wild equilibrium density
  ($P^*$) released as $W_F$ and $W_M$ at $t=0$ (default 0.35 — already
  confirmed to be enough for *Wolbachia* to establish with these
  PROVISIONAL parameters).
- `--t-final`: simulation duration in days (default 200).

## Expected result

With the default PROVISIONAL parameters: $R_0\approx8.18$ with no control
(dengue is self-sustaining). Releasing 35% of the wild equilibrium
population as wMel mosquitoes at $t=0$, *Wolbachia* establishes
completely (the wild population goes extinct) and infections drop to
practically zero in under 100 days — whereas without control
dengue stabilizes at a permanent endemic level.

## Tests

```bash
pytest tests/ -q
```

Covers: parameter validation, reduction of the model to pure wild
dynamics when *Wolbachia* is at zero, the analytic wild equilibrium and
$R_0$, conservation of the human population, positivity, and the
establishment of *Wolbachia* after a single sufficiently large release.
