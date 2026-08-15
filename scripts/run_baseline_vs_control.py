#!/usr/bin/env python3
"""Single case: baseline (no control) vs. Wolbachia population-replacement control.

Runs exactly two simulations from the same parameters and starting point:

  A) baseline  — no Wolbachia is ever released; dengue circulates freely.
  B) control   — a single release of wMel mosquitoes (both sexes) at t=0,
                 sized as a fraction of the baseline wild equilibrium.

Also computes R0 (basic reproduction number) at the disease-free wild
equilibrium, i.e. the baseline scenario before any release.

Usage
-----
    python scripts/run_baseline_vs_control.py
    python scripts/run_baseline_vs_control.py --release-fraction 0.5 --t-final 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dengue_wolbachia.equilibria import basic_reproduction_number, wild_equilibrium
from dengue_wolbachia.parameters import STATE_VARS, load_config
from dengue_wolbachia.plots import plot_infection_comparison, plot_stacked_timeseries
from dengue_wolbachia.simulate import export_csv, integrate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_IDX = {name: i for i, name in enumerate(STATE_VARS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--release-fraction", type=float, default=0.35,
                         help="Fraction of the baseline wild equilibrium P* released as W_F and W_M at t=0")
    parser.add_argument("--t-final", type=float, default=150.0)
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--out-data", type=Path, default=PROJECT_ROOT / "outputs" / "data")
    parser.add_argument("--out-figures", type=Path, default=PROJECT_ROOT / "outputs" / "figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = cfg.parameters
    t_eval = np.linspace(0.0, args.t_final, args.n_points)

    # --- R0 at the disease-free wild equilibrium (baseline, no control) -----
    N_FS_star, N_M_star, P_star = wild_equilibrium(params)
    r0_baseline = basic_reproduction_number(params, N_FS_star, P_star)
    print(f"Wild equilibrium (no Wolbachia): N_FS*={N_FS_star:.1f}, N_M*={N_M_star:.1f}, P*={P_star:.1f}")
    print(f"R0 (baseline, no control)       = {r0_baseline:.3f}")

    # --- A) baseline: integrate the configured initial condition as-is -----
    result_a = integrate(params, cfg.initial_state, (0.0, args.t_final), cfg.numerics, t_eval=t_eval)

    # --- B) control: add a single Wolbachia release to the initial state ---
    release_amount = args.release_fraction * P_star
    y0_b = cfg.initial_state.copy()
    y0_b[_IDX["W_F"]] += release_amount
    y0_b[_IDX["W_M"]] += release_amount
    result_b = integrate(params, y0_b, (0.0, args.t_final), cfg.numerics, t_eval=t_eval)

    I_a_final = result_a.y[_IDX["I"], -1]
    I_b_final = result_b.y[_IDX["I"], -1]

    # R0 is a property of dengue, not of Wolbachia: the same formula is
    # re-evaluated at whatever mosquito state the control run settles into
    # (not the wild-only closed form, which no longer applies once
    # Wolbachia is present) — it naturally goes to 0 once N_FS* -> 0.
    N_FS_b_final = result_b.y[_IDX["N_FS"], -1]
    P_b_final = sum(result_b.y[_IDX[name], -1] for name in ("N_FS", "N_FI", "N_M", "W_F", "W_M"))
    r0_control = basic_reproduction_number(params, N_FS_b_final, P_b_final)

    # Cumulative incidence C: same inflow as I (mu_H*S*N_FI) but never
    # decreases (no recovery term) — total cases accumulated since t=0.
    C_a_final = result_a.y[_IDX["C"], -1]
    C_b_final = result_b.y[_IDX["C"], -1]
    cases_averted = C_a_final - C_b_final

    print(f"\nRelease at t=0: {release_amount:.1f} W_F + {release_amount:.1f} W_M "
          f"({args.release_fraction:.0%} of P*)")
    print(f"Final I           — baseline: {I_a_final:.0f}       control: {I_b_final:.0f}")
    print(f"Final N_FS        — baseline: {N_FS_star:.0f}     control: {N_FS_b_final:.1f}")
    print(f"R0 (dengue)       — baseline: {r0_baseline:.3f}   control: {r0_control:.3f}")
    print(f"Cumulative cases  — baseline: {C_a_final:.0f}     control: {C_b_final:.0f}"
          f"   (averted: {cases_averted:.0f})")

    # --- exports --------------------------------------------------------
    args.out_data.mkdir(parents=True, exist_ok=True)
    args.out_figures.mkdir(parents=True, exist_ok=True)
    export_csv(result_a, args.out_data / "baseline.csv")
    export_csv(result_b, args.out_data / "control.csv")

    # Scalar summary (R0, final states, cumulative cases) — the trajectory
    # CSVs above don't carry these, since they're single numbers per
    # scenario, not a time series. This is the one persistent place to find
    # R0 after the run without re-reading the console output.
    summary_path = args.out_data / "summary.csv"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("metric,baseline,control\n")
        fh.write(f"R0,{r0_baseline:.6f},{r0_control:.6f}\n")
        fh.write(f"final_N_FS,{N_FS_star:.6f},{N_FS_b_final:.6f}\n")
        fh.write(f"final_I,{I_a_final:.6f},{I_b_final:.6f}\n")
        fh.write(f"cumulative_cases_C,{C_a_final:.6f},{C_b_final:.6f}\n")
        fh.write(f"cases_averted,,{cases_averted:.6f}\n")
        fh.write(f"release_amount_each_sex,,{release_amount:.6f}\n")
        fh.write(f"t_final_days,{args.t_final:.6f},{args.t_final:.6f}\n")

    plot_stacked_timeseries(result_a, "Baseline: no Wolbachia control", args.out_figures / "baseline_timeseries")
    plot_stacked_timeseries(result_b, "Wolbachia control: wMel release", args.out_figures / "control_timeseries")
    plot_infection_comparison(
        {"Baseline (no control)": result_a, "Wolbachia control (wMel release at t=0)": result_b},
        "Dengue infections",
        args.out_figures / "comparison",
    )
    plot_infection_comparison(
        {"Baseline (no control)": result_a, "Wolbachia control (wMel release at t=0)": result_b},
        "Cumulative dengue cases",
        args.out_figures / "cumulative_incidence",
        variable="C",
        ylabel="Cumulative cases $C(t)$",
    )

    print(f"\nSummary (incl. R0) written to {summary_path}")
    print(f"Data exported to {args.out_data}")
    print(f"Figures exported to {args.out_figures}")


if __name__ == "__main__":
    main()
