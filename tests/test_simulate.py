"""Integration tests: convergence to equilibrium, conservation, and positivity."""

from __future__ import annotations

import numpy as np
import pytest

from dengue_wolbachia.equilibria import wild_equilibrium_P_star
from dengue_wolbachia.parameters import STATE_VARS, load_config
from dengue_wolbachia.simulate import integrate

_IDX = {name: i for i, name in enumerate(STATE_VARS)}


# ---------------------------------------------------------------------------
# Convergence to the analytic equilibrium
# ---------------------------------------------------------------------------

def test_converges_to_wild_equilibrium_P_star() -> None:
    cfg = load_config()
    params = cfg.parameters
    # No Wolbachia, no dengue: I=N_FI=0 stays exactly at 0
    # (both coupling terms vanish), so the system reduces to the pure
    # wild submodel.
    y0 = np.array([500.0, 0.0, 8000.0, 0.0, 0.0, params.H, 0.0, 0.0, 0.0])
    result = integrate(params, y0, (0.0, 5000.0), cfg.numerics)

    P_final = (
        result.y[_IDX["N_FS"], -1]
        + result.y[_IDX["N_FI"], -1]
        + result.y[_IDX["N_M"], -1]
        + result.y[_IDX["W_F"], -1]
        + result.y[_IDX["W_M"], -1]
    )
    P_star = wild_equilibrium_P_star(params)
    assert P_final == pytest.approx(P_star, rel=1e-6)


# ---------------------------------------------------------------------------
# Conservation of the human population
# ---------------------------------------------------------------------------

def test_human_population_conserved() -> None:
    cfg = load_config()
    t_eval = np.linspace(0.0, 2000.0, 300)
    result = integrate(cfg.parameters, cfg.initial_state, (0.0, 2000.0), cfg.numerics, t_eval=t_eval)
    total = result.y[_IDX["S"]] + result.y[_IDX["I"]] + result.y[_IDX["R"]]
    assert np.max(np.abs(total - cfg.parameters.H)) < 1e-8


# ---------------------------------------------------------------------------
# Positivity
# ---------------------------------------------------------------------------

def test_no_negative_values_in_baseline_simulation() -> None:
    cfg = load_config()
    t_eval = np.linspace(0.0, 2000.0, 300)
    result = integrate(cfg.parameters, cfg.initial_state, (0.0, 2000.0), cfg.numerics, t_eval=t_eval)
    assert np.all(result.y >= 0.0)


def test_positivity_clip_accepts_tiny_negative_noise() -> None:
    from dengue_wolbachia.simulate import _clip_or_raise_negatives

    y = np.array([[1.0, -1e-13], [2.0, 3.0]])
    cleaned = _clip_or_raise_negatives(y)
    assert np.all(cleaned >= 0.0)


def test_positivity_clip_raises_on_appreciable_negative() -> None:
    from dengue_wolbachia.simulate import _clip_or_raise_negatives

    y = np.array([[1.0, -5.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        _clip_or_raise_negatives(y)


# ---------------------------------------------------------------------------
# Single Wolbachia release at t=0 (via initial condition), as in
# scripts/run_baseline_vs_control.py
# ---------------------------------------------------------------------------

def test_single_release_at_t0_establishes_wolbachia() -> None:
    cfg = load_config()
    params = cfg.parameters
    y0 = cfg.initial_state.copy()
    y0[_IDX["W_F"]] += 2000.0
    y0[_IDX["W_M"]] += 2000.0

    t_eval = np.linspace(0.0, 730.0, 200)
    result = integrate(params, y0, (0.0, 730.0), cfg.numerics, t_eval=t_eval)

    # With a sufficiently large release, Wolbachia establishes and
    # displaces the wild population (bistability due to cytoplasmic
    # incompatibility — see model.py).
    assert result.y[_IDX["W_F"], -1] > 0
    assert result.y[_IDX["N_FS"], -1] == pytest.approx(0.0, abs=1e-3)
