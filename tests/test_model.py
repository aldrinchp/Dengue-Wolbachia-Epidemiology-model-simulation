"""Tests de la Etapa 2: derivadas del modelo (model.py)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dengue_wolbachia.model import compatibility_index, rhs
from dengue_wolbachia.parameters import load_config


# ---------------------------------------------------------------------------
# compatibility_index
# ---------------------------------------------------------------------------

def test_compatibility_index_no_males_is_zero() -> None:
    assert compatibility_index(0.0, 0.0, eps=1e-12) == 0.0


def test_compatibility_index_only_wild_males_is_one() -> None:
    ci = compatibility_index(N_M=1000.0, W_M=0.0, eps=1e-12)
    assert ci == pytest.approx(1.0, abs=1e-9)


def test_compatibility_index_only_wolbachia_males_is_zero() -> None:
    ci = compatibility_index(N_M=0.0, W_M=1000.0, eps=1e-12)
    assert ci == pytest.approx(0.0, abs=1e-9)


def test_compatibility_index_collapses_as_wolbachia_males_dominate() -> None:
    ci_balanced = compatibility_index(N_M=100.0, W_M=100.0, eps=1e-12)
    ci_dominated = compatibility_index(N_M=100.0, W_M=100000.0, eps=1e-12)
    assert ci_balanced == pytest.approx(0.5, abs=1e-9)
    assert ci_dominated < 1e-2


# ---------------------------------------------------------------------------
# rhs: pureza y no mutación del argumento de entrada
# ---------------------------------------------------------------------------

def test_rhs_does_not_mutate_input_state() -> None:
    params = load_config().parameters
    y = np.array([1000.0, 10.0, 1000.0, 0.0, 0.0, 9000.0, 100.0, 900.0, 100.0])
    y_copy = y.copy()
    rhs(0.0, y, params)
    assert np.array_equal(y, y_copy)


def test_rhs_is_deterministic() -> None:
    params = load_config().parameters
    y = np.array([1000.0, 10.0, 1000.0, 0.0, 0.0, 9000.0, 100.0, 900.0, 100.0])
    d1 = rhs(0.0, y, params)
    d2 = rhs(5.0, y, params)  # sistema autónomo: t no debe influir
    assert np.array_equal(d1, d2)


# ---------------------------------------------------------------------------
# Wolbachia en cero => se reduce al modelo silvestre + dengue (sin CI activo)
# ---------------------------------------------------------------------------

def _reduced_wild_rhs(t: float, y: np.ndarray, params) -> np.ndarray:
    """Reimplementación directa e independiente del submodelo sin Wolbachia.

    Con W_F = W_M = 0, CI = N_M/(N_M+eps) ~= 1: se reescribe aquí a mano
    (CI = 1 exacto) como referencia independiente para el test de reducción.
    """
    N_FS, N_FI, N_M, S, I, R = y
    P = N_FS + N_FI + N_M
    N_F = N_FI + N_FS

    dN_FS = (
        params.rho_N * params.f * N_F * 1.0
        - params.alpha_N * N_FS
        - params.beta_N * N_FS * P
        - params.mu_N * N_FS * I
    )
    dN_FI = params.mu_N * N_FS * I - params.alpha_N * N_FI - params.beta_N * N_FI * P
    dN_M = (
        params.rho_N * (1.0 - params.f) * N_F * 1.0
        - params.alpha_N * N_M
        - params.beta_N * N_M * P
    )
    dS = params.gamma * R - params.mu_H * S * N_FI
    dI = params.mu_H * S * N_FI - params.alpha_H * I
    dR = params.alpha_H * I - params.gamma * R
    return np.array([dN_FS, dN_FI, dN_M, dS, dI, dR], dtype=float)


def test_wolbachia_zero_reduces_to_wild_model() -> None:
    cfg = load_config()
    params = cfg.parameters

    y0_full = np.array([9000.0, 50.0, 9000.0, 0.0, 0.0, 90000.0, 900.0, 9100.0, 900.0])
    y0_reduced = np.array([9000.0, 50.0, 9000.0, 90000.0, 900.0, 9100.0])

    t_span = (0.0, 200.0)
    t_eval = np.linspace(*t_span, 50)

    sol_full = solve_ivp(
        rhs, t_span, y0_full, method="LSODA", args=(params,),
        t_eval=t_eval, rtol=1e-8, atol=1e-10,
    )
    sol_reduced = solve_ivp(
        _reduced_wild_rhs, t_span, y0_reduced, method="LSODA", args=(params,),
        t_eval=t_eval, rtol=1e-8, atol=1e-10,
    )
    assert sol_full.success
    assert sol_reduced.success

    # W_F, W_M deben permanecer exactamente en cero (sin fuente propia).
    W_F, W_M = sol_full.y[3], sol_full.y[4]
    assert np.allclose(W_F, 0.0, atol=1e-9)
    assert np.allclose(W_M, 0.0, atol=1e-9)

    full_wild_vars = np.vstack([sol_full.y[0], sol_full.y[1], sol_full.y[2],
                                 sol_full.y[5], sol_full.y[6], sol_full.y[7]])
    np.testing.assert_allclose(full_wild_vars, sol_reduced.y, rtol=1e-5, atol=1e-6)


def test_dW_dt_is_zero_when_wolbachia_absent() -> None:
    params = load_config().parameters
    y = np.array([9000.0, 50.0, 9000.0, 0.0, 0.0, 90000.0, 900.0, 9100.0, 900.0])
    dy = rhs(0.0, y, params)
    assert dy[3] == 0.0  # dW_F/dt
    assert dy[4] == 0.0  # dW_M/dt


def test_human_population_derivative_sums_to_zero() -> None:
    params = load_config().parameters
    y = np.array([9000.0, 50.0, 9000.0, 100.0, 100.0, 90000.0, 900.0, 9100.0, 900.0])
    dy = rhs(0.0, y, params)
    dS, dI, dR = dy[5], dy[6], dy[7]
    assert dS + dI + dR == pytest.approx(0.0, abs=1e-9)


def test_dC_dt_equals_incidence_term() -> None:
    params = load_config().parameters
    y = np.array([9000.0, 50.0, 9000.0, 100.0, 100.0, 90000.0, 900.0, 9100.0, 900.0])
    dy = rhs(0.0, y, params)
    S = y[5]
    expected_incidence = params.mu_H * S * y[1]
    assert dy[8] == pytest.approx(expected_incidence)
    # dC/dt no resta nada (a diferencia de dI/dt): C es monótonamente creciente.
    assert dy[8] >= 0
