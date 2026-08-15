"""Tests for the analytic wild equilibrium and R0 (equilibria.py).

Numerical-integration convergence tests are in test_simulate.py.
"""

from __future__ import annotations

import pytest

from dengue_wolbachia.equilibria import (
    basic_reproduction_number,
    next_generation_number,
    wild_equilibrium,
    wild_equilibrium_P_star,
)
from dengue_wolbachia.parameters import Parameters


def _base_params(**overrides) -> Parameters:
    defaults = dict(
        rho_N=1.25,
        alpha_N=0.04,
        beta_N=1.0e-4,
        f=0.5,
        rho_W=1.10,
        alpha_W=0.05,
        beta_W=1.0e-4,
        q=0.5,
        mu_N=5.0e-5,
        mu_H=5.0e-5,
        alpha_H=0.1428,
        gamma=0.0055,
        H=100000,
    )
    defaults.update(overrides)
    return Parameters(**defaults)


def test_equilibrium_raises_when_not_viable() -> None:
    with pytest.warns(UserWarning):
        params = _base_params(rho_N=0.05, f=0.5, alpha_N=0.04)  # rho_N*f=0.025 <= alpha_N
    with pytest.raises(ValueError):
        wild_equilibrium_P_star(params)


def test_wild_equilibrium_p_star_closed_form() -> None:
    params = _base_params(rho_N=1.25, f=0.5, alpha_N=0.04, beta_N=1.0e-4)
    expected = (1.25 * 0.5 - 0.04) / 1.0e-4
    assert wild_equilibrium_P_star(params) == pytest.approx(expected)


def test_wild_equilibrium_breakdown_sums_to_P_star() -> None:
    params = _base_params()
    N_FS, N_M, P = wild_equilibrium(params)
    assert N_FS + N_M == pytest.approx(P)
    assert N_FS == pytest.approx(params.f * P)
    assert N_M == pytest.approx((1 - params.f) * P)


def test_r0_positive_without_release() -> None:
    params = _base_params()
    N_FS_star, _, P_star = wild_equilibrium(params)
    r0 = basic_reproduction_number(params, N_FS_star, P_star)
    assert r0 > 0


def test_r0_zero_when_no_wild_females() -> None:
    params = _base_params()
    assert basic_reproduction_number(params, 0.0, 100.0) == 0.0


def test_r0_matches_closed_form() -> None:
    params = _base_params()
    N_FS_star, _, P_star = wild_equilibrium(params)
    expected_ng = (
        params.mu_H * params.H * params.mu_N * N_FS_star
        / (params.alpha_H * (params.alpha_N + params.beta_N * P_star))
    ) ** 0.5
    assert next_generation_number(params, N_FS_star, P_star) == pytest.approx(expected_ng)
    assert basic_reproduction_number(params, N_FS_star, P_star) == pytest.approx(expected_ng**2)


def test_r0_equals_next_generation_number_squared() -> None:
    params = _base_params()
    N_FS_star, _, P_star = wild_equilibrium(params)
    r0_ng = next_generation_number(params, N_FS_star, P_star)
    assert basic_reproduction_number(params, N_FS_star, P_star) == pytest.approx(r0_ng**2)
