"""Tests for parameter loading and validation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dengue_wolbachia.parameters import STATE_VARS, NumericsConfig, Parameters, load_config


def test_load_config() -> None:
    cfg = load_config()
    assert isinstance(cfg.parameters, Parameters)
    assert cfg.initial_state.shape == (len(STATE_VARS),)
    assert np.all(cfg.initial_state >= 0)
    assert isinstance(cfg.numerics, NumericsConfig)
    assert cfg.numerics.method in ("LSODA", "Radau")


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("no-existe.yaml")


def test_parameters_rejects_nonpositive_rate() -> None:
    with pytest.raises(ValueError):
        Parameters(
            rho_N=1.25,
            alpha_N=0.0,  # invalid: must be > 0
            beta_N=1e-4,
            f=0.5,
            rho_W=1.1,
            alpha_W=0.05,
            beta_W=1e-4,
            q=0.5,
            mu_N=5e-5,
            mu_H=5e-5,
            alpha_H=0.1428,
            gamma=0.0055,
            H=100000,
        )


@pytest.mark.parametrize("bad_value", [0.0, 1.0, -0.1, 1.5])
def test_parameters_rejects_f_outside_open_unit_interval(bad_value: float) -> None:
    with pytest.raises(ValueError):
        Parameters(
            rho_N=1.25,
            alpha_N=0.04,
            beta_N=1e-4,
            f=bad_value,
            rho_W=1.1,
            alpha_W=0.05,
            beta_W=1e-4,
            q=0.5,
            mu_N=5e-5,
            mu_H=5e-5,
            alpha_H=0.1428,
            gamma=0.0055,
            H=100000,
        )


def test_parameters_warns_when_wild_population_not_viable() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Parameters(
            rho_N=0.05,  # rho_N * f = 0.025 <= alpha_N = 0.04
            alpha_N=0.04,
            beta_N=1e-4,
            f=0.5,
            rho_W=1.1,
            alpha_W=0.05,
            beta_W=1e-4,
            q=0.5,
            mu_N=5e-5,
            mu_H=5e-5,
            alpha_H=0.1428,
            gamma=0.0055,
            H=100000,
        )
    assert any("wild population is not viable" in str(w.message) for w in caught)
