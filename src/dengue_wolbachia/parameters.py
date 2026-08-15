"""Loading and validation of dengue-Wolbachia model parameters.

The biological parameters live in :class:`Parameters`, an immutable
dataclass with validation in ``__post_init__``. The simulation
configuration (initial condition, integrator tolerances) lives in
:class:`SimulationConfig`, built from ``config/config.yaml``
via :func:`load_config`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Fixed order of the 9 state variables (8 state + C = cumulative
# incidence). The whole package (model.py, simulate.py, plots.py) must
# build/read state vectors in this order to avoid silent indexing
# errors.
STATE_VARS: tuple[str, ...] = (
    "N_FS",
    "N_FI",
    "N_M",
    "W_F",
    "W_M",
    "S",
    "I",
    "R",
    "C",
)

_CONFIG_PATH_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass(frozen=True)
class Parameters:
    """Biological parameters of the 8-ODE system.

    All rates are in units of day⁻¹ (or the corresponding per-capita
    analog).

    Parameters
    ----------
    rho_N : float
        Per-capita fecundity of wild females.
    alpha_N : float
        Per-capita mortality of wild mosquitoes (both sexes).
    beta_N : float
        Density-dependent competition coefficient, wild.
    f : float
        Proportion of females at birth in the wild population, in (0, 1).
    rho_W : float
        Per-capita fecundity of Wolbachia-carrying females.
    alpha_W : float
        Per-capita mortality of Wolbachia-carrying mosquitoes.
    beta_W : float
        Density-dependent competition coefficient, carriers.
    q : float
        Proportion of females at birth in the Wolbachia-carrying population, in (0, 1).
    mu_N : float
        Human -> mosquito transmission rate (virus acquisition).
    mu_H : float
        Mosquito -> human transmission rate (human force of infection).
    alpha_H : float
        Recovery rate of infected humans.
    gamma : float
        Rate of loss of temporary immunity (R -> S).
    H : float
        Total human population (constant, S + I + R = H).
    eps : float
        Regularization of the cytoplasmic incompatibility ratio
        ``CI = N_M / (W_M + N_M + eps)``. See model.py.
    """

    rho_N: float
    alpha_N: float
    beta_N: float
    f: float
    rho_W: float
    alpha_W: float
    beta_W: float
    q: float
    mu_N: float
    mu_H: float
    alpha_H: float
    gamma: float
    H: float
    eps: float = 1e-12

    def __post_init__(self) -> None:
        positive_fields = (
            "rho_N",
            "alpha_N",
            "beta_N",
            "rho_W",
            "alpha_W",
            "beta_W",
            "mu_N",
            "mu_H",
            "alpha_H",
            "gamma",
            "H",
            "eps",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Parameters.{name} must be positive and finite, got {value!r}"
                )

        for name in ("f", "q"):
            value = getattr(self, name)
            if not (0.0 < value < 1.0):
                raise ValueError(
                    f"Parameters.{name} must be in (0, 1), got {value!r}"
                )

        if self.rho_N * self.f <= self.alpha_N:
            warnings.warn(
                "rho_N * f <= alpha_N: the wild population is not viable "
                "in the absence of Wolbachia (female natality does not "
                "exceed mortality). Wild P* would be <= 0.",
                stacklevel=2,
            )

        if self.rho_W * self.q <= self.alpha_W:
            warnings.warn(
                "rho_W * q <= alpha_W: the Wolbachia population is not "
                "viable on its own (female natality does not exceed "
                "mortality). Wolbachia P* would be <= 0.",
                stacklevel=2,
            )


@dataclass(frozen=True)
class NumericsConfig:
    """ODE integrator configuration."""

    method: str
    rtol: float
    atol: float

    def __post_init__(self) -> None:
        if self.method not in ("LSODA", "Radau"):
            raise ValueError(
                "numerics.method must be 'LSODA' or 'Radau' (the system is "
                f"stiff; odeint is not suitable), got {self.method!r}"
            )
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("numerics.rtol and numerics.atol must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete configuration of a run: parameters + initial state + numerics."""

    parameters: Parameters
    initial_state: np.ndarray
    numerics: NumericsConfig

    def __post_init__(self) -> None:
        if self.initial_state.shape != (len(STATE_VARS),):
            raise ValueError(
                f"initial_state must have shape ({len(STATE_VARS)},), "
                f"got {self.initial_state.shape!r}"
            )
        if np.any(self.initial_state < 0):
            raise ValueError("initial_state cannot contain negative values")


def _initial_state_from_dict(ic: dict[str, float]) -> np.ndarray:
    missing = [name for name in STATE_VARS if name not in ic]
    if missing:
        raise ValueError(f"initial_conditions incomplete, missing: {missing}")
    return np.array([float(ic[name]) for name in STATE_VARS], dtype=float)


def load_config(config_path: str | Path = _CONFIG_PATH_DEFAULT) -> SimulationConfig:
    """Load ``config/config.yaml`` and build the simulation configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to the configuration YAML.

    Returns
    -------
    SimulationConfig
        Configuration ready to pass to ``simulate.integrate``.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    mw = raw["mosquito_wild"]
    mwol = raw["mosquito_wolbachia"]
    hu = raw["human"]

    parameters = Parameters(
        rho_N=float(mw["rho_N"]),
        alpha_N=float(mw["alpha_N"]),
        beta_N=float(mw["beta_N"]),
        f=float(mw["f"]),
        rho_W=float(mwol["rho_W"]),
        alpha_W=float(mwol["alpha_W"]),
        beta_W=float(mwol["beta_W"]),
        q=float(mwol["q"]),
        mu_N=float(hu["mu_N"]),
        mu_H=float(hu["mu_H"]),
        alpha_H=float(hu["alpha_H"]),
        gamma=float(hu["gamma"]),
        H=float(hu["H"]),
    )

    numerics = NumericsConfig(
        method=raw["numerics"]["method"],
        rtol=float(raw["numerics"]["rtol"]),
        atol=float(raw["numerics"]["atol"]),
    )

    initial_state = _initial_state_from_dict(raw["initial_conditions"])

    return SimulationConfig(parameters=parameters, initial_state=initial_state, numerics=numerics)
