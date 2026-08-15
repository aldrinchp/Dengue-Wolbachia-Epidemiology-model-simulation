"""Carga y validación de parámetros del modelo dengue-Wolbachia.

Los parámetros biológicos viven en :class:`Parameters`, una dataclass
inmutable con validación en ``__post_init__``. La configuración de
simulación (condición inicial, tolerancias del integrador) vive en
:class:`SimulationConfig`, construida a partir de ``config/config.yaml``
mediante :func:`load_config`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

# Orden fijo de las 9 variables de estado (8 de estado + C = incidencia
# acumulada). Todo el paquete (model.py, simulate.py, plots.py) debe
# construir/leer vectores de estado en este orden para evitar errores de
# indexación silenciosos.
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
    """Parámetros biológicos del sistema de 8 EDOs.

    Todas las tasas están en unidades de día⁻¹ (o el análogo per-cápita
    correspondiente).

    Parameters
    ----------
    rho_N : float
        Fecundidad per cápita de hembras silvestres.
    alpha_N : float
        Mortalidad per cápita de mosquitos silvestres (ambos sexos).
    beta_N : float
        Coeficiente de competencia denso-dependiente, silvestres.
    f : float
        Proporción de hembras al nacer en la población silvestre, en (0, 1).
    rho_W : float
        Fecundidad per cápita de hembras portadoras de Wolbachia.
    alpha_W : float
        Mortalidad per cápita de mosquitos portadores de Wolbachia.
    beta_W : float
        Coeficiente de competencia denso-dependiente, portadores.
    q : float
        Proporción de hembras al nacer en la población con Wolbachia, en (0, 1).
    mu_N : float
        Tasa de transmisión humano -> mosquito (adquisición del virus).
    mu_H : float
        Tasa de transmisión mosquito -> humano (fuerza de infección humana).
    alpha_H : float
        Tasa de recuperación de humanos infectados.
    gamma : float
        Tasa de pérdida de inmunidad temporal (R -> S).
    H : float
        Población humana total (constante, S + I + R = H).
    eps : float
        Regularización del cociente de compatibilidad citoplasmática
        ``CI = N_M / (W_M + N_M + eps)``. Ver model.py.
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
                    f"Parameters.{name} debe ser positivo y finito, recibido {value!r}"
                )

        for name in ("f", "q"):
            value = getattr(self, name)
            if not (0.0 < value < 1.0):
                raise ValueError(
                    f"Parameters.{name} debe estar en (0, 1), recibido {value!r}"
                )

        if self.rho_N * self.f <= self.alpha_N:
            warnings.warn(
                "rho_N * f <= alpha_N: la población silvestre no es viable "
                "en ausencia de Wolbachia (natalidad de hembras no supera "
                "la mortalidad). P* silvestre sería <= 0.",
                stacklevel=2,
            )

        if self.rho_W * self.q <= self.alpha_W:
            warnings.warn(
                "rho_W * q <= alpha_W: la población con Wolbachia no es "
                "viable de forma autónoma (natalidad de hembras no supera "
                "la mortalidad). P* con Wolbachia sería <= 0.",
                stacklevel=2,
            )


@dataclass(frozen=True)
class NumericsConfig:
    """Configuración del integrador ODE."""

    method: str
    rtol: float
    atol: float

    def __post_init__(self) -> None:
        if self.method not in ("LSODA", "Radau"):
            raise ValueError(
                "numerics.method debe ser 'LSODA' o 'Radau' (el sistema es "
                f"rígido; odeint no es apto), recibido {self.method!r}"
            )
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("numerics.rtol y numerics.atol deben ser positivos")


@dataclass(frozen=True)
class SimulationConfig:
    """Configuración completa de una corrida: parámetros + estado inicial + numérica."""

    parameters: Parameters
    initial_state: np.ndarray
    numerics: NumericsConfig

    def __post_init__(self) -> None:
        if self.initial_state.shape != (len(STATE_VARS),):
            raise ValueError(
                f"initial_state debe tener forma ({len(STATE_VARS)},), "
                f"recibido {self.initial_state.shape!r}"
            )
        if np.any(self.initial_state < 0):
            raise ValueError("initial_state no puede contener valores negativos")


def _initial_state_from_dict(ic: dict[str, float]) -> np.ndarray:
    missing = [name for name in STATE_VARS if name not in ic]
    if missing:
        raise ValueError(f"initial_conditions incompleto, faltan: {missing}")
    return np.array([float(ic[name]) for name in STATE_VARS], dtype=float)


def load_config(config_path: str | Path = _CONFIG_PATH_DEFAULT) -> SimulationConfig:
    """Carga ``config/config.yaml`` y construye la configuración de simulación.

    Parameters
    ----------
    config_path : str or Path
        Ruta al YAML de configuración.

    Returns
    -------
    SimulationConfig
        Configuración lista para pasar a ``simulate.integrate``.
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
