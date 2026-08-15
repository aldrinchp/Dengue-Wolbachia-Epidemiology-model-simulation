"""Integración numérica del modelo dengue-Wolbachia.

La liberación de Wolbachia de este proyecto ocurre una sola vez, en t=0
(reemplazo poblacional), así que se modela directamente como parte de la
condición inicial (se le suma la cantidad liberada a ``W_F``/``W_M`` antes
de integrar) — no hace falta partir la integración en tramos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from dengue_wolbachia.model import rhs
from dengue_wolbachia.parameters import NumericsConfig, Parameters, STATE_VARS

# Tolerancia de recorte de positividad: valores negativos de esta magnitud o
# menores se consideran ruido numérico del integrador y se recortan a 0.
# Valores más negativos indican un error real y deben fallar ruidosamente.
_POSITIVITY_CLIP_TOL = 1e-6


@dataclass(frozen=True)
class SimulationResult:
    """Resultado de una integración: tiempos y trayectorias de las 8 variables."""

    t: np.ndarray
    y: np.ndarray  # forma (8, n_tiempos), filas en el orden de STATE_VARS

    def __post_init__(self) -> None:
        if self.y.shape[0] != len(STATE_VARS):
            raise ValueError(
                f"y debe tener {len(STATE_VARS)} filas (una por variable de "
                f"estado), recibido {self.y.shape[0]}"
            )
        if self.y.shape[1] != self.t.shape[0]:
            raise ValueError("t e y deben tener el mismo número de columnas/tiempos")

    def as_dict(self) -> dict[str, np.ndarray]:
        """Devuelve las trayectorias como ``{"t": ..., "N_FS": ..., ...}``."""
        out: dict[str, np.ndarray] = {"t": self.t}
        for i, name in enumerate(STATE_VARS):
            out[name] = self.y[i]
        return out

    def final_state(self) -> np.ndarray:
        """Último vector de estado de la trayectoria."""
        return self.y[:, -1]


def _clip_or_raise_negatives(y: np.ndarray, tol: float = _POSITIVITY_CLIP_TOL) -> np.ndarray:
    """Recorta ruido numérico negativo; falla ruidosamente ante negativos apreciables.

    Ninguna de las 8 variables de estado tiene sentido biológico negativo.
    LSODA/Radau pueden producir residuos negativos del orden de ``1e-12``
    por redondeo; eso se recorta a 0. Un valor más negativo que ``tol``
    indica un error de integración o de modelo, no ruido, y debe
    propagarse como excepción en vez de esconderse.
    """
    min_val = float(np.min(y))
    if min_val < -tol:
        idx = np.unravel_index(np.argmin(y), y.shape)
        var_name = STATE_VARS[idx[0]]
        raise ValueError(
            f"Valor negativo apreciable detectado en '{var_name}': {min_val:.6g} "
            f"(tolerancia de recorte: {-tol:.1e}). Esto indica un problema real "
            "de integración o de modelo, no ruido numérico."
        )
    return np.clip(y, 0.0, None)


def integrate(
    params: Parameters,
    y0: np.ndarray,
    t_span: tuple[float, float],
    numerics: NumericsConfig,
    t_eval: np.ndarray | None = None,
) -> SimulationResult:
    """Integra el sistema de 8 EDOs.

    Parameters
    ----------
    params : Parameters
        Parámetros biológicos del modelo.
    y0 : np.ndarray
        Estado inicial, orden ``parameters.STATE_VARS``.
    t_span : tuple[float, float]
        ``(t0, tf)`` del intervalo a integrar.
    numerics : NumericsConfig
        Método (``"LSODA"`` o ``"Radau"``) y tolerancias ``rtol``/``atol``.
    t_eval : np.ndarray, optional
        Instantes en los que reportar la solución. Si es ``None``,
        ``solve_ivp`` elige su propia grilla adaptativa.

    Returns
    -------
    SimulationResult

    Raises
    ------
    RuntimeError
        Si el integrador no converge (``solve_ivp`` reporta ``success=False``).
    ValueError
        Si aparecen valores negativos apreciables (ver ``_clip_or_raise_negatives``).
    """
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method=numerics.method,
        args=(params,),
        t_eval=t_eval,
        rtol=numerics.rtol,
        atol=numerics.atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp no convergió: {sol.message}")

    y_clean = _clip_or_raise_negatives(sol.y)
    return SimulationResult(t=sol.t, y=y_clean)


def export_csv(result: SimulationResult, path: str | Path) -> None:
    """Exporta la trayectoria a CSV con columnas ``t, N_FS, N_FI, ..., R``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "t," + ",".join(STATE_VARS)
    data = np.vstack([result.t, result.y])
    np.savetxt(path, data.T, delimiter=",", header=header, comments="")
