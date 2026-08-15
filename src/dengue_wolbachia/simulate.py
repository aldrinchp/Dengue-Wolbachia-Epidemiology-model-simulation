"""Numerical integration of the dengue-Wolbachia model.

The Wolbachia release in this project happens exactly once, at t=0
(population replacement), so it is modeled directly as part of the
initial condition (the released amount is added to ``W_F``/``W_M`` before
integrating) — there is no need to split the integration into segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from dengue_wolbachia.model import rhs
from dengue_wolbachia.parameters import NumericsConfig, Parameters, STATE_VARS

# Positivity clipping tolerance: negative values of this magnitude or
# smaller are considered integrator numerical noise and are clipped to 0.
# More negative values indicate a real error and must fail loudly.
_POSITIVITY_CLIP_TOL = 1e-6


@dataclass(frozen=True)
class SimulationResult:
    """Result of an integration: time points and trajectories of the 8 variables."""

    t: np.ndarray
    y: np.ndarray  # shape (8, n_times), rows in STATE_VARS order

    def __post_init__(self) -> None:
        if self.y.shape[0] != len(STATE_VARS):
            raise ValueError(
                f"y must have {len(STATE_VARS)} rows (one per state "
                f"variable), got {self.y.shape[0]}"
            )
        if self.y.shape[1] != self.t.shape[0]:
            raise ValueError("t and y must have the same number of columns/times")

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the trajectories as ``{"t": ..., "N_FS": ..., ...}``."""
        out: dict[str, np.ndarray] = {"t": self.t}
        for i, name in enumerate(STATE_VARS):
            out[name] = self.y[i]
        return out

    def final_state(self) -> np.ndarray:
        """Last state vector of the trajectory."""
        return self.y[:, -1]


def _clip_or_raise_negatives(y: np.ndarray, tol: float = _POSITIVITY_CLIP_TOL) -> np.ndarray:
    """Clip negative numerical noise; fail loudly on appreciable negatives.

    None of the 8 state variables has a biologically meaningful negative
    value. LSODA/Radau can produce negative residuals on the order of
    ``1e-12`` from rounding; those are clipped to 0. A value more negative
    than ``tol`` indicates an integration or model error, not noise, and
    must propagate as an exception instead of being hidden.
    """
    min_val = float(np.min(y))
    if min_val < -tol:
        idx = np.unravel_index(np.argmin(y), y.shape)
        var_name = STATE_VARS[idx[0]]
        raise ValueError(
            f"Appreciable negative value detected in '{var_name}': {min_val:.6g} "
            f"(clip tolerance: {-tol:.1e}). This indicates a real "
            "integration or model problem, not numerical noise."
        )
    return np.clip(y, 0.0, None)


def integrate(
    params: Parameters,
    y0: np.ndarray,
    t_span: tuple[float, float],
    numerics: NumericsConfig,
    t_eval: np.ndarray | None = None,
) -> SimulationResult:
    """Integrate the 8-ODE system.

    Parameters
    ----------
    params : Parameters
        Biological parameters of the model.
    y0 : np.ndarray
        Initial state, in ``parameters.STATE_VARS`` order.
    t_span : tuple[float, float]
        ``(t0, tf)`` of the interval to integrate.
    numerics : NumericsConfig
        Method (``"LSODA"`` or ``"Radau"``) and ``rtol``/``atol`` tolerances.
    t_eval : np.ndarray, optional
        Instants at which to report the solution. If ``None``,
        ``solve_ivp`` chooses its own adaptive grid.

    Returns
    -------
    SimulationResult

    Raises
    ------
    RuntimeError
        If the integrator does not converge (``solve_ivp`` reports ``success=False``).
    ValueError
        If appreciable negative values appear (see ``_clip_or_raise_negatives``).
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
        raise RuntimeError(f"solve_ivp did not converge: {sol.message}")

    y_clean = _clip_or_raise_negatives(sol.y)
    return SimulationResult(t=sol.t, y=y_clean)


def export_csv(result: SimulationResult, path: str | Path) -> None:
    """Export the trajectory to CSV with columns ``t, N_FS, N_FI, ..., R``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "t," + ",".join(STATE_VARS)
    data = np.vstack([result.t, result.y])
    np.savetxt(path, data.T, delimiter=",", header=header, comments="")
