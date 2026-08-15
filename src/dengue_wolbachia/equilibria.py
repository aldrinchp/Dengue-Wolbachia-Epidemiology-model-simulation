"""Analytic wild equilibrium and basic reproduction number of dengue.

This is the model's verification reference: the convergence test in
``tests/test_simulate.py`` integrates from an arbitrary condition and
checks that it converges to ``wild_equilibrium_P_star`` with a relative
tolerance of 1e-6.
"""

from __future__ import annotations

from dengue_wolbachia.parameters import Parameters


def wild_equilibrium_P_star(params: Parameters) -> float:
    """Total wild equilibrium density, ``P* = (rho_N*f - alpha_N)/beta_N``.

    Closed-form exact result from setting ``dP/dt = 0`` with ``W_F=W_M=0``
    and no dengue (``CI=1``, cytoplasmic incompatibility irrelevant
    without carrier males).

    Raises
    ------
    ValueError
        If the wild population does not persist (``rho_N*f <= alpha_N``):
        the non-positive equilibrium has no biological meaning.
    """
    if params.rho_N * params.f <= params.alpha_N:
        raise ValueError(
            "The wild population does not persist (rho_N*f <= alpha_N); "
            "wild P* is undefined (it would be <= 0)."
        )
    return (params.rho_N * params.f - params.alpha_N) / params.beta_N


def wild_equilibrium(params: Parameters) -> tuple[float, float, float]:
    """Pure wild equilibrium, broken down: ``(N_FS*, N_M*, P*)``.

    With ``CI=1`` the relation ``dN_FS/dt=0`` forces ``N_FS* = f*P*`` and
    ``N_M* = (1-f)*P*`` (derived by combining the ``N_FS`` and ``N_M``
    equations at equilibrium).
    """
    P_star = wild_equilibrium_P_star(params)
    return params.f * P_star, (1.0 - params.f) * P_star, P_star


def next_generation_number(
    params: Parameters, N_FS_star: float, P_star: float, S_star: float | None = None
) -> float:
    """``R0^NG``, spectral radius of the next-generation matrix ``F*V^-1``.

    ``R0^NG = sqrt( mu_H*S* * mu_N*N_FS* / (alpha_H*(alpha_N+beta_N*P*)) )``,
    obtained via the next-generation matrix (van den Driessche & Watmough,
    2002) over the infected compartments ``(I, N_FI)``.

    This quantity describes a single stage of the transmission cycle
    (host -> vector): the off-diagonal structure of the new-infections
    matrix implies that an infected human never generates new infected
    humans directly, only through an intermediate mosquito. That's why it
    is **not** the final basic reproduction number —
    see :func:`basic_reproduction_number`.

    Parameters
    ----------
    params : Parameters
        Model parameters.
    N_FS_star : float
        Susceptible wild females at the disease-free equilibrium.
    P_star : float
        Total mosquito density at that same equilibrium.
    S_star : float, optional
        Susceptible humans at equilibrium; defaults to ``params.H``
        (all humans susceptible, no prior dengue).

    Returns
    -------
    float
        ``R0^NG``. Returns 0.0 if ``N_FS_star <= 0`` (no susceptible wild
        mosquitoes: there can be no transmission — this is what happens
        when Wolbachia population replacement has succeeded).
    """
    if N_FS_star <= 0:
        return 0.0
    if S_star is None:
        S_star = params.H
    denom = params.alpha_H * (params.alpha_N + params.beta_N * P_star)
    return (params.mu_H * S_star * params.mu_N * N_FS_star / denom) ** 0.5


def basic_reproduction_number(
    params: Parameters, N_FS_star: float, P_star: float, S_star: float | None = None
) -> float:
    """``R0``, basic reproduction number evaluated at the disease-free equilibrium.

    ``R0 = (R0^NG)^2``: the dengue transmission cycle is host ->
    vector -> host, so the expected number of secondary human cases
    caused by a single infected human (in a fully susceptible population)
    requires closing the full cycle, not just one stage.
    Following Gibbs et al. (2026) for the same sex-structured model,
    ``R0`` is obtained by squaring :func:`next_generation_number`.

    Parameters
    ----------
    params : Parameters
        Model parameters.
    N_FS_star : float
        Susceptible wild females at the disease-free equilibrium.
    P_star : float
        Total mosquito density at that same equilibrium.
    S_star : float, optional
        Susceptible humans at equilibrium; defaults to ``params.H``
        (all humans susceptible, no prior dengue).

    Returns
    -------
    float
        ``R0``. Returns 0.0 if ``N_FS_star <= 0`` (no susceptible wild
        mosquitoes: there can be no transmission — this is what happens
        when Wolbachia population replacement has succeeded).
    """
    return next_generation_number(params, N_FS_star, P_star, S_star) ** 2
