"""Derivatives of the 9-ODE dengue-Wolbachia system (8 state variables + 1
cumulative incidence variable).

This module contains a single public function, :func:`rhs`, with no global
state: it receives ``params`` explicitly (so it can be used as ``args`` of
``scipy.integrate.solve_ivp``) and does not depend on any external mutable
value.

Model assumptions documented here because they are deliberate omissions or
simplifications, not oversights:

1. **Viral blocking.** Wolbachia-carrying females (``W_F``) do not transmit
   the dengue virus: there is no ``W_FI`` compartment and ``W_F`` does not
   appear in ``dI/dt``. Wolbachia blocks viral replication inside the
   mosquito, so a carrier female never becomes infectious.
2. **Cytoplasmic incompatibility (CI).** When a wild female mates with a
   Wolbachia-carrying male, the offspring are not viable.
   ``CI = N_M / (W_M + N_M)`` is the fraction of compatible matings;
   that's why it multiplies wild natality. If ``W_M >> N_M``, ``CI -> 0``
   and wild natality collapses: this is the IIT mechanism.
3. **Maternal inheritance.** Wolbachia is transmitted only from mother to
   offspring: the equations for ``W_F`` and ``W_M`` depend solely on
   ``W_F`` (never on ``W_M`` as a parent) and carry no compatibility
   factor, because a carrier female always produces viable carrier
   offspring.
4. **Shared density-dependent competition.** The ``beta*X*P`` terms
   are additional mortality proportional to the total mosquito density
   ``P`` (wild + carriers): they share the same pool of larval resources,
   so one population competes with the other for space/food.
5. **Closed human population.** There is no human natality or mortality in
   the model: ``d(S+I+R)/dt = 0`` by construction (verified in
   ``tests/test_simulate.py``).
6. **Cumulative incidence (``C``).** ``I`` is prevalence (how many are
   infected at instant ``t``: it goes up and down because there is inflow
   and outflow). ``C`` is cumulative incidence: only inflow, nobody ever
   leaves ``C``. Both share the same inflow term
   (``mu_H*S*N_FI``), but ``dC/dt`` has no ``-alpha_H*I`` recovery term,
   so ``C`` is monotonically increasing and ``C(t) >= I(t)`` always — it
   is the count of "total cases since day 0", as in an epidemiological
   surveillance report.
"""

from __future__ import annotations

import numpy as np

from dengue_wolbachia.parameters import Parameters


def compatibility_index(N_M: float, W_M: float, eps: float) -> float:
    """Fraction of compatible matings (cytoplasmic incompatibility).

    ``CI = N_M / (W_M + N_M)``: the probability that a wild female mates
    with a wild male (compatible) instead of a Wolbachia-carrying one
    (incompatible, producing non-viable offspring).

    Parameters
    ----------
    N_M : float
        Wild males.
    W_M : float
        Wolbachia-carrying males.
    eps : float
        Regularization to avoid the 0/0 indeterminacy when there are no
        males of either kind.

    Returns
    -------
    float
        ``CI`` in ``[0, 1]``. Returns 0 if the regularized denominator is
        smaller than ``eps`` (no males: there can be no mating).
    """
    denom = W_M + N_M
    if denom < eps:
        return 0.0
    return N_M / (denom + eps)


def rhs(t: float, y: np.ndarray, params: Parameters) -> np.ndarray:
    """Derivatives of the dengue-Wolbachia system at instant ``t``.

    Signature compatible with ``scipy.integrate.solve_ivp(fun=rhs, ..., args=(params,))``.

    Parameters
    ----------
    t : float
        Time (not used explicitly: the system is autonomous, but
        ``solve_ivp`` always passes ``t`` as the first argument).
    y : np.ndarray
        9-component state vector in the order
        ``(N_FS, N_FI, N_M, W_F, W_M, S, I, R, C)`` (see ``parameters.STATE_VARS``).
    params : Parameters
        Biological parameters of the model.

    Returns
    -------
    np.ndarray
        Derivative vector ``dy/dt``, same order and shape as ``y``.
    """
    N_FS, N_FI, N_M, W_F, W_M, S, I, R, C = y

    P = N_FS + N_FI + N_M + W_F + W_M
    CI = compatibility_index(N_M, W_M, params.eps)

    N_F = N_FI + N_FS  # total wild females (susceptible + infected)

    dN_FS = (
        params.rho_N * params.f * N_F * CI
        - params.alpha_N * N_FS
        - params.beta_N * N_FS * P
        - params.mu_N * N_FS * I
    )
    dN_FI = params.mu_N * N_FS * I - params.alpha_N * N_FI - params.beta_N * N_FI * P
    dN_M = (
        params.rho_N * (1.0 - params.f) * N_F * CI
        - params.alpha_N * N_M
        - params.beta_N * N_M * P
    )

    # W_F carries no CI factor: pure maternal inheritance (assumption 3).
    dW_F = params.rho_W * params.q * W_F - params.alpha_W * W_F - params.beta_W * W_F * P
    dW_M = (
        params.rho_W * (1.0 - params.q) * W_F
        - params.alpha_W * W_M
        - params.beta_W * W_M * P
    )

    # W_F does not appear here: viral blocking (assumption 1), no W_FI compartment.
    incidence = params.mu_H * S * N_FI
    dS = params.gamma * R - incidence
    dI = incidence - params.alpha_H * I
    dR = params.alpha_H * I - params.gamma * R

    # C accumulates the same inflow as I, but without the recovery term
    # (assumption 6): cumulative incidence, never decreases.
    dC = incidence

    return np.array([dN_FS, dN_FI, dN_M, dW_F, dW_M, dS, dI, dR, dC], dtype=float)
