"""Equilibrio silvestre analítico y número reproductivo básico del dengue.

Es la referencia de verificación del modelo: el test de convergencia en
``tests/test_simulate.py`` integra desde una condición arbitraria y
comprueba que converge a ``wild_equilibrium_P_star`` con tolerancia relativa
de 1e-6.
"""

from __future__ import annotations

from dengue_wolbachia.parameters import Parameters


def wild_equilibrium_P_star(params: Parameters) -> float:
    """Densidad total de equilibrio silvestre, ``P* = (rho_N*f - alpha_N)/beta_N``.

    Resultado cerrado y exacto de igualar ``dP/dt = 0`` con ``W_F=W_M=0`` y
    sin dengue (``CI=1``, incompatibilidad citoplasmática irrelevante sin
    machos portadores).

    Raises
    ------
    ValueError
        Si la población silvestre no persiste (``rho_N*f <= alpha_N``): el
        equilibrio no positivo no tiene sentido biológico.
    """
    if params.rho_N * params.f <= params.alpha_N:
        raise ValueError(
            "La población silvestre no persiste (rho_N*f <= alpha_N); "
            "P* silvestre no está definido (sería <= 0)."
        )
    return (params.rho_N * params.f - params.alpha_N) / params.beta_N


def wild_equilibrium(params: Parameters) -> tuple[float, float, float]:
    """Equilibrio silvestre puro desglosado: ``(N_FS*, N_M*, P*)``.

    Con ``CI=1`` la relación ``dN_FS/dt=0`` fuerza ``N_FS* = f*P*`` y
    ``N_M* = (1-f)*P*`` (se deriva de combinar las ecuaciones de ``N_FS`` y
    ``N_M`` en equilibrio).
    """
    P_star = wild_equilibrium_P_star(params)
    return params.f * P_star, (1.0 - params.f) * P_star, P_star


def next_generation_number(
    params: Parameters, N_FS_star: float, P_star: float, S_star: float | None = None
) -> float:
    """``R0^NG``, radio espectral de la matriz de próxima generación ``F*V^-1``.

    ``R0^NG = sqrt( mu_H*S* * mu_N*N_FS* / (alpha_H*(alpha_N+beta_N*P*)) )``,
    obtenido por matriz de próxima generación (van den Driessche & Watmough,
    2002) sobre los compartimentos infectados ``(I, N_FI)``.

    Esta cantidad describe una única etapa del ciclo de transmisión
    (huésped -> vector): la estructura fuera-de-diagonal de la matriz de
    nuevas infecciones implica que un humano infectado nunca genera nuevos
    humanos infectados directamente, sino a través de un mosquito
    intermedio. Por eso **no** es el número reproductivo básico final —
    ver :func:`basic_reproduction_number`.

    Parameters
    ----------
    params : Parameters
        Parámetros del modelo.
    N_FS_star : float
        Hembras silvestres susceptibles en el equilibrio libre de enfermedad.
    P_star : float
        Densidad total de mosquitos en ese mismo equilibrio.
    S_star : float, optional
        Humanos susceptibles en el equilibrio; por defecto ``params.H``
        (todos los humanos susceptibles, sin dengue previo).

    Returns
    -------
    float
        ``R0^NG``. Devuelve 0.0 si ``N_FS_star <= 0`` (no hay mosquitos
        silvestres susceptibles: no puede haber transmisión — es lo que pasa
        cuando el reemplazo poblacional con Wolbachia tuvo éxito).
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
    """``R0``, número reproductivo básico evaluado en el equilibrio libre de enfermedad.

    ``R0 = (R0^NG)^2``: el ciclo de transmisión del dengue es huésped ->
    vector -> huésped, así que el número esperado de casos humanos
    secundarios por un único humano infectado (en una población totalmente
    susceptible) requiere cerrar el ciclo completo, no solo una etapa.
    Siguiendo a Gibbs et al. (2026) para el mismo modelo estructurado por
    sexo, ``R0`` se obtiene elevando al cuadrado :func:`next_generation_number`.

    Parameters
    ----------
    params : Parameters
        Parámetros del modelo.
    N_FS_star : float
        Hembras silvestres susceptibles en el equilibrio libre de enfermedad.
    P_star : float
        Densidad total de mosquitos en ese mismo equilibrio.
    S_star : float, optional
        Humanos susceptibles en el equilibrio; por defecto ``params.H``
        (todos los humanos susceptibles, sin dengue previo).

    Returns
    -------
    float
        ``R0``. Devuelve 0.0 si ``N_FS_star <= 0`` (no hay mosquitos
        silvestres susceptibles: no puede haber transmisión — es lo que pasa
        cuando el reemplazo poblacional con Wolbachia tuvo éxito).
    """
    return next_generation_number(params, N_FS_star, P_star, S_star) ** 2
