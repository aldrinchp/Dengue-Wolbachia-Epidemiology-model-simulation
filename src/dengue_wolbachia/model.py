"""Derivadas del sistema de 9 EDOs dengue-Wolbachia (8 de estado + 1 de
incidencia acumulada).

Este módulo contiene una única función pública, :func:`rhs`, sin estado
global: recibe ``params`` de forma explícita (para usar como ``args`` de
``scipy.integrate.solve_ivp``) y no depende de ningún valor mutable externo.

Supuestos del modelo que se documentan aquí porque son ausencias o
simplificaciones deliberadas, no descuidos:

1. **Bloqueo viral.** Las hembras con Wolbachia (``W_F``) no transmiten el
   virus del dengue: no existe un compartimento ``W_FI`` y ``W_F`` no
   aparece en ``dI/dt``. Wolbachia bloquea la replicación viral dentro del
   mosquito, así que una hembra portadora nunca se vuelve infecciosa.
2. **Incompatibilidad citoplasmática (CI).** Cuando una hembra silvestre se
   apareacon un macho portador de Wolbachia, la descendencia no es viable.
   ``CI = N_M / (W_M + N_M)`` es la fracción de apareamientos compatibles;
   por eso multiplica la natalidad silvestre. Si ``W_M >> N_M``, ``CI -> 0``
   y la natalidad silvestre colapsa: este es el mecanismo de IIT.
3. **Herencia materna.** Wolbachia se transmite solo de madre a hijos: las
   ecuaciones de ``W_F`` y ``W_M`` dependen únicamente de ``W_F`` (nunca de
   ``W_M`` como progenitor) y no llevan factor de compatibilidad, porque una
   hembra portadora siempre produce descendencia portadora viable.
4. **Competencia denso-dependiente compartida.** Los términos ``beta*X*P``
   son mortalidad adicional proporcional a la densidad total de mosquitos
   ``P`` (silvestres + portadores): comparten el mismo pozo de recursos
   larvarios, así que una población compite con la otra por espacio/comida.
5. **Población humana cerrada.** No hay natalidad ni mortalidad humana en el
   modelo: ``d(S+I+R)/dt = 0`` por construcción (verificado en
   ``tests/test_simulate.py``).
6. **Incidencia acumulada (``C``).** ``I`` es prevalencia (cuántos están
   infectados en el instante ``t``: sube y baja porque hay entrada y
   salida). ``C`` es incidencia acumulada: solo entrada, nunca sale nadie de
   ``C``. Ambas comparten el mismo término de entrada
   (``mu_H*S*N_FI``), pero ``dC/dt`` no tiene el término ``-alpha_H*I`` de
   recuperación, así que ``C`` es monótonamente creciente y ``C(t) >= I(t)``
   siempre — es el conteo de "casos totales desde el día 0", como en un
   reporte de vigilancia epidemiológica.
"""

from __future__ import annotations

import numpy as np

from dengue_wolbachia.parameters import Parameters


def compatibility_index(N_M: float, W_M: float, eps: float) -> float:
    """Fracción de apareamientos compatibles (incompatibilidad citoplasmática).

    ``CI = N_M / (W_M + N_M)``: la probabilidad de que una hembra silvestre
    se aparee con un macho silvestre (compatible) en lugar de uno portador
    de Wolbachia (incompatible, produce descendencia inviable).

    Parameters
    ----------
    N_M : float
        Machos silvestres.
    W_M : float
        Machos portadores de Wolbachia.
    eps : float
        Regularización para evitar la indeterminación 0/0 cuando no hay
        machos de ningún tipo.

    Returns
    -------
    float
        ``CI`` en ``[0, 1]``. Devuelve 0 si el denominador regularizado es
        menor que ``eps`` (no hay machos: no puede haber apareamiento).
    """
    denom = W_M + N_M
    if denom < eps:
        return 0.0
    return N_M / (denom + eps)


def rhs(t: float, y: np.ndarray, params: Parameters) -> np.ndarray:
    """Derivadas del sistema dengue-Wolbachia en el instante ``t``.

    Firma compatible con ``scipy.integrate.solve_ivp(fun=rhs, ..., args=(params,))``.

    Parameters
    ----------
    t : float
        Tiempo (no usado explícitamente: el sistema es autónomo, pero
        ``solve_ivp`` siempre pasa ``t`` como primer argumento).
    y : np.ndarray
        Vector de estado de 9 componentes en el orden
        ``(N_FS, N_FI, N_M, W_F, W_M, S, I, R, C)`` (ver ``parameters.STATE_VARS``).
    params : Parameters
        Parámetros biológicos del modelo.

    Returns
    -------
    np.ndarray
        Vector de derivadas ``dy/dt``, mismo orden y forma que ``y``.
    """
    N_FS, N_FI, N_M, W_F, W_M, S, I, R, C = y

    P = N_FS + N_FI + N_M + W_F + W_M
    CI = compatibility_index(N_M, W_M, params.eps)

    N_F = N_FI + N_FS  # total de hembras silvestres (susceptibles + infectadas)

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

    # W_F no lleva factor CI: herencia materna pura (supuesto 3).
    dW_F = params.rho_W * params.q * W_F - params.alpha_W * W_F - params.beta_W * W_F * P
    dW_M = (
        params.rho_W * (1.0 - params.q) * W_F
        - params.alpha_W * W_M
        - params.beta_W * W_M * P
    )

    # W_F no aparece aquí: bloqueo viral (supuesto 1), no hay compartimento W_FI.
    incidence = params.mu_H * S * N_FI
    dS = params.gamma * R - incidence
    dI = incidence - params.alpha_H * I
    dR = params.alpha_H * I - params.gamma * R

    # C acumula la misma entrada que I, pero sin el término de recuperación
    # (supuesto 6): incidencia acumulada, nunca decrece.
    dC = incidence

    return np.array([dN_FS, dN_FI, dN_M, dW_F, dW_M, dS, dI, dR, dC], dtype=float)
