# dengue-wolbachia

Modelo epidemiológico de dengue con control biológico por *Wolbachia*: un
sistema de 8 ecuaciones diferenciales ordinarias que acopla mosquitos
silvestres, mosquitos portadores de *Wolbachia* y humanos con dinámica SIRS.

Este proyecto simula **un solo caso**: la línea base (sin control) contra
una liberación única de mosquitos wMel (reemplazo poblacional) en $t=0$,
para mostrar cómo *Wolbachia* actúa como control biológico del dengue.

> **Nota sobre los parámetros.** Todos los valores numéricos en
> `config/config.yaml` son **PROVISIONALES** (órdenes de magnitud de
> literatura general sobre *Aedes aegypti*, con la unidad de tiempo en
> días) y están marcados explícitamente como tales en el archivo.
> Reemplázalos por los valores de tu referencia bibliográfica antes de usar
> los resultados en un informe.

## Instalación

Requiere Python ≥ 3.10.

```bash
cd dengue-wolbachia
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Sin dependencias más allá de `numpy`, `scipy`, `matplotlib`, `pyyaml` y
`pytest` (declaradas en `pyproject.toml`).

## Formulación matemática

### Variables de estado

| Variable | Significado |
|---|---|
| $N_{FS}$ | hembras silvestres susceptibles al dengue |
| $N_{FI}$ | hembras silvestres infectadas de dengue |
| $N_M$    | machos silvestres |
| $W_F$    | hembras portadoras de *Wolbachia* |
| $W_M$    | machos portadores de *Wolbachia* |
| $S$      | humanos susceptibles |
| $I$      | humanos infectados |
| $R$      | humanos recuperados (inmunidad temporal) |

con las cantidades auxiliares

$$
P = N_{FS}+N_{FI}+N_M+W_F+W_M, \qquad
H = S+I+R, \qquad
CI = \frac{N_M}{W_M+N_M}.
$$

### Sistema de EDOs

$$
\begin{aligned}
\dot N_{FS} &= \rho_N f (N_{FI}+N_{FS})\,CI - \alpha_N N_{FS} - \beta_N N_{FS} P - \mu_N N_{FS} I \\
\dot N_{FI} &= \mu_N N_{FS} I - \alpha_N N_{FI} - \beta_N N_{FI} P \\
\dot N_M &= \rho_N (1-f)(N_{FI}+N_{FS})\,CI - \alpha_N N_M - \beta_N N_M P \\
\dot W_F &= \rho_W q\, W_F - \alpha_W W_F - \beta_W W_F P \\
\dot W_M &= \rho_W (1-q)\, W_F - \alpha_W W_M - \beta_W W_M P \\
\dot S &= \gamma R - \mu_H S N_{FI} \\
\dot I &= \mu_H S N_{FI} - \alpha_H I \\
\dot R &= \alpha_H I - \gamma R
\end{aligned}
$$

implementado en [`src/dengue_wolbachia/model.py`](src/dengue_wolbachia/model.py)
como la función pura `rhs(t, y, params)`, sin estado global, lista para usar
como `args` de `scipy.integrate.solve_ivp`.

### Supuestos deliberados (ver también los comentarios en `model.py`)

1. **Bloqueo viral.** $W_F$ no aparece en $\dot I$: las hembras con
   *Wolbachia* no transmiten dengue, así que no existe compartimento
   $W_{FI}$.
2. **Incompatibilidad citoplasmática (CI).** $CI=N_M/(W_M+N_M)$ es la
   fracción de apareamientos compatibles. Si $W_M \gg N_M$, $CI\to 0$ y la
   natalidad silvestre colapsa — es el mecanismo por el cual una liberación
   suficientemente grande hace que *Wolbachia* desplace a los silvestres.
3. **Herencia materna.** $\dot W_F$ y $\dot W_M$ dependen solo de $W_F$, sin
   factor $CI$: *Wolbachia* se hereda únicamente de la madre. Por eso no hay
   ninguna ecuación que convierta un mosquito silvestre en portador — la
   única forma de que aparezcan portadores es que nazcan de una madre
   portadora, o que se liberen desde afuera del sistema.
4. **Competencia denso-dependiente compartida.** Los términos $\beta X P$
   son mortalidad adicional proporcional a la densidad total $P$: silvestres
   y portadores comparten el mismo pozo de recursos larvarios.
5. **Población humana cerrada.** $\dot S+\dot I+\dot R=0$ por construcción
   (sin natalidad/mortalidad humana en el modelo).

### Detalles numéricos

- $CI$ se regulariza como `CI = N_M / (W_M + N_M + eps)` con `eps=1e-12`.
- Integración con `scipy.integrate.solve_ivp`, método `LSODA` o `Radau`
  (nunca `odeint`): el sistema es rígido porque las tasas de mosquito
  (escala de días) convive con la pérdida de inmunidad (escala de meses).
  `rtol=1e-8`, `atol=1e-10` por defecto, configurables en `config/config.yaml`.
- Positividad: valores negativos de orden $10^{-12}$ (ruido del integrador)
  se recortan a 0; valores negativos apreciables (> `1e-6`) hacen fallar la
  integración con un `ValueError` explícito en vez de esconderse.

### El control: liberación única en $t=0$

La liberación de *Wolbachia* ocurre una sola vez, antes de que arranque la
simulación, así que se modela directamente como parte de la condición
inicial (se le suma la cantidad liberada a $W_F(0)$/$W_M(0)$) — no hace
falta partir la integración en tramos ni ningún término de liberación
dentro de las ecuaciones.

### Número reproductivo básico ($R_0$)

$$R_0=\sqrt{\dfrac{\mu_H S^{*}\,\mu_N N_{FS}^{*}}{\alpha_H(\alpha_N+\beta_N P^{*})}},$$

evaluado en el equilibrio silvestre libre de enfermedad (línea base, sin
control), por matriz de próxima generación sobre $(I,N_{FI})$. Implementado
en [`equilibria.py`](src/dengue_wolbachia/equilibria.py).

## Estructura del proyecto

```
dengue-wolbachia/
├── config/config.yaml       # todos los parámetros + condición inicial
├── src/dengue_wolbachia/
│   ├── model.py              # derivadas del sistema (rhs)
│   ├── parameters.py         # Parameters, SimulationConfig, load_config
│   ├── simulate.py           # integración (solve_ivp) + exportación CSV
│   ├── equilibria.py         # equilibrio silvestre analítico y R0
│   └── plots.py              # figuras (matplotlib puro, etiquetas en inglés)
├── scripts/
│   └── run_baseline_vs_control.py   # el único script: corre el caso completo
├── tests/                    # pytest
└── outputs/{data,figures}/   # CSV y PNG/PDF generados
```

## Uso

```bash
python scripts/run_baseline_vs_control.py
```

Esto corre las dos simulaciones (línea base y control), imprime $R_0$ y los
valores finales de infectados en consola, y exporta:

- `outputs/data/{baseline,control}.csv` — trayectorias completas de las 9 variables (incluye `C`, incidencia acumulada)
- `outputs/data/summary.csv` — $R_0$, estado final y casos acumulados/evitados (el único lugar donde $R_0$ queda guardado, no solo impreso en consola)
- `outputs/figures/comparison.{png,pdf}` — $I(t)$: línea base vs. control, en un solo eje
- `outputs/figures/baseline_timeseries.{png,pdf}` — paneles apilados de la línea base
- `outputs/figures/control_timeseries.{png,pdf}` — paneles apilados del caso con control

Parámetros del experimento ajustables por línea de comandos:

```bash
python scripts/run_baseline_vs_control.py --release-fraction 0.5 --t-final 1000
```

- `--release-fraction`: fracción de la densidad de equilibrio silvestre
  ($P^*$) liberada como $W_F$ y $W_M$ en $t=0$ (por defecto 0.35 — ya
  comprobado que basta para que *Wolbachia* se establezca con estos
  parámetros PROVISIONALES).
- `--t-final`: duración de la simulación en días (por defecto 200).

## Resultado esperado

Con los parámetros PROVISIONALES por defecto: $R_0\approx2.86$ sin control
(el dengue se sostiene solo). Liberando el 35% de la población silvestre de
equilibrio como mosquitos wMel en $t=0$, *Wolbachia* se establece por
completo (los silvestres se extinguen) y las infecciones caen a
prácticamente cero en menos de 100 días — mientras que sin control el
dengue se estabiliza en un nivel endémico permanente.

## Tests

```bash
pytest tests/ -q
```

Cubren: validación de parámetros, reducción del modelo a la dinámica
silvestre pura cuando *Wolbachia* está en cero, el equilibrio silvestre
analítico y $R_0$, conservación de la población humana, positividad, y el
establecimiento de *Wolbachia* tras una liberación única suficientemente
grande.
