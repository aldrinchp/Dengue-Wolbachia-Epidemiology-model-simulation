"""Figures: stacked time series and infection comparison.

Plain matplotlib (no seaborn), colorblind-friendly palette (Okabe-Ito),
print-legible font, PNG (300 dpi) and vector PDF export.
Figure labels and titles in English (as requested). Each function is
presentation-only: it receives an already-computed ``SimulationResult`` and
does not run any integration itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dengue_wolbachia.simulate import SimulationResult

# Okabe-Ito palette, colorblind-friendly.
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 30,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "legend.fontsize": 19,
        "legend.title_fontsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 28,
        "figure.titleweight": "bold",
        "lines.linewidth": 2.2,
        "legend.markerscale": 1.3,
        "legend.handlelength": 2.2,
        "legend.labelspacing": 0.6,
        "legend.borderpad": 0.7,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # fonts embedded as text, not curves, in the PDF
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, out_stem: str | Path) -> None:
    """Save ``fig`` as ``{out_stem}.png`` (300 dpi) and ``{out_stem}.pdf`` (vector)."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_stacked_timeseries(
    result: SimulationResult, title: str, out_stem: str | Path
) -> None:
    """Time series of the three blocks (wild mosquitoes, Wolbachia mosquitoes,
    humans) in stacked panels sharing the time axis."""
    d = result.as_dict()
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    ax = axes[0]
    ax.plot(d["t"], d["N_FS"], color=PALETTE["blue"], label="$N_{FS}$ (susceptible females)")
    ax.plot(d["t"], d["N_FI"], color=PALETTE["vermillion"], label="$N_{FI}$ (infected females)")
    ax.plot(d["t"], d["N_M"], color=PALETTE["black"], linestyle="--", label="$N_M$ (males)")
    ax.set_ylabel("Wild\nmosquitoes")
    ax.legend(loc="upper right", ncol=1)

    ax = axes[1]
    ax.plot(d["t"], d["W_F"], color=PALETTE["bluish_green"], label="$W_F$ (females)")
    ax.plot(d["t"], d["W_M"], color=PALETTE["orange"], linestyle="--", label="$W_M$ (males)")
    ax.set_ylabel("$\\mathit{Wolbachia}$-carrying\nmosquitoes")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(d["t"], d["S"], color=PALETTE["sky_blue"], label="$S$")
    ax.plot(d["t"], d["I"], color=PALETTE["vermillion"], label="$I$")
    ax.plot(d["t"], d["R"], color=PALETTE["reddish_purple"], label="$R$")
    ax.set_ylabel("Humans")
    ax.set_xlabel("Time (days)")
    ax.legend(loc="lower right")

    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_stem)


def plot_infection_comparison(
    results: dict[str, SimulationResult],
    title: str,
    out_stem: str | Path,
    variable: str = "I",
    ylabel: str | None = None,
) -> None:
    """Compare a state variable (prevalence ``I(t)`` by default, or cumulative
    incidence ``C(t)``) across scenarios on a single axis."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = list(PALETTE.values())
    for i, (label, result) in enumerate(results.items()):
        d = result.as_dict()
        ax.plot(d["t"], d[variable], color=colors[i % len(colors)], label=label)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel or f"Infected humans ${variable}(t)$")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    fig.tight_layout()
    _save(fig, out_stem)
