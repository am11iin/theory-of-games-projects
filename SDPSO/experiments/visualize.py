"""
Génération des graphiques pour les expériences PSO vs SDPSO.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Pas besoin d'interface graphique
import matplotlib.pyplot as plt


STRATEGY_NAMES = ["LIPS", "UPSO", "LDWPSO", "CLPSO"]
COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd"]


def _style():
    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "legend.fontsize": 10})


def plot_convergence(func_name, pso_runs, sdpso_runs, output_path):
    """
    Courbes de convergence en semi-log (médiane + intervalle 25-75 %).
    pso_runs, sdpso_runs : tableaux (nb_runs × nb_evals)
    """
    _style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    n_evals = min(pso_runs.shape[1], sdpso_runs.shape[1])
    evals = np.arange(1, n_evals + 1)

    for arr, label, color in [
        (pso_runs,   "PSO classique", "#d9534f"),
        (sdpso_runs, "SDPSO (Théorie des jeux)", "#0275d8"),
    ]:
        arr = np.maximum(arr[:, :n_evals], 1e-16)
        med  = np.median(arr, axis=0)
        p25  = np.percentile(arr, 25, axis=0)
        p75  = np.percentile(arr, 75, axis=0)
        ax.semilogy(evals, med, label=label, color=color, linewidth=2)
        ax.fill_between(evals, p25, p75, color=color, alpha=0.15)

    ax.set_title(f"Convergence — {func_name}", fontweight="bold")
    ax.set_xlabel("Évaluations de la fonction (FES)")
    ax.set_ylabel("f(x) — échelle logarithmique")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_strategy_dynamics(func_name, fes_history, p_history, output_path):
    """
    Évolution des probabilités d'adoption des 4 stratégies au fil des évaluations.
    Montre la dynamique du réplicateur, les restarts et le seuil de mutation.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    for k in range(4):
        ax.plot(fes_history, p_history[:, k], label=STRATEGY_NAMES[k],
                color=COLORS[k], linewidth=2)

    ax.axhline(0.25, color="gray", linestyle=":", alpha=0.6, label="Équilibre initial (0.25)")
    ax.axhline(0.02, color="red",  linestyle="--", alpha=0.5, label="Seuil de mutation (0.02)")

    ax.set_title(f"Dynamique des stratégies — {func_name}", fontweight="bold")
    ax.set_xlabel("Évaluations (FES)")
    ax.set_ylabel("Probabilite d'adoption p_k")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_strategy_bar(usage_by_func, output_path):
    """
    Histogramme groupé du nombre de fois que chaque stratégie
    a été sélectionnée, par fonction benchmark.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    funcs = list(usage_by_func.keys())
    x = np.arange(len(funcs))
    width = 0.2

    for k, (name, color) in enumerate(zip(STRATEGY_NAMES, COLORS)):
        counts = [usage_by_func[f][k] for f in funcs]
        ax.bar(x + k * width, counts, width, label=name, color=color, alpha=0.9)

    ax.set_title("Fréquence de sélection des stratégies (sous-population EGT)", fontweight="bold")
    ax.set_xlabel("Fonction benchmark")
    ax.set_ylabel("Nombre de sélections")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(funcs, rotation=15)
    ax.legend(title="Stratégies")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
