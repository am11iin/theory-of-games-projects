"""
experiments/experiments_comparison.py
Protocole expérimental complet de comparaison :
GWO classique vs GT-GWO sur Rastrigin et Sac à Dos 0/1.
Sauvegarde automatique dans results/figures et results/data.
"""

import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from src.gwo import gwo as classic_gwo
from src.gt_gwo import gt_gwo, objective_rastrigin
from src.knapsack_problem import get_standard_instance
from src.binary_gwo import binary_gwo as classic_bgwo
from src.binary_gt_gwo import binary_gt_gwo
from src.game_theory import STRATEGY_NAMES

FIGURES_DIR = ROOT_DIR / "results" / "figures"
DATA_DIR = ROOT_DIR / "results" / "data"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 30
POP_SIZE = 30
MAX_ITERS = 100


def run_benchmark():
    print("=" * 70)
    print("  PROTOCOLE COMPARATIF COMPLET : GWO CLASSIQUE vs GT-GWO")
    print("=" * 70)

    # 1. RASTRIGIN
    print("\n>>> [1/2] Lancement du benchmark continu sur Rastrigin (30 runs)...")
    r_classic_fits, r_classic_times, r_classic_curves = [], [], []
    r_gt_fits, r_gt_times, r_gt_curves, r_gt_strats = [], [], [], []

    for run in range(N_RUNS):
        seed = 42 + run
        np.random.seed(seed)
        t0 = time.perf_counter()
        _, fit_c, curve_c = classic_gwo(population_size=POP_SIZE, dimensions=2, max_iterations=MAX_ITERS, verbose=False)
        t1 = time.perf_counter()
        r_classic_fits.append(fit_c)
        r_classic_times.append(t1 - t0)
        r_classic_curves.append(curve_c)

        np.random.seed(seed)
        t0 = time.perf_counter()
        _, fit_gt, curve_gt, strat_hist = gt_gwo(objective_function=objective_rastrigin, population_size=POP_SIZE, dimensions=2, max_iterations=MAX_ITERS, verbose=False)
        t1 = time.perf_counter()
        r_gt_fits.append(fit_gt)
        r_gt_times.append(t1 - t0)
        r_gt_curves.append(curve_gt)
        r_gt_strats.append(strat_hist)

    r_classic_fits = np.array(r_classic_fits)
    r_gt_fits = np.array(r_gt_fits)

    diff_r = r_classic_fits - r_gt_fits
    wilcox_p_rastrigin = 1.0 if np.all(diff_r == 0) else wilcoxon(r_classic_fits, r_gt_fits)[1]

    # 2. SAC À DOS
    print("\n>>> [2/2] Lancement du benchmark combinatoire sur Sac à Dos (30 runs)...")
    knapsack_problem = get_standard_instance()
    exact_optimum, _ = knapsack_problem.solve_exact_dp()

    k_classic_vals, k_classic_times, k_classic_curves = [], [], []
    k_gt_vals, k_gt_times, k_gt_curves, k_gt_strats = [], [], [], []

    for run in range(N_RUNS):
        seed = 42 + run
        np.random.seed(seed)
        t0 = time.perf_counter()
        _, val_c, _, curve_c = classic_bgwo(problem=knapsack_problem, population_size=POP_SIZE, max_iterations=MAX_ITERS, verbose=False)
        t1 = time.perf_counter()
        k_classic_vals.append(val_c)
        k_classic_times.append(t1 - t0)
        k_classic_curves.append(curve_c)

        np.random.seed(seed)
        t0 = time.perf_counter()
        _, val_gt, _, curve_gt, strat_hist = binary_gt_gwo(problem=knapsack_problem, population_size=POP_SIZE, max_iterations=MAX_ITERS, verbose=False)
        t1 = time.perf_counter()
        k_gt_vals.append(val_gt)
        k_gt_times.append(t1 - t0)
        k_gt_curves.append(curve_gt)
        k_gt_strats.append(strat_hist)

    k_classic_vals = np.array(k_classic_vals)
    k_gt_vals = np.array(k_gt_vals)

    diff_k = k_gt_vals - k_classic_vals
    wilcox_p_knapsack = 1.0 if np.all(diff_k == 0) else wilcoxon(k_classic_vals, k_gt_vals)[1]

    # RÉSULTATS
    metrics = [
        {
            "Problème": "Rastrigin (Continu)",
            "Algorithme": "GWO classique",
            "Best": f"{np.min(r_classic_fits):.4e}",
            "Worst": f"{np.max(r_classic_fits):.4e}",
            "Mean": f"{np.mean(r_classic_fits):.4e}",
            "Std": f"{np.std(r_classic_fits, ddof=1):.4e}",
            "Succès (%)": f"{np.mean(r_classic_fits <= 1e-8) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(r_classic_times):.4f}",
            "Wilcoxon p-value": "-"
        },
        {
            "Problème": "Rastrigin (Continu)",
            "Algorithme": "GT-GWO (Hybride)",
            "Best": f"{np.min(r_gt_fits):.4e}",
            "Worst": f"{np.max(r_gt_fits):.4e}",
            "Mean": f"{np.mean(r_gt_fits):.4e}",
            "Std": f"{np.std(r_gt_fits, ddof=1):.4e}",
            "Succès (%)": f"{np.mean(r_gt_fits <= 1e-8) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(r_gt_times):.4f}",
            "Wilcoxon p-value": f"{wilcox_p_rastrigin:.4e}"
        },
        {
            "Problème": "Sac à Dos 0/1 (Discret)",
            "Algorithme": "BGWO classique",
            "Best": f"{np.max(k_classic_vals):.1f}",
            "Worst": f"{np.min(k_classic_vals):.1f}",
            "Mean": f"{np.mean(k_classic_vals):.2f}",
            "Std": f"{np.std(k_classic_vals, ddof=1):.4f}",
            "Succès (%)": f"{np.mean(k_classic_vals == exact_optimum) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(k_classic_times):.4f}",
            "Wilcoxon p-value": "-"
        },
        {
            "Problème": "Sac à Dos 0/1 (Discret)",
            "Algorithme": "BGT-GWO (Hybride)",
            "Best": f"{np.max(k_gt_vals):.1f}",
            "Worst": f"{np.min(k_gt_vals):.1f}",
            "Mean": f"{np.mean(k_gt_vals):.2f}",
            "Std": f"{np.std(k_gt_vals, ddof=1):.4f}",
            "Succès (%)": f"{np.mean(k_gt_vals == exact_optimum) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(k_gt_times):.4f}",
            "Wilcoxon p-value": f"{wilcox_p_knapsack:.4e}"
        }
    ]

    csv_file = DATA_DIR / "comparison_summary.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    print("\n" + "=" * 80)
    print("  TABLEAU COMPARATIF COMPLET (30 RUNS INDÉPENDANTS)")
    print("=" * 80)
    print(f"{'Problème':<23} | {'Algorithme':<18} | {'Best':<10} | {'Mean':<10} | {'Std':<10} | {'Succès':<8} | {'Temps':<8} | {'p-value'}")
    print("-" * 115)
    for m in metrics:
        print(f"{m['Problème']:<23} | {m['Algorithme']:<18} | {m['Best']:<10} | {m['Mean']:<10} | {m['Std']:<10} | {m['Succès (%)']:<8} | {m['Temps moyen (s)']:<8} | {m['Wilcoxon p-value']}")

    # Graphique Rastrigin
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, MAX_ITERS + 1), np.mean(np.array(r_classic_curves), axis=0), label="GWO classique", color="#d62728", linestyle="--", linewidth=2)
    plt.plot(range(1, MAX_ITERS + 1), np.mean(np.array(r_gt_curves), axis=0), label="GT-GWO (Hybride Théorie des Jeux)", color="#2ca02c", linewidth=2.5)
    plt.yscale("symlog", linthresh=1e-8)
    plt.title("Rastrigin : Comparaison de convergence moyenne (GWO vs GT-GWO)")
    plt.xlabel("Itération")
    plt.ylabel("Fitness moyenne (symlog)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_rastrigin.png", dpi=300)
    plt.close()

    # Graphique Sac à dos
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, MAX_ITERS + 1), np.mean(np.array(k_classic_curves), axis=0), label="BGWO classique", color="#d62728", linestyle="--", linewidth=2)
    plt.plot(range(1, MAX_ITERS + 1), np.mean(np.array(k_gt_curves), axis=0), label="BGT-GWO (Hybride Théorie des Jeux)", color="#2ca02c", linewidth=2.5)
    plt.axhline(y=exact_optimum, color="blue", linestyle=":", label=f"Optimum théorique ({exact_optimum})")
    plt.title("Sac à Dos 0/1 : Comparaison de convergence moyenne (BGWO vs BGT-GWO)")
    plt.xlabel("Itération")
    plt.ylabel("Valeur totale du sac à dos")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparison_knapsack.png", dpi=300)
    plt.close()

    # Évolution stratégies Rastrigin
    avg_strat_r = np.mean(np.array(r_gt_strats), axis=0)
    plt.figure(figsize=(9, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k in range(len(STRATEGY_NAMES)):
        plt.plot(avg_strat_r[:, k], label=f"Stratégie {STRATEGY_NAMES[k]}", color=colors[k], linewidth=2)
    plt.title("Dynamique du Réplicateur : Évolution des stratégies sur Rastrigin")
    plt.xlabel("Itération")
    plt.ylabel("Probabilité d'adoption p_k")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "strategy_evolution_rastrigin.png", dpi=300)
    plt.close()

    # Évolution stratégies Sac à dos
    avg_strat_k = np.mean(np.array(k_gt_strats), axis=0)
    plt.figure(figsize=(9, 5))
    for k in range(len(STRATEGY_NAMES)):
        plt.plot(avg_strat_k[:, k], label=f"Stratégie {STRATEGY_NAMES[k]}", color=colors[k], linewidth=2)
    plt.title("Dynamique du Réplicateur : Évolution des stratégies sur le Sac à Dos 0/1")
    plt.xlabel("Itération")
    plt.ylabel("Probabilité d'adoption p_k")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "strategy_evolution_knapsack.png", dpi=300)
    plt.close()

    print(f"\n>>> Résultats sauvegardés dans :")
    print(f"    - {FIGURES_DIR}")
    print(f"    - {DATA_DIR}")


if __name__ == "__main__":
    run_benchmark()
