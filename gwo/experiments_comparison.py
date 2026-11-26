"""
experiments_comparison.py
Protocole expérimental complet de comparaison :
GWO classique vs GT-GWO (hybride avec Théorie des Jeux Évolutive)
sur les deux familles de problèmes :
1. Problème continu : Rastrigin (30 runs)
2. Problème combinatoire discret : Sac à dos 0/1 (30 runs)

Effectue l'analyse statistique, les tests de significativité de Wilcoxon,
génère les courbes comparatives et l'évolution des stratégies.
"""

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Imports des algorithmes
from Gwo import gwo as classic_gwo
from gt_gwo import gt_gwo, objective_rastrigin
from knapsack_problem import get_standard_instance
from binary_gwo import binary_gwo as classic_bgwo
from binary_gt_gwo import binary_gt_gwo
from game_theory_model import STRATEGY_NAMES


N_RUNS = 30
POP_SIZE = 30
MAX_ITERS = 100


def run_benchmark():
    print("=" * 70)
    print("  PROTOCOLE COMPARATIF COMPLET : GWO CLASSIQUE vs GT-GWO")
    print("=" * 70)

    # ==============================================================
    # PARTIE 1 : BENCHMARK SUR LE PROBLÈME CONTINU (RASTRIGIN)
    # ==============================================================
    print("\n>>> [1/2] Lancement du benchmark continu sur Rastrigin (30 runs)...")

    rastrigin_classic_fits = []
    rastrigin_classic_times = []
    rastrigin_classic_curves = []

    rastrigin_gt_fits = []
    rastrigin_gt_times = []
    rastrigin_gt_curves = []
    rastrigin_gt_strat_histories = []

    for run in range(N_RUNS):
        seed = 42 + run

        # 1. GWO classique
        np.random.seed(seed)
        t0 = time.perf_counter()
        _, fit_c, curve_c = classic_gwo(
            population_size=POP_SIZE,
            dimensions=2,
            max_iterations=MAX_ITERS,
            verbose=False
        )
        t1 = time.perf_counter()
        rastrigin_classic_fits.append(fit_c)
        rastrigin_classic_times.append(t1 - t0)
        rastrigin_classic_curves.append(curve_c)

        # 2. GT-GWO (hybride théorie des jeux)
        np.random.seed(seed)
        t0 = time.perf_counter()
        _, fit_gt, curve_gt, strat_hist = gt_gwo(
            objective_function=objective_rastrigin,
            population_size=POP_SIZE,
            dimensions=2,
            max_iterations=MAX_ITERS,
            verbose=False
        )
        t1 = time.perf_counter()
        rastrigin_gt_fits.append(fit_gt)
        rastrigin_gt_times.append(t1 - t0)
        rastrigin_gt_curves.append(curve_gt)
        rastrigin_gt_strat_histories.append(strat_hist)

        if (run + 1) % 10 == 0:
            print(f"    Rastrigin : Run {run + 1:02d}/{N_RUNS} terminé.")

    rastrigin_classic_fits = np.array(rastrigin_classic_fits)
    rastrigin_gt_fits = np.array(rastrigin_gt_fits)

    # Test de Wilcoxon sur Rastrigin
    diff_r = rastrigin_classic_fits - rastrigin_gt_fits
    if np.all(diff_r == 0):
        wilcox_p_rastrigin = 1.0
    else:
        try:
            _, wilcox_p_rastrigin = wilcoxon(rastrigin_classic_fits, rastrigin_gt_fits)
        except Exception:
            wilcox_p_rastrigin = 1.0

    # ==============================================================
    # PARTIE 2 : BENCHMARK SUR LE PROBLÈME DISCRET (SAC À DOS 0/1)
    # ==============================================================
    print("\n>>> [2/2] Lancement du benchmark combinatoire sur Sac à Dos (30 runs)...")
    knapsack_problem = get_standard_instance()
    exact_optimum, _ = knapsack_problem.solve_exact_dp()

    knapsack_classic_vals = []
    knapsack_classic_times = []
    knapsack_classic_curves = []

    knapsack_gt_vals = []
    knapsack_gt_times = []
    knapsack_gt_curves = []
    knapsack_gt_strat_histories = []

    for run in range(N_RUNS):
        seed = 42 + run

        # 1. Binary GWO classique
        np.random.seed(seed)
        t0 = time.perf_counter()
        _, val_c, _, curve_c = classic_bgwo(
            problem=knapsack_problem,
            population_size=POP_SIZE,
            max_iterations=MAX_ITERS,
            verbose=False
        )
        t1 = time.perf_counter()
        knapsack_classic_vals.append(val_c)
        knapsack_classic_times.append(t1 - t0)
        knapsack_classic_curves.append(curve_c)

        # 2. Binary GT-GWO (hybride théorie des jeux)
        np.random.seed(seed)
        t0 = time.perf_counter()
        _, val_gt, _, curve_gt, strat_hist = binary_gt_gwo(
            problem=knapsack_problem,
            population_size=POP_SIZE,
            max_iterations=MAX_ITERS,
            verbose=False
        )
        t1 = time.perf_counter()
        knapsack_gt_vals.append(val_gt)
        knapsack_gt_times.append(t1 - t0)
        knapsack_gt_curves.append(curve_gt)
        knapsack_gt_strat_histories.append(strat_hist)

        if (run + 1) % 10 == 0:
            print(f"    Sac à Dos : Run {run + 1:02d}/{N_RUNS} terminé.")

    knapsack_classic_vals = np.array(knapsack_classic_vals)
    knapsack_gt_vals = np.array(knapsack_gt_vals)

    # Test de Wilcoxon sur Sac à Dos
    diff_k = knapsack_gt_vals - knapsack_classic_vals
    if np.all(diff_k == 0):
        wilcox_p_knapsack = 1.0
    else:
        try:
            _, wilcox_p_knapsack = wilcoxon(knapsack_classic_vals, knapsack_gt_vals)
        except Exception:
            wilcox_p_knapsack = 1.0

    # ==============================================================
    # PARTIE 3 : TABLEAU RÉCAPITULATIF DES MÉTRIQUES
    # ==============================================================
    metrics = [
        # Rastrigin (Minimisation)
        {
            "Problème": "Rastrigin (Continu)",
            "Algorithme": "GWO classique",
            "Best": f"{np.min(rastrigin_classic_fits):.4e}",
            "Worst": f"{np.max(rastrigin_classic_fits):.4e}",
            "Mean": f"{np.mean(rastrigin_classic_fits):.4e}",
            "Std": f"{np.std(rastrigin_classic_fits, ddof=1):.4e}",
            "Succès (%)": f"{np.mean(rastrigin_classic_fits <= 1e-8) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(rastrigin_classic_times):.4f}",
            "Wilcoxon p-value": "-"
        },
        {
            "Problème": "Rastrigin (Continu)",
            "Algorithme": "GT-GWO (Hybride)",
            "Best": f"{np.min(rastrigin_gt_fits):.4e}",
            "Worst": f"{np.max(rastrigin_gt_fits):.4e}",
            "Mean": f"{np.mean(rastrigin_gt_fits):.4e}",
            "Std": f"{np.std(rastrigin_gt_fits, ddof=1):.4e}",
            "Succès (%)": f"{np.mean(rastrigin_gt_fits <= 1e-8) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(rastrigin_gt_times):.4f}",
            "Wilcoxon p-value": f"{wilcox_p_rastrigin:.4e}"
        },
        # Sac à Dos (Maximisation)
        {
            "Problème": "Sac à Dos 0/1 (Discret)",
            "Algorithme": "BGWO classique",
            "Best": f"{np.max(knapsack_classic_vals):.1f}",
            "Worst": f"{np.min(knapsack_classic_vals):.1f}",
            "Mean": f"{np.mean(knapsack_classic_vals):.2f}",
            "Std": f"{np.std(knapsack_classic_vals, ddof=1):.4f}",
            "Succès (%)": f"{np.mean(knapsack_classic_vals == exact_optimum) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(knapsack_classic_times):.4f}",
            "Wilcoxon p-value": "-"
        },
        {
            "Problème": "Sac à Dos 0/1 (Discret)",
            "Algorithme": "BGT-GWO (Hybride)",
            "Best": f"{np.max(knapsack_gt_vals):.1f}",
            "Worst": f"{np.min(knapsack_gt_vals):.1f}",
            "Mean": f"{np.mean(knapsack_gt_vals):.2f}",
            "Std": f"{np.std(knapsack_gt_vals, ddof=1):.4f}",
            "Succès (%)": f"{np.mean(knapsack_gt_vals == exact_optimum) * 100:.1f}%",
            "Temps moyen (s)": f"{np.mean(knapsack_gt_times):.4f}",
            "Wilcoxon p-value": f"{wilcox_p_knapsack:.4e}"
        }
    ]

    # Sauvegarde CSV
    with open("comparison_summary.csv", "w", newline="", encoding="utf-8") as f:
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

    # ==============================================================
    # PARTIE 4 : GÉNÉRATION DES FIGURES HAUTE RÉSOLUTION
    # ==============================================================

    # 1. Comparaison Convergence Rastrigin
    mean_curve_rc = np.mean(np.array(rastrigin_classic_curves), axis=0)
    mean_curve_rgt = np.mean(np.array(rastrigin_gt_curves), axis=0)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, MAX_ITERS + 1), mean_curve_rc, label="GWO classique", color="#d62728", linestyle="--", linewidth=2)
    plt.plot(range(1, MAX_ITERS + 1), mean_curve_rgt, label="GT-GWO (Hybride Théorie des Jeux)", color="#2ca02c", linewidth=2.5)
    plt.yscale("symlog", linthresh=1e-8)
    plt.title("Rastrigin : Comparaison de convergence moyenne (GWO vs GT-GWO)")
    plt.xlabel("Itération")
    plt.ylabel("Fitness moyenne (symlog)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("comparison_rastrigin.png", dpi=300)
    plt.close()

    # 2. Comparaison Convergence Sac à Dos
    mean_curve_kc = np.mean(np.array(knapsack_classic_curves), axis=0)
    mean_curve_kgt = np.mean(np.array(knapsack_gt_curves), axis=0)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, MAX_ITERS + 1), mean_curve_kc, label="BGWO classique", color="#d62728", linestyle="--", linewidth=2)
    plt.plot(range(1, MAX_ITERS + 1), mean_curve_kgt, label="BGT-GWO (Hybride Théorie des Jeux)", color="#2ca02c", linewidth=2.5)
    plt.axhline(y=exact_optimum, color="blue", linestyle=":", label=f"Optimum théorique ({exact_optimum})")
    plt.title("Sac à Dos 0/1 : Comparaison de convergence moyenne (BGWO vs BGT-GWO)")
    plt.xlabel("Itération")
    plt.ylabel("Valeur totale du sac à dos")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("comparison_knapsack.png", dpi=300)
    plt.close()

    # 3. Évolution des stratégies sur Rastrigin
    avg_strat_r = np.mean(np.array(rastrigin_gt_strat_histories), axis=0)
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
    plt.savefig("strategy_evolution_rastrigin.png", dpi=300)
    plt.close()

    # 4. Évolution des stratégies sur Sac à Dos
    avg_strat_k = np.mean(np.array(knapsack_gt_strat_histories), axis=0)
    plt.figure(figsize=(9, 5))
    for k in range(len(STRATEGY_NAMES)):
        plt.plot(avg_strat_k[:, k], label=f"Stratégie {STRATEGY_NAMES[k]}", color=colors[k], linewidth=2)
    plt.title("Dynamique du Réplicateur : Évolution des stratégies sur le Sac à Dos 0/1")
    plt.xlabel("Itération")
    plt.ylabel("Probabilité d'adoption p_k")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("strategy_evolution_knapsack.png", dpi=300)
    plt.close()

    print("\n>>> Tous les graphiques ont été générés avec succès :")
    print("    - comparison_rastrigin.png")
    print("    - comparison_knapsack.png")
    print("    - strategy_evolution_rastrigin.png")
    print("    - strategy_evolution_knapsack.png")
    print("    - comparison_summary.csv")


if __name__ == "__main__":
    run_benchmark()
