"""
experiments_knapsack.py
Protocole expérimental : 30 exécutions indépendantes de Binary GWO
sur le problème du sac à dos 0/1 (20 objets, capacité 300).
Génère :
- knapsack_experiments.csv
- knapsack_mean_convergence.png
"""

import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from knapsack_problem import get_standard_instance
from binary_gwo import binary_gwo


# ==========================================
# 1. CONFIGURATION EXPÉRIMENTALE
# ==========================================

NUMBER_OF_RUNS = 30
POPULATION_SIZE = 30
MAX_ITERATIONS = 100

problem = get_standard_instance()
theoretical_optimum, opt_items = problem.solve_exact_dp()

print("==================================================")
print("EXPÉRIENCES BINARY GWO - SAC À DOS 0/1")
print(f"Nombre d'objets : {problem.n_items}")
print(f"Capacité du sac : {problem.capacity}")
print(f"Optimum théorique exact : {theoretical_optimum}")
print(f"Nombre d'exécutions : {NUMBER_OF_RUNS}")
print("==================================================")


# ==========================================
# 2. EXÉCUTION DES 30 RUNS
# ==========================================

results_values = []
results_weights = []
execution_times = []
convergence_curves = []
best_solutions = []

for run in range(NUMBER_OF_RUNS):

    # Graine aléatoire reproductible différente pour chaque run
    np.random.seed(42 + run)

    start_time = time.perf_counter()

    best_bin, best_val, best_w, curve = binary_gwo(
        problem=problem,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        verbose=False
    )

    end_time = time.perf_counter()
    runtime = end_time - start_time

    results_values.append(best_val)
    results_weights.append(best_w)
    execution_times.append(runtime)
    convergence_curves.append(curve)
    best_solutions.append(best_bin)

    is_optimal = " (OPTIMUM ATTEINT !)" if best_val == theoretical_optimum else ""
    print(
        f"Run {run + 1:02d}/{NUMBER_OF_RUNS} | "
        f"Valeur : {best_val:.1f} | "
        f"Poids : {best_w:.1f}/{problem.capacity} | "
        f"Temps : {runtime:.4f}s{is_optimal}"
    )


# ==========================================
# 3. ANALYSE STATISTIQUE
# ==========================================

results_values = np.array(results_values)
execution_times = np.array(execution_times)

best_val = np.max(results_values)   # En maximisation, la meilleure est le max
worst_val = np.min(results_values)  # La pire est le min
mean_val = np.mean(results_values)
std_val = np.std(results_values, ddof=1)

success_count = np.sum(results_values == theoretical_optimum)
success_rate = (success_count / NUMBER_OF_RUNS) * 100.0

mean_runtime = np.mean(execution_times)
total_runtime = np.sum(execution_times)


# ==========================================
# 4. AFFICHAGE DES RÉSULTATS
# ==========================================

print("\n========== RÉSULTATS STATISTIQUES (30 RUNS) ==========")
print(f"Optimum théorique : {theoretical_optimum}")
print(f"Meilleure valeur (Best) : {best_val:.2f}")
print(f"Pire valeur (Worst)     : {worst_val:.2f}")
print(f"Moyenne (Mean)          : {mean_val:.2f}")
print(f"Écart-type (Std)        : {std_val:.4f}")
print(f"Taux de succès          : {success_count}/{NUMBER_OF_RUNS} ({success_rate:.2f}%)")
print(f"Temps moyen par run     : {mean_runtime:.4f} secondes")
print(f"Temps total             : {total_runtime:.4f} secondes")


# ==========================================
# 5. COURBE DE CONVERGENCE MOYENNE
# ==========================================

convergence_curves = np.array(convergence_curves)
mean_curve = np.mean(convergence_curves, axis=0)

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, MAX_ITERATIONS + 1),
    mean_curve,
    label="Convergence moyenne BGWO",
    color="#1f77b4",
    linewidth=2
)

# Ligne horizontale de l'optimum théorique
plt.axhline(
    y=theoretical_optimum,
    color="r",
    linestyle="--",
    label=f"Optimum théorique ({theoretical_optimum})"
)

plt.title("Binary GWO - Convergence moyenne sur le Sac à Dos 0/1 (20 objets)")
plt.xlabel("Itération")
plt.ylabel("Valeur totale du sac à dos")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()

image_path = "knapsack_mean_convergence.png"
plt.savefig(image_path, dpi=300)
plt.close()
print(f"\nGraphique sauvegardé : {image_path}")


# ==========================
# 6. SAUVEGARDE DANS UN CSV
# ==========================

csv_path = "knapsack_experiments.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Valeur", "Poids", "Temps (s)", "Solution"])
    for i in range(NUMBER_OF_RUNS):
        writer.writerow([
            i + 1,
            results_values[i],
            results_weights[i],
            f"{execution_times[i]:.6f}",
            str(best_solutions[i].tolist())
        ])

print(f"Données sauvegardées : {csv_path}")
