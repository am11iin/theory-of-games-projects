"""
src/gwo.py
Implémentation de base de l'algorithme Grey Wolf Optimizer (GWO) classique.
"""

import numpy as np


def objective_function(x):
    """Fonction objectif standard : Rastrigin."""
    n = len(x)
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))


def gwo(
    population_size=30,
    dimensions=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    max_iterations=100,
    verbose=True
):
    """
    Exécute l'algorithme GWO standard.
    """
    wolves = np.random.uniform(
        lower_bound,
        upper_bound,
        (population_size, dimensions)
    )

    global_best_position = None
    global_best_score = float("inf")
    convergence_curve = []

    for t in range(max_iterations):
        fitness = np.array([objective_function(wolf) for wolf in wolves])

        sorted_indices = np.argsort(fitness)
        alpha = wolves[sorted_indices[0]].copy()
        beta = wolves[sorted_indices[1]].copy()
        delta = wolves[sorted_indices[2]].copy()

        alpha_score = fitness[sorted_indices[0]]

        if alpha_score < global_best_score:
            global_best_score = alpha_score
            global_best_position = alpha.copy()

        a = 2.0 - 2.0 * (t / max_iterations)

        for i in range(population_size):
            for j in range(dimensions):
                r1, r2 = np.random.random(2)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.random(2)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * beta[j] - wolves[i, j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.random(2)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * delta[j] - wolves[i, j])
                X3 = delta[j] - A3 * D_delta

                wolves[i, j] = (X1 + X2 + X3) / 3.0

        wolves = np.clip(wolves, lower_bound, upper_bound)
        convergence_curve.append(global_best_score)

        if verbose:
            print(f"Iteration {t + 1}/{max_iterations} | Best fitness: {global_best_score:.8f}")

    final_fitness = np.array([objective_function(wolf) for wolf in wolves])
    best_index = np.argmin(final_fitness)

    if final_fitness[best_index] < global_best_score:
        global_best_score = final_fitness[best_index]
        global_best_position = wolves[best_index].copy()

    return global_best_position, global_best_score, convergence_curve


Gwo = gwo


if __name__ == "__main__":
    best_pos, best_fit, curve = gwo(verbose=False)
    print("========== GWO TEST ==========")
    print("Best position:", best_pos)
    print("Best fitness:", best_fit)
