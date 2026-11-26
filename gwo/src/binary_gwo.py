"""
src/binary_gwo.py
Implémentation de l'algorithme Grey Wolf Optimizer Binaire (BGWO) classique.
"""

import numpy as np


def sigmoid(x):
    x_clipped = np.clip(x, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def binarize(continuous_position):
    probs = sigmoid(continuous_position)
    random_thresholds = np.random.random(len(continuous_position))
    return (probs >= random_thresholds).astype(int)


def binary_gwo(
    problem,
    population_size=30,
    max_iterations=100,
    lower_bound=-4.0,
    upper_bound=4.0,
    verbose=True
):
    dimensions = problem.n_items

    wolves = np.random.uniform(
        lower_bound,
        upper_bound,
        (population_size, dimensions)
    )

    alpha_pos = np.zeros(dimensions)
    beta_pos = np.zeros(dimensions)
    delta_pos = np.zeros(dimensions)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    global_best_binary = None
    global_best_value = 0.0
    global_best_weight = 0.0

    convergence_curve = []

    for t in range(max_iterations):
        for i in range(population_size):
            bin_sol = binarize(wolves[i])
            repaired_sol, value, weight = problem.repair_and_evaluate(bin_sol)
            fitness = -value

            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = alpha_score
                beta_pos = alpha_pos.copy()

                alpha_score = fitness
                alpha_pos = wolves[i].copy()

                global_best_binary = repaired_sol.copy()
                global_best_value = value
                global_best_weight = weight

            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = fitness
                beta_pos = wolves[i].copy()

            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i].copy()

        a = 2.0 - 2.0 * (t / max_iterations)

        for i in range(population_size):
            for j in range(dimensions):
                r1, r2 = np.random.random(2)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.random(2)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.random(2)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                wolves[i, j] = (X1 + X2 + X3) / 3.0

        wolves = np.clip(wolves, lower_bound, upper_bound)
        convergence_curve.append(global_best_value)

        if verbose and (t + 1) % 10 == 0:
            print(f"Itération {t + 1:03d}/{max_iterations} | Valeur : {global_best_value:.1f} | Poids : {global_best_weight:.1f}")

    return global_best_binary, global_best_value, global_best_weight, convergence_curve
