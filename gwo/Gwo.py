
import numpy as np


# ==========================================
# 1. OBJECTIVE FUNCTION: RASTRIGIN
# ==========================================

def objective_function(x):
    n = len(x)

    return 10 * n + np.sum(
        x**2 - 10 * np.cos(2 * np.pi * x)
    )


# ==========================================
# 2. GREY WOLF OPTIMIZER
# ==========================================

def gwo(
    population_size=30,
    dimensions=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    max_iterations=100,
    verbose=True
):

    # Initialize wolf population
    wolves = np.random.uniform(
        lower_bound,
        upper_bound,
        (population_size, dimensions)
    )

    # Best solution found so far
    global_best_position = None
    global_best_score = float("inf")

    convergence_curve = []

    # Main loop
    for t in range(max_iterations):

        # Evaluate current population
        fitness = np.array([
            objective_function(wolf) for wolf in wolves
        ])

        # Identify Alpha, Beta and Delta
        sorted_indices = np.argsort(fitness)

        alpha = wolves[sorted_indices[0]].copy()
        beta = wolves[sorted_indices[1]].copy()
        delta = wolves[sorted_indices[2]].copy()

        alpha_score = fitness[sorted_indices[0]]

        # Update global best
        if alpha_score < global_best_score:
            global_best_score = alpha_score
            global_best_position = alpha.copy()

        # Decrease parameter a from 2 to 0
        a = 2 - 2 * (t / max_iterations)

        # Update wolf positions
        for i in range(population_size):

            for j in range(dimensions):

                # Alpha
                r1, r2 = np.random.random(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A1 * D_alpha

                # Beta
                r1, r2 = np.random.random(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * beta[j] - wolves[i, j])
                X2 = beta[j] - A2 * D_beta

                # Delta
                r1, r2 = np.random.random(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * delta[j] - wolves[i, j])
                X3 = delta[j] - A3 * D_delta

                # New position
                wolves[i, j] = (X1 + X2 + X3) / 3

        # Keep wolves inside search boundaries
        wolves = np.clip(wolves, lower_bound, upper_bound)

        # Save convergence information
        convergence_curve.append(global_best_score)

        if verbose:
           print(
        f"Iteration {t + 1}/{max_iterations} | "
        f"Best fitness: {global_best_score:.8f}"
            )

    # Evaluate final population as well
    final_fitness = np.array([
        objective_function(wolf) for wolf in wolves
    ])

    best_index = np.argmin(final_fitness)

    if final_fitness[best_index] < global_best_score:
        global_best_score = final_fitness[best_index]
        global_best_position = wolves[best_index].copy()

    return global_best_position, global_best_score, convergence_curve


# Alias for compatibility with capitalized import (from Gwo import Gwo)
Gwo = gwo


# ==========================================
# 3. EXECUTION
# ==========================================

if __name__ == "__main__":

    best_position, best_fitness, curve = gwo()

    print("\n========== GWO RESULTS ==========")

    print("Best position:", best_position)
    print("Best fitness:", best_fitness)

    print("Theoretical minimum: 0")

    import matplotlib.pyplot as plt

    plt.plot(curve)

    plt.title("GWO Convergence - Rastrigin")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")

    plt.grid(True)
    plt.show()
