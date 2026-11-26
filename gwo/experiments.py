
import numpy as np
import time
import matplotlib.pyplot as plt

from Gwo import gwo, Gwo


# ==========================================
# 1. EXPERIMENT CONFIGURATION
# ==========================================

NUMBER_OF_RUNS = 30

POPULATION_SIZE = 30
DIMENSIONS = 2
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
MAX_ITERATIONS = 100

# A run is successful if fitness <= this threshold
SUCCESS_THRESHOLD = 1e-8


# ==========================================
# 2. RUN EXPERIMENTS
# ==========================================

results = []
execution_times = []
convergence_curves = []
best_positions = []

print("Starting GWO experiments on Rastrigin...")
print("------------------------------------------")

for run in range(NUMBER_OF_RUNS):

    # Different reproducible random sequence for each run
    np.random.seed(42 + run)

    start_time = time.perf_counter()

    best_position, best_fitness, curve = gwo(
        population_size=POPULATION_SIZE,
        dimensions=DIMENSIONS,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        max_iterations=MAX_ITERATIONS,
        verbose=False
    )

    end_time = time.perf_counter()

    runtime = end_time - start_time

    results.append(best_fitness)
    execution_times.append(runtime)
    convergence_curves.append(curve)
    best_positions.append(best_position)

    print(
        f"Run {run + 1:02d}/{NUMBER_OF_RUNS} | "
        f"Best fitness: {best_fitness:.10f} | "
        f"Time: {runtime:.4f}s"
    )


# ==========================================
# 3. STATISTICAL ANALYSIS
# ==========================================

results = np.array(results)
execution_times = np.array(execution_times)

best_result = np.min(results)
worst_result = np.max(results)
mean_result = np.mean(results)
std_result = np.std(results, ddof=1)

success_count = np.sum(results <= SUCCESS_THRESHOLD)
success_rate = (success_count / NUMBER_OF_RUNS) * 100

mean_runtime = np.mean(execution_times)
total_runtime = np.sum(execution_times)


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\n========== EXPERIMENT RESULTS ==========")

print(f"Number of runs: {NUMBER_OF_RUNS}")

print(f"Best fitness: {best_result:.12e}")
print(f"Worst fitness: {worst_result:.12e}")
print(f"Mean fitness: {mean_result:.12e}")
print(f"Standard deviation: {std_result:.12e}")

print(f"\nSuccessful runs: {success_count}/{NUMBER_OF_RUNS}")
print(f"Success rate: {success_rate:.2f}%")

print(f"\nMean execution time: {mean_runtime:.6f} seconds")
print(f"Total execution time: {total_runtime:.6f} seconds")


# ==========================================
# 5. AVERAGE CONVERGENCE CURVE
# ==========================================

convergence_curves = np.array(convergence_curves)

mean_curve = np.mean(convergence_curves, axis=0)

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, MAX_ITERATIONS + 1),
    mean_curve,
    label="Mean convergence",
    linewidth=2
)

plt.title("GWO - Average Convergence on Rastrigin")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("rastrigin_mean_convergence.png", dpi=300)

plt.show()


# ==========================================
# 6. SAVE RESULTS TO CSV
# ==========================================

import csv

with open("rastrigin_experiments.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Run",
        "Best Fitness",
        "Runtime (seconds)",
        "Best Position"
    ])

    for i in range(NUMBER_OF_RUNS):

        writer.writerow([
            i + 1,
            results[i],
            execution_times[i],
            best_positions[i].tolist()
        ])

print("\nResults saved to rastrigin_experiments.csv")
