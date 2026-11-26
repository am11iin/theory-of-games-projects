"""
src/knapsack_problem.py
Définition du problème du sac à dos 0/1 et solveur exact par programmation dynamique.
"""

import numpy as np


class KnapsackProblem:
    def __init__(self, values, weights, capacity):
        self.values = np.array(values, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.capacity = float(capacity)
        self.n_items = len(values)
        self.ratios = self.values / self.weights

    def repair_and_evaluate(self, binary_solution):
        sol = binary_solution.copy().astype(int)
        total_weight = np.sum(sol * self.weights)

        if total_weight > self.capacity:
            selected_indices = np.where(sol == 1)[0]
            sorted_indices = selected_indices[np.argsort(self.ratios[selected_indices])]

            for idx in sorted_indices:
                sol[idx] = 0
                total_weight -= self.weights[idx]
                if total_weight <= self.capacity:
                    break

        total_value = np.sum(sol * self.values)
        return sol, total_value, total_weight

    def evaluate_fitness(self, binary_solution):
        _, total_value, _ = self.repair_and_evaluate(binary_solution)
        return -total_value

    def solve_exact_dp(self):
        W = int(self.capacity)
        N = self.n_items
        weights = [int(w) for w in self.weights]
        values = [int(v) for v in self.values]

        dp = [[0] * (W + 1) for _ in range(N + 1)]

        for i in range(1, N + 1):
            w_i = weights[i - 1]
            v_i = values[i - 1]
            for w in range(W + 1):
                if w_i <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)
                else:
                    dp[i][w] = dp[i - 1][w]

        optimal_value = dp[N][W]
        w = W
        best_items = [0] * N
        for i in range(N, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                best_items[i - 1] = 1
                w -= weights[i - 1]

        return optimal_value, best_items


def get_standard_instance():
    values = [
        92, 57, 49, 68, 60, 43, 67, 84, 87, 72,
        46, 48, 20, 24, 76, 55, 61, 35, 27, 45
    ]
    weights = [
        23, 31, 29, 44, 53, 38, 63, 85, 89, 82,
        21, 22, 10, 12, 45, 34, 40, 25, 20, 30
    ]
    capacity = 300
    return KnapsackProblem(values, weights, capacity)
