"""
knapsack_problem.py
Définition du problème du sac à dos 0/1 (0/1 Knapsack Problem)
et calcul de la solution optimale exacte par programmation dynamique.
"""

import numpy as np


class KnapsackProblem:
    """
    Représente une instance du problème du sac à dos 0/1.
    """

    def __init__(self, values, weights, capacity):
        self.values = np.array(values, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.capacity = float(capacity)
        self.n_items = len(values)
        
        # Ratios valeur / poids pour la stratégie de réparation gloutonne
        self.ratios = self.values / self.weights

    def repair_and_evaluate(self, binary_solution):
        """
        Vérifie si la solution binaire respecte la contrainte de capacité.
        Si la capacité est dépassée, effectue une réparation gloutonne
        en retirant les objets de plus faible ratio valeur/poids.

        Retourne :
            repaired_solution (np.ndarray) : solution binaire valide
            total_value (float) : valeur totale obtenue
            total_weight (float) : poids total consommé
        """
        sol = binary_solution.copy().astype(int)
        total_weight = np.sum(sol * self.weights)

        # Si dépassement de capacité : réparation
        if total_weight > self.capacity:
            # Indices des objets sélectionnés, triés par ratio croissant (les moins rentables en premier)
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
        """
        Calcule la fitness pour GWO (minimisation).
        Puisque nous maximisons la valeur, fitness = -valeur_totale.
        """
        _, total_value, _ = self.repair_and_evaluate(binary_solution)
        return -total_value

    def solve_exact_dp(self):
        """
        Résout le problème du sac à dos exactement en utilisant
        la programmation dynamique (pour connaître l'optimum théorique).
        Nécessite des poids et une capacité entiers.
        """
        W = int(self.capacity)
        N = self.n_items
        weights = [int(w) for w in self.weights]
        values = [int(v) for v in self.values]

        # Table DP : dp[i][w] = valeur max avec les i premiers objets et capacité w
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

        # Reconstruction de la solution optimale
        w = W
        best_items = [0] * N
        for i in range(N, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                best_items[i - 1] = 1
                w -= weights[i - 1]

        return optimal_value, best_items


def get_standard_instance():
    """
    Retourne une instance standard de 20 objets (benchmark classique).
    """
    # 20 objets : valeurs et poids
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


if __name__ == "__main__":
    problem = get_standard_instance()
    opt_val, opt_items = problem.solve_exact_dp()

    total_w = sum(opt_items[i] * problem.weights[i] for i in range(problem.n_items))

    print("========== INSTANCE BENCHMARK DU SAC À DOS ==========")
    print(f"Nombre d'objets : {problem.n_items}")
    print(f"Capacité max : {problem.capacity}")
    print(f"Valeur optimale théorique (Exact DP) : {opt_val}")
    print(f"Poids total optimal : {total_w} / {problem.capacity}")
    print(f"Objets sélectionnés : {opt_items}")
