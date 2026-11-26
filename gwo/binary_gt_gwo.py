"""
binary_gt_gwo.py
Algorithme hybride Binary GT-GWO (Game Theory Grey Wolf Optimizer Binaire)
pour le problème combinatoire du sac à dos 0/1.
"""

import numpy as np
from game_theory_model import ReplicatorDynamics, levy_flight, STRATEGY_NAMES
from binary_gwo import sigmoid, binarize


def binary_gt_gwo(
    problem,
    population_size=30,
    max_iterations=100,
    lower_bound=-4.0,
    upper_bound=4.0,
    learning_rate=0.15,
    verbose=False
):
    """
    Exécute l'algorithme Binary GT-GWO sur une instance de sac à dos.

    Retourne :
        global_best_binary (np.ndarray) : meilleure solution binaire retenue
        global_best_value (float) : meilleure valeur obtenue
        global_best_weight (float) : poids consommé
        convergence_curve (list) : historique de la meilleure valeur
        strategy_history (np.ndarray) : historique des probabilités des stratégies
    """
    dimensions = problem.n_items
    game = ReplicatorDynamics(learning_rate=learning_rate)

    # Initialisation de la population continue
    wolves = np.random.uniform(
        lower_bound,
        upper_bound,
        (population_size, dimensions)
    )

    # Évaluation initiale
    values = np.zeros(population_size)
    weights = np.zeros(population_size)
    binary_solutions = []

    for i in range(population_size):
        b_sol = binarize(wolves[i])
        r_sol, val, w = problem.repair_and_evaluate(b_sol)
        values[i] = val
        weights[i] = w
        binary_solutions.append(r_sol)

    # En maximisation, la meilleure valeur est la plus haute (fitness négative la plus basse)
    fitness = -values

    sorted_idx = np.argsort(fitness)
    alpha_pos = wolves[sorted_idx[0]].copy()
    beta_pos = wolves[sorted_idx[1]].copy()
    delta_pos = wolves[sorted_idx[2]].copy()

    global_best_binary = binary_solutions[sorted_idx[0]].copy()
    global_best_value = values[sorted_idx[0]]
    global_best_weight = weights[sorted_idx[0]]

    convergence_curve = []

    # Boucle des itérations
    for t in range(max_iterations):

        a = 2.0 - 2.0 * (t / max_iterations)

        for i in range(population_size):

            k = game.select_strategy()
            old_val = values[i]
            cand_cont = np.zeros(dimensions)

            if k == 0:
                # 0 : TRIO_COOP (Chasse standard GWO)
                for j in range(dimensions):
                    r1, r2 = np.random.random(2)
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_a = abs(C1 * alpha_pos[j] - wolves[i, j])
                    X1 = alpha_pos[j] - A1 * D_a

                    r1, r2 = np.random.random(2)
                    A2 = 2.0 * a * r1 - a
                    C2 = 2.0 * r2
                    D_b = abs(C2 * beta_pos[j] - wolves[i, j])
                    X2 = beta_pos[j] - A2 * D_b

                    r1, r2 = np.random.random(2)
                    A3 = 2.0 * a * r1 - a
                    C3 = 2.0 * r2
                    D_d = abs(C3 * delta_pos[j] - wolves[i, j])
                    X3 = delta_pos[j] - A3 * D_d

                    cand_cont[j] = (X1 + X2 + X3) / 3.0

            elif k == 1:
                # 1 : ALPHA_EXPLOIT (Attaque ciblée sur Alpha)
                for j in range(dimensions):
                    r1, r2 = np.random.random(2)
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_a = abs(C1 * alpha_pos[j] - wolves[i, j])
                    cand_cont[j] = alpha_pos[j] - A1 * D_a

            elif k == 2:
                # 2 : LEVY_EXPLORE (Exploration longue portée pour sortir des minima locaux)
                step = levy_flight(dimensions)
                cand_cont = wolves[i] + 0.1 * step * (wolves[i] - alpha_pos)

            elif k == 3:
                # 3 : SOCIAL_PEER (Interaction entre pairs de la meute)
                candidates = [idx for idx in range(population_size) if idx != i]
                r1_idx, r2_idx = np.random.choice(candidates, 2, replace=False)
                phi = np.random.uniform(0.0, 1.0, dimensions)
                cand_cont = wolves[i] + phi * (wolves[r1_idx] - wolves[r2_idx])

            cand_cont = np.clip(cand_cont, lower_bound, upper_bound)

            # Binarisation et réparation
            cand_bin = binarize(cand_cont)
            repaired_bin, new_val, new_w = problem.repair_and_evaluate(cand_bin)

            # Calcul du gain dans la dynamique du jeu (maximisation : new_val > old_val)
            # En minimisation équivalente : old_f = -old_val, new_f = -new_val
            game.record_payoff(strategy_k=k, old_fitness=-old_val, new_fitness=-new_val)

            # Règle d'acceptation gloutonne
            if new_val >= old_val:
                wolves[i] = cand_cont.copy()
                values[i] = new_val
                weights[i] = new_w
                binary_solutions[i] = repaired_bin.copy()

        # Mise à jour des leaders
        fitness = -values
        sorted_idx = np.argsort(fitness)
        alpha_pos = wolves[sorted_idx[0]].copy()
        beta_pos = wolves[sorted_idx[1]].copy()
        delta_pos = wolves[sorted_idx[2]].copy()

        if values[sorted_idx[0]] > global_best_value:
            global_best_value = values[sorted_idx[0]]
            global_best_weight = weights[sorted_idx[0]]
            global_best_binary = binary_solutions[sorted_idx[0]].copy()

        # Mise à jour de la population des stratégies
        game.update_population()

        convergence_curve.append(global_best_value)

        if verbose and (t + 1) % 10 == 0:
            print(f"Itér {t + 1:03d}/{max_iterations} | Valeur max : {global_best_value:.1f} | Poids : {global_best_weight:.1f}")

    strategy_history = np.array(game.history_p)
    return global_best_binary, global_best_value, global_best_weight, convergence_curve, strategy_history


if __name__ == "__main__":
    from knapsack_problem import get_standard_instance

    problem = get_standard_instance()
    opt_val, _ = problem.solve_exact_dp()

    print("Test unitaire de Binary GT-GWO sur le sac à dos...")
    best_bin, best_val, best_w, curve, strat_hist = binary_gt_gwo(
        problem,
        population_size=30,
        max_iterations=100,
        verbose=True
    )

    print("\n========== RÉSULTATS BGT-GWO ==========")
    print(f"Optimum théorique : {opt_val}")
    print(f"Valeur trouvée : {best_val}")
    print(f"Poids : {best_w} / {problem.capacity}")
    print(f"Probabilités finales des stratégies : {strat_hist[-1]}")
