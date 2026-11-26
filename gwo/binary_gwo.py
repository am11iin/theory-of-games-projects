"""
binary_gwo.py
Implémentation de l'algorithme Grey Wolf Optimizer Binaire (BGWO)
adapté à l'optimisation combinatoire (Sac à dos 0/1).
"""

import numpy as np


def sigmoid(x):
    """
    Fonction de transfert sigmoïde pour projeter les coordonnées
    continues réelles dans l'intervalle [0, 1] (probabilité d'activation).
    """
    # Clip pour éviter les dépassements numériques (overflow) dans exp
    x_clipped = np.clip(x, -10.0, 10.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def binarize(continuous_position):
    """
    Transforme un vecteur continu en vecteur binaire {0, 1}
    à l'aide de la règle stochastique sigmoïde (Mirjalili, 2014).
    """
    probs = sigmoid(continuous_position)
    random_thresholds = np.random.random(len(continuous_position))
    binary_position = (probs >= random_thresholds).astype(int)
    return binary_position


def binary_gwo(
    problem,
    population_size=30,
    max_iterations=100,
    lower_bound=-4.0,
    upper_bound=4.0,
    verbose=True
):
    """
    Exécute Binary Grey Wolf Optimizer sur une instance de problème.

    Paramètres :
        problem : instance de KnapsackProblem
        population_size (int) : nombre de loups dans la meute
        max_iterations (int) : nombre d'itérations
        lower_bound (float) : borne inférieure de l'espace continu
        upper_bound (float) : borne supérieure de l'espace continu
        verbose (bool) : affiche la progression si True

    Retourne :
        best_binary_sol (np.ndarray) : meilleure solution binaire trouvée
        best_value (float) : meilleure valeur de sac à dos obtenue
        best_weight (float) : poids total correspondant
        convergence_curve (list) : historique de la meilleure valeur par itération
    """
    dimensions = problem.n_items

    # 1. Initialisation de la population continue dans [lower_bound, upper_bound]
    wolves = np.random.uniform(
        lower_bound,
        upper_bound,
        (population_size, dimensions)
    )

    # Positions continues des 3 leaders
    alpha_pos = np.zeros(dimensions)
    beta_pos = np.zeros(dimensions)
    delta_pos = np.zeros(dimensions)

    # Fitness des 3 leaders (minimisation, donc initialisé à +l'infini)
    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    # Mémorisation de la meilleure solution binaire valide globale
    global_best_binary = None
    global_best_value = 0.0
    global_best_weight = 0.0

    convergence_curve = []

    # 2. Boucle principale des itérations
    for t in range(max_iterations):

        # Évaluation de chaque loup
        for i in range(population_size):

            # Binarisation stochastique
            bin_sol = binarize(wolves[i])

            # Réparation et évaluation de la valeur
            repaired_sol, value, weight = problem.repair_and_evaluate(bin_sol)

            # En minimisation : fitness = -valeur
            fitness = -value

            # Mise à jour de la hiérarchie : Alpha, Beta, Delta
            if fitness < alpha_score:
                # Delta prend l'ancienne place de Beta
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                # Beta prend l'ancienne place de Alpha
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()

                # Alpha devient le nouveau meilleur
                alpha_score = fitness
                alpha_pos = wolves[i].copy()

                # Mise à jour de la meilleure solution binaire
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

        # Paramètre d'exploration/exploitation 'a' qui décroît linéairement de 2 à 0
        a = 2.0 - 2.0 * (t / max_iterations)

        # 3. Mise à jour des positions des loups
        for i in range(population_size):
            for j in range(dimensions):

                # Guidage par Alpha
                r1, r2 = np.random.random(2)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                # Guidage par Beta
                r1, r2 = np.random.random(2)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                # Guidage par Delta
                r1, r2 = np.random.random(2)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                # Nouvelle position continue candidate (moyenne)
                wolves[i, j] = (X1 + X2 + X3) / 3.0

        # Garder les loups dans les bornes de l'espace continu
        wolves = np.clip(wolves, lower_bound, upper_bound)

        # Historique de convergence (valeur maximisée)
        convergence_curve.append(global_best_value)

        if verbose and (t + 1) % 10 == 0:
            print(f"Itération {t + 1:03d}/{max_iterations} | Meilleure valeur : {global_best_value:.1f} | Poids : {global_best_weight:.1f}")

    return global_best_binary, global_best_value, global_best_weight, convergence_curve


if __name__ == "__main__":
    from knapsack_problem import get_standard_instance

    problem = get_standard_instance()
    opt_val, _ = problem.solve_exact_dp()

    print("Test d'exécution unitaire de Binary GWO...")
    best_bin, best_val, best_w, curve = binary_gwo(
        problem,
        population_size=30,
        max_iterations=100,
        verbose=True
    )

    print("\n========== RÉSULTATS BGWO ==========")
    print(f"Optimum théorique (Exact DP) : {opt_val}")
    print(f"Meilleure valeur trouvée : {best_val}")
    print(f"Poids consommé : {best_w} / {problem.capacity}")
    print(f"Objets retenus : {best_bin.tolist()}")
