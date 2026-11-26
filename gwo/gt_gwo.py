"""
gt_gwo.py
Implémentation de l'algorithme hybride GT-GWO (Game Theory Grey Wolf Optimizer)
pour les problèmes continus (Sphere, Rastrigin).
"""

import numpy as np
from game_theory_model import ReplicatorDynamics, levy_flight, STRATEGY_NAMES


def objective_rastrigin(x):
    """Fonction de test de Rastrigin (multimodale avec de nombreux minima locaux)."""
    n = len(x)
    return 10.0 * n + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))


def objective_sphere(x):
    """Fonction de test Sphere (unimodale)."""
    return np.sum(x**2)


def gt_gwo(
    objective_function=objective_rastrigin,
    population_size=30,
    dimensions=2,
    lower_bound=-5.12,
    upper_bound=5.12,
    max_iterations=100,
    learning_rate=0.15,
    verbose=False
):
    """
    Exécute l'algorithme hybride GT-GWO.

    Retourne :
        global_best_position (np.ndarray) : meilleure solution trouvée
        global_best_score (float) : meilleure fitness obtenue
        convergence_curve (list) : historique de convergence
        strategy_history (np.ndarray) : historique des probabilités des stratégies
    """
    # Initialisation du moteur de théorie des jeux évolutive
    game = ReplicatorDynamics(learning_rate=learning_rate)

    # Initialisation uniforme de la meute
    wolves = np.random.uniform(
        lower_bound,
        upper_bound,
        (population_size, dimensions)
    )

    # Évaluation initiale de la population
    fitness = np.array([objective_function(w) for w in wolves], dtype=float)

    # Tri pour identifier Alpha, Beta, Delta
    sorted_idx = np.argsort(fitness)
    alpha = wolves[sorted_idx[0]].copy()
    beta = wolves[sorted_idx[1]].copy()
    delta = wolves[sorted_idx[2]].copy()

    alpha_score = fitness[sorted_idx[0]]
    beta_score = fitness[sorted_idx[1]]
    delta_score = fitness[sorted_idx[2]]

    global_best_position = alpha.copy()
    global_best_score = alpha_score

    convergence_curve = []

    # Boucle principale
    for t in range(max_iterations):

        # Paramètre d'exploration standard 'a' décroissant de 2 à 0
        a = 2.0 - 2.0 * (t / max_iterations)

        for i in range(population_size):

            # 1. Le loup sélectionne sa stratégie via le jeu évolutif
            k = game.select_strategy()

            old_f = fitness[i]
            x_cand = np.zeros(dimensions)

            # 2. Exécution de la stratégie choisie
            if k == 0:
                # STRATÉGIE 0 : TRIO_COOP (Chasse coopérative classique GWO)
                for j in range(dimensions):
                    r1, r2 = np.random.random(2)
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_a = abs(C1 * alpha[j] - wolves[i, j])
                    X1 = alpha[j] - A1 * D_a

                    r1, r2 = np.random.random(2)
                    A2 = 2.0 * a * r1 - a
                    C2 = 2.0 * r2
                    D_b = abs(C2 * beta[j] - wolves[i, j])
                    X2 = beta[j] - A2 * D_b

                    r1, r2 = np.random.random(2)
                    A3 = 2.0 * a * r1 - a
                    C3 = 2.0 * r2
                    D_d = abs(C3 * delta[j] - wolves[i, j])
                    X3 = delta[j] - A3 * D_d

                    x_cand[j] = (X1 + X2 + X3) / 3.0

            elif k == 1:
                # STRATÉGIE 1 : ALPHA_EXPLOIT (Attaque ciblée / Exploitation intensive d'Alpha)
                for j in range(dimensions):
                    r1, r2 = np.random.random(2)
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_a = abs(C1 * alpha[j] - wolves[i, j])
                    x_cand[j] = alpha[j] - A1 * D_a

            elif k == 2:
                # STRATÉGIE 2 : LEVY_EXPLORE (Éclaireur / Saut de Lévy anti-piège local)
                step = levy_flight(dimensions)
                # Déplacement longue portée autour de la position actuelle
                x_cand = wolves[i] + 0.05 * step * (wolves[i] - alpha)

            elif k == 3:
                # STRATÉGIE 3 : SOCIAL_PEER (Entraide horizontale entre deux pairs aléatoires)
                candidates = [idx for idx in range(population_size) if idx != i]
                r1_idx, r2_idx = np.random.choice(candidates, 2, replace=False)
                phi = np.random.uniform(0.0, 1.0, dimensions)
                x_cand = wolves[i] + phi * (wolves[r1_idx] - wolves[r2_idx])

            # Respect des bornes de l'espace de recherche
            x_cand = np.clip(x_cand, lower_bound, upper_bound)

            # Évaluation de la nouvelle position
            new_f = objective_function(x_cand)

            # Enregistrement du résultat dans le jeu pour la dynamique du réplicateur
            game.record_payoff(strategy_k=k, old_fitness=old_f, new_fitness=new_f)

            # Règle d'acceptation gloutonne
            if new_f < old_f:
                wolves[i] = x_cand.copy()
                fitness[i] = new_f

        # 3. Mise à jour des leaders Alpha, Beta, Delta
        sorted_idx = np.argsort(fitness)
        alpha = wolves[sorted_idx[0]].copy()
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()

        alpha_score = fitness[sorted_idx[0]]
        beta_score = fitness[sorted_idx[1]]
        delta_score = fitness[sorted_idx[2]]

        if alpha_score < global_best_score:
            global_best_score = alpha_score
            global_best_position = alpha.copy()

        # 4. Mise à jour de la population des stratégies via l'équation du réplicateur
        game.update_population()

        convergence_curve.append(global_best_score)

        if verbose and (t + 1) % 10 == 0:
            probs_str = " | ".join([f"{STRATEGY_NAMES[k]}: {game.p[k]:.2f}" for k in range(len(STRATEGY_NAMES))])
            print(f"Itér {t + 1:03d}/{max_iterations} | Best: {global_best_score:.8e} | Stratégies -> {probs_str}")

    strategy_history = np.array(game.history_p)
    return global_best_position, global_best_score, convergence_curve, strategy_history


if __name__ == "__main__":
    print("Test unitaire de GT-GWO sur Rastrigin...")
    pos, score, curve, strat_hist = gt_gwo(
        objective_function=objective_rastrigin,
        population_size=30,
        dimensions=2,
        max_iterations=100,
        verbose=True
    )
    print("\n========== RÉSULTATS DU TEST ==========")
    print(f"Meilleure position : {pos}")
    print(f"Meilleure fitness : {score}")
    print(f"Probabilités finales des stratégies : {strat_hist[-1]}")
