"""
SDPSO — Strategy Dynamics Particle Swarm Optimizer.
Implémentation Python fidèle de Liu & Nishi (2022), Information Sciences.

Architecture de la population (N = 40 par défaut) :
  - Sous-population 1 (30 %) : CLPSO permanent → réservoir de diversité exploratoire.
  - Sous-population 2 (70 %) : Sélection dynamique par théorie des jeux évolutive.
    À chaque itération, chaque particule tire une stratégie selon p = [p1, p2, p3, p4]
    mis à jour par la dynamique du réplicateur.

Les 4 stratégies en compétition :
  0 - LIPS   : voisinage spatial local (sans gbest)
  1 - UPSO   : compromis global/local via topologie en anneau
  2 - LDWPSO : PSO standard avec inertie et coefficients variables dans le temps
  3 - CLPSO  : apprentissage par dimension à partir des pbest d'autres particules
"""
import numpy as np
from .game_theory import GameDynamics, N_STRATEGIES


# ---------------------------------------------------------------------------
# Outils CLPSO : exemplaires et rafraîchissement
# ---------------------------------------------------------------------------

def _compute_learning_proba(n):
    """
    Calcule les probabilités d'apprentissage Pc pour n particules.
    Formule du papier : Pc(i) = 0.5 * (exp(10*i/(n-1)) - 1) / (exp(10) - 1)
    """
    j = np.linspace(0, 10, n)
    exp_j = np.exp(j)
    return 0.5 * (exp_j - exp_j[0]) / (exp_j[-1] - exp_j[0])


def _init_exemplars(n_particles, dim, pool_pbest_pos, pool_pbest_val, Pc):
    """
    Initialise les exemplaires CLPSO pour n_particles particules.
    Pour chaque particule i et dimension d, on sélectionne par tournoi
    l'indice du pbest dont on va apprendre.
    """
    pool_size = len(pool_pbest_val)
    fri_best_pos = np.zeros((n_particles, dim))

    for i in range(n_particles):
        fri_best_pos[i] = _refresh_one(i, dim, pool_size, pool_pbest_pos, pool_pbest_val, Pc)

    return fri_best_pos


def _refresh_one(i, dim, pool_size, pool_pbest_pos, pool_pbest_val, Pc):
    """
    Rafraîchit les exemplaires d'une particule i par sélection par tournoi.
    Retourne le vecteur fri_best_pos[i] mis à jour.
    """
    c1 = np.random.randint(0, pool_size, dim)
    c2 = np.random.randint(0, pool_size, dim)
    winner = np.where(pool_pbest_val[c1] < pool_pbest_val[c2], c1, c2)

    # toss : True → apprendre de soi-même, False → apprendre du gagnant du tournoi
    toss = np.random.rand(dim) > Pc[i % len(Pc)]
    exemplars = np.where(toss, i % pool_size, winner)

    # Si toutes les dims apprennent de soi-même, en forcer au moins une vers un autre
    if np.all(exemplars == i % pool_size):
        exemplars[np.random.randint(dim)] = winner[np.random.randint(dim)]

    return pool_pbest_pos[exemplars, np.arange(dim)]


# ---------------------------------------------------------------------------
# Mises à jour de vitesse pour chaque stratégie
# ---------------------------------------------------------------------------

def _vel_lips(i, pos, vel, pbest_pos, chi=0.7298, nsize=3):
    """
    LIPS : voisinage local basé sur la distance euclidienne aux pbest.
    Calcule FIP = moyenne pondérée des nsize pbest les plus proches.
    χ = 0.7298 est le facteur de constriction standard (Clerc & Kennedy, 2002).
    """
    dists = np.linalg.norm(pbest_pos - pos[i], axis=1)
    dists[i] = np.inf  # Exclure soi-même
    neighbors = np.argsort(dists)[:nsize]

    fi = np.random.uniform(0, 4.1 / nsize, (nsize, pos.shape[1]))
    sum_fi = fi.sum(axis=0)
    fip = (fi * pbest_pos[neighbors]).sum(axis=0) / sum_fi

    return chi * (vel[i] + sum_fi * (fip - pos[i]))


def _vel_upso(i, pos, vel, pbest_pos, pbest_val, gbest_pos, w, c1, c2, u=0.1):
    """
    UPSO : unifie la vitesse globale (vers gbest) et locale (vers lbest en anneau).
    u = 0.1 est le facteur d'unification.
    La perturbation r ~ N(0, 0.01²) est issue du code MATLAB original.
    """
    n = len(pbest_val)
    dim = pos.shape[1]
    left, right = (i - 1) % n, (i + 1) % n
    lbest = left if pbest_val[left] < pbest_val[right] else right

    r1, r2 = np.random.rand(dim), np.random.rand(dim)
    vel1 = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])

    r3, r4 = np.random.rand(dim), np.random.rand(dim)
    vel2 = w * vel[i] + c1 * r3 * (pbest_pos[i] - pos[i]) + c2 * r4 * (pbest_pos[lbest] - pos[i])

    noise = np.random.normal(0, 0.01, dim)
    return noise * u * vel1 + (1 - u) * vel2


def _vel_ldwpso(i, pos, vel, pbest_pos, gbest_pos, w, c1, c2):
    """
    LDWPSO : PSO standard avec poids d'inertie et coefficients variables dans le temps.
    Les paramètres w, c1, c2 évoluent linéairement dans la boucle principale.
    """
    dim = pos.shape[1]
    r1, r2 = np.random.rand(dim), np.random.rand(dim)
    return w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])


def _vel_clpso(i_local, pos_sub, vel_sub, fri_best_pos, w, c):
    """
    CLPSO : apprentissage par dimension à partir d'exemplaires préalablement sélectionnés.
    i_local est l'indice dans la sous-population.
    """
    dim = pos_sub.shape[1]
    r = np.random.rand(dim)
    return w * vel_sub[i_local] + c * r * (fri_best_pos[i_local] - pos_sub[i_local])


# ---------------------------------------------------------------------------
# Algorithme SDPSO principal
# ---------------------------------------------------------------------------

def run_sdpso(func, dim, bounds, popsize=40, max_fes=None, alpha=0.1,
              learning_period=200, restart_every=4, seed=None):
    """
    Lance SDPSO et retourne les résultats.

    Paramètres
    ----------
    func            : fonction à minimiser, prend un vecteur numpy
    dim             : dimension de l'espace de recherche
    bounds          : (min, max) des variables
    popsize         : taille du swarm (défaut : 40, comme dans le papier)
    max_fes         : budget d'évaluations (défaut : 10 000 * dim)
    alpha           : taux d'apprentissage de l'équation du réplicateur (0.1)
    learning_period : fréquence de mise à jour du réplicateur (toutes les 200 FES)
    restart_every   : nombre de restarts sur la durée totale (4 → à 25%, 50%, 75%)
    seed            : graine aléatoire

    Retourne
    --------
    dict avec 'best_value', 'best_position', 'convergence',
              'strategy_proba_history', 'strategy_fes_history', 'strategy_usage'
    """
    if seed is not None:
        np.random.seed(seed)
    if max_fes is None:
        max_fes = 10_000 * dim

    low, high = bounds
    v_max = 0.5 * (high - low)
    max_iter = int(np.ceil(max_fes / popsize))

    # Taille des deux sous-populations (30% / 70%)
    n1 = int(np.floor(0.3 * popsize))   # Sous-pop 1 : CLPSO permanent
    n2 = popsize - n1                    # Sous-pop 2 : dynamique EGT

    # Initialisation du swarm
    pos = np.random.uniform(low, high, (popsize, dim))
    vel = np.random.uniform(-v_max, v_max, (popsize, dim))

    # Évaluation initiale
    pbest_pos = pos.copy()
    pbest_val = np.array([func(pos[i]) for i in range(popsize)])
    fes = popsize

    best_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[best_idx].copy()
    gbest_val = float(pbest_val[best_idx])

    convergence = [gbest_val] * fes

    # Initialisation CLPSO pour les deux sous-populations
    Pc1 = _compute_learning_proba(n1)
    Pc2 = _compute_learning_proba(n2)

    fri1 = _init_exemplars(n1, dim, pbest_pos[:n1], pbest_val[:n1], Pc1)
    fri2 = _init_exemplars(n2, dim, pbest_pos, pbest_val, Pc2)

    stagnation1 = np.zeros(n1, dtype=int)   # Compteur de stagnation sous-pop 1
    stagnation2 = np.zeros(n2, dtype=int)   # Compteur de stagnation sous-pop 2
    REFRESH_GAP = 5                          # Seuil de rafraîchissement CLPSO

    # Module de théorie des jeux
    game = GameDynamics(alpha=alpha, learning_period=learning_period, restart_every=restart_every)
    strategy_usage = np.zeros(N_STRATEGIES, dtype=int)

    for k in range(1, max_iter + 1):
        if fes >= max_fes:
            break

        # Paramètres temporellement variables (mêmes formules que le MATLAB original)
        t = k / max_iter
        w   = 0.9 - 0.7 * t          # Inertie : 0.9 → 0.2
        c_a = 2.5 - 2.0 * t          # c1(t) : 2.5 → 0.5
        c_b = 0.5 + 2.0 * t          # c2(t) : 0.5 → 2.5
        c_clpso1 = 3.0 - 1.5 * t     # Coefficient CLPSO sous-pop 1 : 3.0 → 1.5

        # ------------------------------------------------------------------
        # Sous-population 1 : CLPSO permanent (exploration)
        # ------------------------------------------------------------------
        for j in range(n1):
            if fes >= max_fes:
                break

            # Rafraîchir les exemplaires si stagnation
            if stagnation1[j] > REFRESH_GAP:
                fri1[j] = _refresh_one(j, dim, n1, pbest_pos[:n1], pbest_val[:n1], Pc1)
                stagnation1[j] = 0

            new_v = _vel_clpso(j, pos[:n1], vel[:n1], fri1, w, c_clpso1)
            vel[j] = np.clip(new_v, -v_max, v_max)
            pos[j] = pos[j] + vel[j]

            if np.all(pos[j] >= low) and np.all(pos[j] <= high):
                val = func(pos[j])
                fes += 1
                game.step(fes, max_fes)
                convergence.append(gbest_val)

                if val < pbest_val[j]:
                    pbest_val[j] = val
                    pbest_pos[j] = pos[j].copy()
                    stagnation1[j] = 0
                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = pos[j].copy()
                else:
                    stagnation1[j] += 1

        # ------------------------------------------------------------------
        # Sous-population 2 : sélection dynamique par théorie des jeux (EGT)
        # ------------------------------------------------------------------
        for jj in range(n2):
            if fes >= max_fes:
                break

            i = n1 + jj   # Indice global dans le swarm

            # Tirage de la stratégie par roulette selon p
            strategy = game.select_strategy()
            strategy_usage[strategy] += 1

            if strategy == 0:
                new_v = _vel_lips(i, pos, vel, pbest_pos)

            elif strategy == 1:
                new_v = _vel_upso(i, pos, vel, pbest_pos, pbest_val, gbest_pos, w, c_a, c_b)

            elif strategy == 2:
                new_v = _vel_ldwpso(i, pos, vel, pbest_pos, gbest_pos, w, c_a, c_b)

            else:  # strategy == 3 : CLPSO
                if stagnation2[jj] > REFRESH_GAP:
                    fri2[jj] = _refresh_one(jj, dim, popsize, pbest_pos, pbest_val, Pc2)
                    stagnation2[jj] = 0
                new_v = _vel_clpso(jj, pos[n1:], vel[n1:], fri2, w, c_a)

            vel[i] = np.clip(new_v, -v_max, v_max)
            pos[i] = pos[i] + vel[i]

            if np.all(pos[i] >= low) and np.all(pos[i] <= high):
                val = func(pos[i])
                fes += 1
                game.step(fes, max_fes)
                convergence.append(gbest_val)

                # Enregistrer le gain pour le réplicateur
                game.record_outcome(strategy, pbest_val[i], val)

                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = pos[i].copy()
                    stagnation2[jj] = 0
                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = pos[i].copy()
                else:
                    stagnation2[jj] += 1

    return {
        "best_value": gbest_val,
        "best_position": gbest_pos,
        "convergence": np.array(convergence),
        "strategy_proba_history": np.array(game.history_p),
        "strategy_fes_history": np.array(game.history_fes),
        "strategy_usage": strategy_usage,
    }
