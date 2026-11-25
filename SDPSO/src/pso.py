"""
PSO classique — algorithme de référence (baseline).

Chaque particule met à jour sa vitesse avec :
  v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
  x = x + v

Paramètres du papier de référence (Liu & Nishi, 2022) :
  - Poids d'inertie w : 0.9 → 0.2 (décroissance linéaire)
  - Coefficients c1, c2 : 2.0 (standards)
  - Vitesse max : 50 % de l'intervalle de recherche
"""
import numpy as np


def run_pso(func, dim, bounds, popsize=40, max_fes=None, seed=None):
    """
    Lance le PSO classique et retourne les résultats.

    Paramètres
    ----------
    func     : fonction à minimiser, prend un vecteur numpy
    dim      : dimension de l'espace de recherche
    bounds   : (min, max) des variables
    popsize  : nombre de particules
    max_fes  : budget d'évaluations (défaut : 10 000 * dim)
    seed     : graine aléatoire pour la reproductibilité

    Retourne
    --------
    dict avec 'best_value', 'best_position', 'convergence'
    """
    if seed is not None:
        np.random.seed(seed)
    if max_fes is None:
        max_fes = 10_000 * dim

    low, high = bounds
    v_max = 0.5 * (high - low)
    max_iter = int(np.ceil(max_fes / popsize))

    # Initialisation aléatoire des positions et vitesses
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

    for k in range(1, max_iter + 1):
        if fes >= max_fes:
            break

        # Poids d'inertie décroissant linéairement : 0.9 → 0.2
        w = 0.9 - (0.9 - 0.2) * (k / max_iter)

        for i in range(popsize):
            if fes >= max_fes:
                break

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[i] = (w * vel[i]
                      + 2.0 * r1 * (pbest_pos[i] - pos[i])
                      + 2.0 * r2 * (gbest_pos - pos[i]))
            vel[i] = np.clip(vel[i], -v_max, v_max)
            pos[i] = pos[i] + vel[i]

            # On évalue uniquement si la particule reste dans les bornes
            if np.all(pos[i] >= low) and np.all(pos[i] <= high):
                val = func(pos[i])
                fes += 1
                convergence.append(gbest_val)

                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = pos[i].copy()

                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = pos[i].copy()

    return {
        "best_value": gbest_val,
        "best_position": gbest_pos,
        "convergence": np.array(convergence),
    }
