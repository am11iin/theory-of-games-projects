"""
Théorie des jeux évolutive pour SDPSO.

Ce module implémente les trois mécanismes décrits par Liu & Nishi (2022) :

1. Calcul des gains (payoffs)
   - Chaque fois qu'une stratégie améliore pbest d'une particule,
     son gain relatif est crédité : Δf = (pbest_ancien - f_nouveau) / |pbest_ancien|

2. Dynamique du réplicateur (Taylor & Jonker, discrète)
   - Toutes les `learning_period` évaluations, les probabilités p_k sont mises à jour :
       p_k ← p_k + α * p_k * (f_k - f̄)
     où f_k est le payoff normalisé de la stratégie k
     et f̄ = Σ p_j * f_j est le payoff moyen du swarm.

3. Seuil de mutation (anti-extinction)
   - p_k = max(p_k, 0.02) pour éviter qu'une stratégie disparaisse définitivement.

4. Redémarrage périodique (restart)
   - À 25 %, 50 %, 75 % du budget d'évaluations, p redevient [0.25, 0.25, 0.25, 0.25].
"""
import numpy as np

STRATEGY_NAMES = ["LIPS", "UPSO", "LDWPSO", "CLPSO"]
N_STRATEGIES = 4
MUTATION_THRESHOLD = 0.02  # Seuil anti-extinction (Section 3.3 du papier)


class GameDynamics:
    """
    Gère les probabilités d'adoption des 4 stratégies de recherche.
    """

    def __init__(self, alpha=0.1, learning_period=200, restart_every=4):
        """
        alpha          : taux d'apprentissage de l'équation du réplicateur
        learning_period: nombre d'évaluations entre deux mises à jour
        restart_every  : nombre de restarts sur toute la durée (ex: 4 → à 25%, 50%, 75%)
        """
        self.alpha = alpha
        self.learning_period = learning_period
        self.restart_every = restart_every

        # p[k] = probabilité d'adoption de la stratégie k
        self.p = np.ones(N_STRATEGIES) / N_STRATEGIES

        # Mémoires de performance pour la période en cours
        self.gains = np.zeros(N_STRATEGIES)       # Somme des améliorations relatives
        self.successes = np.zeros(N_STRATEGIES)   # Nombre de succès
        self.failures = np.zeros(N_STRATEGIES)    # Nombre d'échecs

        # Historique pour les graphiques
        self.history_p = [self.p.copy()]
        self.history_fes = [0]

    def select_strategy(self):
        """Sélection par roulette selon les probabilités courantes p."""
        r = np.random.rand()
        cumul = 0.0
        for k, pk in enumerate(self.p):
            cumul += pk
            if r <= cumul:
                return k
        return N_STRATEGIES - 1

    def record_outcome(self, strategy_k, old_pbest, new_fitness):
        """
        Enregistre le résultat d'un déplacement utilisant la stratégie k.
        Si new_fitness < old_pbest → succès, le gain relatif est crédité.
        Sinon → échec.
        """
        if new_fitness < old_pbest:
            denom = abs(old_pbest) if abs(old_pbest) > 1e-14 else 1.0
            gain = (old_pbest - new_fitness) / denom
            self.gains[strategy_k] += max(0.0, gain)
            self.successes[strategy_k] += 1
        else:
            self.failures[strategy_k] += 1

    def _update_replicator(self, current_fes):
        """
        Applique l'équation discrète du réplicateur, puis mutation et renormalisation.
        Appelée toutes les `learning_period` évaluations.
        """
        total = self.successes + self.failures
        total = np.where(total == 0, 1.0, total)

        # Payoff moyen par tentative pour chaque stratégie
        payoffs = self.gains / total

        # Si aucune stratégie n'a produit de gain, on garde l'équipartition
        if np.all(payoffs == 0):
            payoffs = np.ones(N_STRATEGIES)
        else:
            # Les stratégies sans gain reçoivent un micro-payoff pour rester vivantes
            min_nonzero = np.min(payoffs[payoffs > 0])
            payoffs[payoffs == 0] = 0.1 * min_nonzero

        # Normalisation des payoffs en distribution f
        f = payoffs / payoffs.sum()

        # Payoff moyen de la population : f̄ = Σ p_k * f_k
        f_bar = np.dot(self.p, f)

        # Équation du réplicateur : p_k ← p_k + α * p_k * (f_k - f̄)
        self.p = self.p + self.alpha * self.p * (f - f_bar)

        # Seuil de mutation : p_k ≥ 0.02 (anti-extinction)
        self.p = np.maximum(self.p, MUTATION_THRESHOLD)

        # Renormalisation pour que Σ p_k = 1
        self.p /= self.p.sum()

        # Remise à zéro des mémoires pour la prochaine période
        self.gains[:] = 0
        self.successes[:] = 0
        self.failures[:] = 0

        self.history_p.append(self.p.copy())
        self.history_fes.append(current_fes)

    def _check_restart(self, current_fes, max_fes):
        """
        Réinitialise p à l'équipartition à 25 %, 50 %, 75 % du budget total.
        """
        step = max_fes // self.restart_every
        if step > 0 and current_fes % step == 0:
            self.p = np.ones(N_STRATEGIES) / N_STRATEGIES
            self.history_p.append(self.p.copy())
            self.history_fes.append(current_fes)

    def step(self, current_fes, max_fes):
        """
        À appeler après chaque évaluation de la fonction objectif.
        Vérifie les déclencheurs de restart et de mise à jour du réplicateur.
        """
        self._check_restart(current_fes, max_fes)
        if current_fes % self.learning_period == 0:
            self._update_replicator(current_fes)
