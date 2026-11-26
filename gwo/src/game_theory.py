"""
src/game_theory.py
Modèle de Théorie des Jeux Évolutive (EGT - Evolutionary Game Theory)
basé sur la dynamique du réplicateur de Taylor & Jonker (1978).
"""

import numpy as np


STRATEGY_NAMES = [
    "TRIO_COOP",      # 0 : Chasse classique du trio (Alpha + Beta + Delta) / 3
    "ALPHA_EXPLOIT",  # 1 : Exploitation ciblée sur Alpha
    "LEVY_EXPLORE",   # 2 : Exploration / Échappement par vol de Lévy
    "SOCIAL_PEER"     # 3 : Interaction sociale horizontale entre pairs (X_r1 - X_r2)
]

N_STRATEGIES = len(STRATEGY_NAMES)
MUTATION_THRESHOLD = 0.05


def levy_flight(dimensions, beta=1.5):
    """
    Génère un pas de vol de Lévy selon l'algorithme de Mantegna (1994).
    """
    from math import gamma, sin, pi

    numerator = gamma(1.0 + beta) * sin(pi * beta / 2.0)
    denominator = gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (numerator / denominator) ** (1.0 / beta)
    sigma_v = 1.0

    u = np.random.normal(0, sigma_u, size=dimensions)
    v = np.random.normal(0, sigma_v, size=dimensions)

    step = u / (np.abs(v) ** (1.0 / beta))
    return step


class ReplicatorDynamics:
    """
    Gère la dynamique de population des stratégies selon l'équation du réplicateur.
    """

    def __init__(self, learning_rate=0.15, mutation_threshold=MUTATION_THRESHOLD):
        self.learning_rate = learning_rate
        self.mutation_threshold = mutation_threshold
        self.p = np.ones(N_STRATEGIES, dtype=float) / N_STRATEGIES
        self.gains_sum = np.zeros(N_STRATEGIES, dtype=float)
        self.strategy_counts = np.zeros(N_STRATEGIES, dtype=int)
        self.history_p = [self.p.copy()]

    def select_strategy(self):
        """Sélectionne une stratégie par roulette proportionnelle aux probabilités p."""
        r = np.random.random()
        cumul = 0.0
        for k in range(N_STRATEGIES):
            cumul += self.p[k]
            if r <= cumul:
                return k
        return N_STRATEGIES - 1

    def record_payoff(self, strategy_k, old_fitness, new_fitness):
        """Enregistre le gain relatif apporté par la stratégie."""
        self.strategy_counts[strategy_k] += 1
        if new_fitness < old_fitness:
            denom = abs(old_fitness) if abs(old_fitness) > 1e-12 else 1.0
            relative_gain = (old_fitness - new_fitness) / denom
            clipped_gain = min(float(relative_gain), 10.0)
            self.gains_sum[strategy_k] += max(0.0, clipped_gain)

    def update_population(self):
        """Applique l'équation discrète du réplicateur de Taylor & Jonker (1978)."""
        f_k = np.zeros(N_STRATEGIES, dtype=float)
        for k in range(N_STRATEGIES):
            if self.strategy_counts[k] > 0:
                f_k[k] = self.gains_sum[k] / self.strategy_counts[k]
            else:
                f_k[k] = 0.0

        f_bar = np.sum(self.p * f_k)
        delta_p = self.learning_rate * self.p * (f_k - f_bar)
        self.p += delta_p
        self.p = np.maximum(self.p, self.mutation_threshold)
        self.p /= np.sum(self.p)
        self.history_p.append(self.p.copy())

        self.gains_sum.fill(0.0)
        self.strategy_counts.fill(0)
