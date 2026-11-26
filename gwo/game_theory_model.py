"""
game_theory_model.py
Modèle de Théorie des Jeux Évolutive (EGT - Evolutionary Game Theory)
basé sur la dynamique du réplicateur de Taylor & Jonker (1978).

Ce module gère l'arbitrage adaptatif entre 4 stratégies de recherche
pour les loups dans l'algorithme GT-GWO :
1. TRIO_COOP      : Chasse coopérative standard GWO (guidée par Alpha, Beta, Delta).
2. ALPHA_EXPLOIT  : Exploitation intensive focalisée sur Alpha (attaque directe).
3. LEVY_EXPLORE   : Exploration globale stochastique par saut de Lévy (anti-blocage).
4. SOCIAL_PEER    : Coopération horizontale par apprentissage entre pairs (diversité).
"""

import numpy as np


STRATEGY_NAMES = [
    "TRIO_COOP",      # 0 : Chasse classique du trio (Alpha + Beta + Delta) / 3
    "ALPHA_EXPLOIT",  # 1 : Exploitation ciblée sur Alpha
    "LEVY_EXPLORE",   # 2 : Exploration / Échappement par vol de Lévy
    "SOCIAL_PEER"     # 3 : Interaction sociale horizontale entre pairs (X_r1 - X_r2)
]

N_STRATEGIES = len(STRATEGY_NAMES)
MUTATION_THRESHOLD = 0.05  # Seuil minimal anti-extinction pour maintenir la diversité


def levy_flight(dimensions, beta=1.5):
    """
    Génère un pas de vol de Lévy selon l'algorithme de Mantegna (1994).
    Permet des sauts occasionnels à longue distance pour s'échapper des minima locaux.
    """
    from math import gamma, sin, pi

    # Calcul de sigma_u
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
        """
        Paramètres :
            learning_rate (float) : vitesse d'apprentissage (alpha dans Taylor & Jonker)
            mutation_threshold (float) : probabilité minimale garantie pour chaque stratégie
        """
        self.learning_rate = learning_rate
        self.mutation_threshold = mutation_threshold

        # Vecteur de probabilités initialisé uniformément à 1 / K
        self.p = np.ones(N_STRATEGIES, dtype=float) / N_STRATEGIES

        # Mémoires des gains de l'itération en cours
        self.gains_sum = np.zeros(N_STRATEGIES, dtype=float)
        self.strategy_counts = np.zeros(N_STRATEGIES, dtype=int)

        # Historique pour le traçage de l'évolution des stratégies
        self.history_p = [self.p.copy()]

    def select_strategy(self):
        """
        Sélectionne une stratégie par roulette proportionnelle aux probabilités p.
        """
        r = np.random.random()
        cumul = 0.0
        for k in range(N_STRATEGIES):
            cumul += self.p[k]
            if r <= cumul:
                return k
        return N_STRATEGIES - 1

    def record_payoff(self, strategy_k, old_fitness, new_fitness):
        """
        Calcule et enregistre le payoff (gain relatif) apporté par la stratégie.
        En minimisation : une amélioration signifie new_fitness < old_fitness.
        """
        self.strategy_counts[strategy_k] += 1

        if new_fitness < old_fitness:
            # Amélioration stricte : calcul du gain relatif normalisé
            denom = abs(old_fitness) if abs(old_fitness) > 1e-12 else 1.0
            relative_gain = (old_fitness - new_fitness) / denom
            # Borner le gain pour éviter les instabilités numériques
            clipped_gain = min(float(relative_gain), 10.0)
            self.gains_sum[strategy_k] += max(0.0, clipped_gain)

    def update_population(self):
        """
        Applique l'équation discrète du réplicateur de Taylor & Jonker (1978) :
            p_k(t+1) = p_k(t) + alpha * p_k(t) * (f_k - f_bar)
        suivie du seuil anti-extinction et de la renormalisation.
        """
        # Calcul du gain moyen f_k obtenu par chaque stratégie
        f_k = np.zeros(N_STRATEGIES, dtype=float)
        for k in range(N_STRATEGIES):
            if self.strategy_counts[k] > 0:
                f_k[k] = self.gains_sum[k] / self.strategy_counts[k]
            else:
                f_k[k] = 0.0

        # Gain moyen global de la meute : f_bar = sum(p_k * f_k)
        f_bar = np.sum(self.p * f_k)

        # Équation du réplicateur
        delta_p = self.learning_rate * self.p * (f_k - f_bar)
        self.p += delta_p

        # Seuil anti-extinction (mutation) pour éviter qu'une stratégie ne disparaisse
        self.p = np.maximum(self.p, self.mutation_threshold)

        # Renormalisation pour que la somme fasse strictement 1
        self.p /= np.sum(self.p)

        # Mémorisation dans l'historique
        self.history_p.append(self.p.copy())

        # Réinitialisation des compteurs pour la prochaine itération
        self.gains_sum.fill(0.0)
        self.strategy_counts.fill(0)
