"""
Tests unitaires — vérifie que PSO et SDPSO fonctionnent correctement.
Exécution : python -m pytest tests/ -v
        ou : python -m unittest tests/test_pso.py
"""
import unittest
import numpy as np

from src.pso import run_pso
from src.hybrid_pso import run_sdpso
from src.game_theory import GameDynamics, N_STRATEGIES
from experiments.benchmark import sphere, rastrigin, ackley


class TestBenchmarkFunctions(unittest.TestCase):
    """Les fonctions doivent retourner 0 en x = 0."""

    def test_sphere_zero(self):
        self.assertAlmostEqual(sphere(np.zeros(10)), 0.0)

    def test_rastrigin_zero(self):
        self.assertAlmostEqual(rastrigin(np.zeros(10)), 0.0)

    def test_ackley_zero(self):
        self.assertAlmostEqual(ackley(np.zeros(5)), 0.0, places=10)


class TestClassicPSO(unittest.TestCase):

    def test_returns_expected_keys(self):
        result = run_pso(sphere, dim=5, bounds=(-10, 10), popsize=10, max_fes=200, seed=42)
        self.assertIn("best_value", result)
        self.assertIn("best_position", result)
        self.assertIn("convergence", result)

    def test_sphere_converges(self):
        """Sur Sphere 5D avec 5000 FES, PSO doit descendre sous 1.0."""
        result = run_pso(sphere, dim=5, bounds=(-10, 10), popsize=20, max_fes=5000, seed=0)
        self.assertLess(result["best_value"], 1.0)

    def test_reproducible_with_seed(self):
        """Deux runs avec la même graine donnent exactement le même résultat."""
        r1 = run_pso(sphere, dim=5, bounds=(-10, 10), popsize=10, max_fes=300, seed=99)
        r2 = run_pso(sphere, dim=5, bounds=(-10, 10), popsize=10, max_fes=300, seed=99)
        self.assertAlmostEqual(r1["best_value"], r2["best_value"])

    def test_convergence_is_non_increasing(self):
        """La courbe de convergence doit être monotone décroissante."""
        result = run_pso(sphere, dim=3, bounds=(-5, 5), popsize=10, max_fes=200, seed=7)
        conv = result["convergence"]
        self.assertTrue(np.all(np.diff(conv) <= 1e-12))


class TestGameTheory(unittest.TestCase):

    def test_initial_probabilities(self):
        """Probabilités initiales uniformes et somme = 1."""
        g = GameDynamics()
        self.assertAlmostEqual(g.p.sum(), 1.0)
        self.assertTrue(np.allclose(g.p, 0.25))

    def test_replicator_sums_to_one(self):
        """Après mise à jour du réplicateur, les probas doivent toujours sommer à 1."""
        g = GameDynamics(alpha=0.1, learning_period=50)
        g.record_outcome(0, old_pbest=100.0, new_fitness=50.0)   # Stratégie 0 : succès
        g.record_outcome(1, old_pbest=100.0, new_fitness=120.0)  # Stratégie 1 : échec
        g._update_replicator(current_fes=50)
        self.assertAlmostEqual(g.p.sum(), 1.0, places=10)

    def test_mutation_threshold(self):
        """Aucune stratégie ne doit descendre sous 0.02 après le réplicateur."""
        g = GameDynamics(alpha=0.5)   # alpha élevé pour amplifier les écarts
        # Écraser les mémoires pour simuler une domination extrême
        g.gains[:] = [100, 0, 0, 0]
        g.successes[:] = [100, 1, 1, 1]
        g.failures[:] = [0, 100, 100, 100]
        g._update_replicator(current_fes=200)
        self.assertTrue(np.all(g.p >= 0.02))

    def test_restart_resets_probabilities(self):
        """Le restart doit remettre p à [0.25, 0.25, 0.25, 0.25]."""
        g = GameDynamics(restart_every=4)
        g.p = np.array([0.9, 0.05, 0.03, 0.02])  # État arbitraire
        g._check_restart(current_fes=500, max_fes=2000)  # 500 = 2000/4 → restart
        self.assertTrue(np.allclose(g.p, 0.25))


class TestSDPSO(unittest.TestCase):

    def test_returns_expected_keys(self):
        result = run_sdpso(sphere, dim=5, bounds=(-10, 10), popsize=10, max_fes=200, seed=42)
        for key in ["best_value", "best_position", "convergence",
                    "strategy_proba_history", "strategy_usage"]:
            self.assertIn(key, result)

    def test_uses_game_theory(self):
        """SDPSO doit réellement utiliser les 4 stratégies (aucune à 0 appels)."""
        result = run_sdpso(rastrigin, dim=5, bounds=(-5.12, 5.12),
                           popsize=20, max_fes=2000, seed=0)
        self.assertEqual(len(result["strategy_usage"]), N_STRATEGIES)
        # Avec 2000 FES et 4 stratégies, chacune doit être appelée au moins une fois
        self.assertTrue(np.all(result["strategy_usage"] >= 1))

    def test_strategy_probabilities_valid(self):
        """À chaque mise à jour, les probas doivent sommer à 1 et rester dans [0.02, 1]."""
        result = run_sdpso(sphere, dim=5, bounds=(-10, 10), popsize=10, max_fes=1000, seed=1)
        history = result["strategy_proba_history"]
        for row in history:
            self.assertAlmostEqual(row.sum(), 1.0, places=8)
            self.assertTrue(np.all(row >= 0.02))

    def test_reproducible_with_seed(self):
        r1 = run_sdpso(sphere, dim=5, bounds=(-10, 10), popsize=10, max_fes=300, seed=42)
        r2 = run_sdpso(sphere, dim=5, bounds=(-10, 10), popsize=10, max_fes=300, seed=42)
        self.assertAlmostEqual(r1["best_value"], r2["best_value"])

    def test_convergence_non_increasing(self):
        result = run_sdpso(sphere, dim=3, bounds=(-5, 5), popsize=10, max_fes=200, seed=5)
        conv = result["convergence"]
        self.assertTrue(np.all(np.diff(conv) <= 1e-12))


if __name__ == "__main__":
    unittest.main()
