"""
Fonctions benchmark et comparaison expérimentale.

Fonctions de test utilisées :
  - Sphere       : unimodale, séparable         → facile, valide la convergence de base
  - Schwefel2.22 : unimodale, non-séparable     → teste la coopération entre dimensions
  - Rosenbrock   : vallée en banane, non-lin.   → difficile même en faible dimension
  - Rastrigin    : multimodale très accidentée  → teste la diversité de l'essaim
  - Ackley       : multimodale, plateau ext.    → piège les algorithmes gloutons
  - Griewank     : multimodale, non-séparable   → corrélations entre dimensions
"""
import os
import json
import time
import numpy as np

from src.pso import run_pso
from src.hybrid_pso import run_sdpso
from experiments.visualize import plot_convergence, plot_strategy_dynamics, plot_strategy_bar


# ---------------------------------------------------------------------------
# Fonctions mathématiques benchmark
# ---------------------------------------------------------------------------

def sphere(x):
    return float(np.sum(x ** 2))


def schwefel_2_22(x):
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x):
    d = len(x)
    return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    d = len(x)
    t1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
    t2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
    return float(t1 + t2 + 20 + np.e)


def griewank(x):
    d = len(x)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1)))) + 1)


# Description de chaque fonction pour l'affichage et les graphiques
BENCHMARKS = [
    {"name": "Sphere",       "func": sphere,       "bounds": (-100, 100),  "type": "Unimodale"},
    {"name": "Schwefel2.22", "func": schwefel_2_22, "bounds": (-10, 10),   "type": "Unimodale"},
    {"name": "Rosenbrock",   "func": rosenbrock,   "bounds": (-30, 30),    "type": "Unimodale / Vallée"},
    {"name": "Rastrigin",    "func": rastrigin,    "bounds": (-5.12, 5.12),"type": "Multimodale"},
    {"name": "Ackley",       "func": ackley,       "bounds": (-32, 32),    "type": "Multimodale"},
    {"name": "Griewank",     "func": griewank,     "bounds": (-600, 600),  "type": "Multimodale"},
]


# ---------------------------------------------------------------------------
# Boucle de comparaison
# ---------------------------------------------------------------------------

def run_comparison(dim=30, runs=10, max_fes=None, popsize=40, output_dir="results"):
    """
    Lance la comparaison Classic PSO vs SDPSO sur toutes les fonctions benchmark.

    Paramètres
    ----------
    dim        : dimension de recherche
    runs       : nombre de répétitions indépendantes (Monte Carlo)
    max_fes    : budget d'évaluations (défaut : 1000 * dim)
    popsize    : taille du swarm
    output_dir : dossier de sauvegarde des graphiques et statistiques
    """
    if max_fes is None:
        max_fes = 1000 * dim

    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*65}")
    print(f"  PSO classique vs SDPSO (Théorie des Jeux Évolutive)")
    print(f"  Dimension : {dim} | Swarm : {popsize} | MaxFES : {max_fes} | Runs : {runs}")
    print(f"{'='*65}")

    all_results = {}
    all_strategy_usage = {}

    for bench in BENCHMARKS:
        name  = bench["name"]
        func  = bench["func"]
        bnds  = bench["bounds"]
        btype = bench["type"]

        pso_vals, sdpso_vals = [], []
        pso_convs, sdpso_convs = [], []
        total_usage = np.zeros(4, dtype=int)
        last_proba_history = None

        t0 = time.time()
        for r in range(runs):
            seed = 1000 + r * 37

            res_pso   = run_pso(func, dim, bnds, popsize, max_fes, seed=seed)
            res_sdpso = run_sdpso(func, dim, bnds, popsize, max_fes, seed=seed)

            pso_vals.append(res_pso["best_value"])
            sdpso_vals.append(res_sdpso["best_value"])
            pso_convs.append(res_pso["convergence"])
            sdpso_convs.append(res_sdpso["convergence"])
            total_usage += res_sdpso["strategy_usage"]

            if r == 0:
                last_proba_history = {
                    "p":   res_sdpso["strategy_proba_history"],
                    "fes": res_sdpso["strategy_fes_history"],
                }

        elapsed = time.time() - t0

        pso_arr   = np.array(pso_vals)
        sdpso_arr = np.array(sdpso_vals)

        # Aligner les courbes de convergence à la même longueur
        min_len = min(min(len(c) for c in pso_convs), min(len(c) for c in sdpso_convs))
        pso_conv_arr   = np.array([c[:min_len] for c in pso_convs])
        sdpso_conv_arr = np.array([c[:min_len] for c in sdpso_convs])

        # Graphiques
        plot_convergence(name, pso_conv_arr, sdpso_conv_arr,
                         os.path.join(output_dir, f"conv_{name.lower()}.png"))
        if last_proba_history:
            plot_strategy_dynamics(name, last_proba_history["fes"], last_proba_history["p"],
                                   os.path.join(output_dir, f"dynamics_{name.lower()}.png"))

        # Statistiques
        stats = {
            "PSO":   {"mean": float(pso_arr.mean()),   "std": float(pso_arr.std()),
                      "best": float(pso_arr.min()),    "worst": float(pso_arr.max())},
            "SDPSO": {"mean": float(sdpso_arr.mean()), "std": float(sdpso_arr.std()),
                      "best": float(sdpso_arr.min()),  "worst": float(sdpso_arr.max())},
        }
        all_results[name] = stats
        all_strategy_usage[name] = total_usage

        # Affichage console
        print(f"\n[{name}] ({btype}) — {elapsed:.1f}s")
        print(f"  PSO   : moy={stats['PSO']['mean']:.3e}  std={stats['PSO']['std']:.3e}  best={stats['PSO']['best']:.3e}")
        print(f"  SDPSO : moy={stats['SDPSO']['mean']:.3e}  std={stats['SDPSO']['std']:.3e}  best={stats['SDPSO']['best']:.3e}")

    # Graphique global de l'utilisation des stratégies
    plot_strategy_bar(all_strategy_usage, os.path.join(output_dir, "strategy_usage.png"))

    # Sauvegarde JSON
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Résultats sauvegardés dans '{output_dir}/'")
    print(f"{'='*65}")
    return all_results
