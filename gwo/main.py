"""
main.py
Point d'entrée principal pour le projet GT-GWO
(Game Theory Grey Wolf Optimizer).
"""

import sys
from pathlib import Path

# Ajouter la racine au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gwo import gwo as classic_gwo
from src.gt_gwo import gt_gwo, objective_rastrigin
from src.knapsack_problem import get_standard_instance
from src.binary_gwo import binary_gwo
from src.binary_gt_gwo import binary_gt_gwo


def print_banner():
    print("=" * 70)
    print("  GT-GWO : GAME THEORY GREY WOLF OPTIMIZER")
    print("  Projet universitaire : Théorie des Jeux dans une Métaheuristique")
    print("=" * 70)


def menu():
    print_banner()
    print("Choisissez une option à exécuter :")
    print("  [1] Tester GWO classique sur Rastrigin (1 run)")
    print("  [2] Tester GT-GWO hybride sur Rastrigin (1 run avec dynamiques de jeu)")
    print("  [3] Tester Sac à Dos 0/1 avec solveur exact DP")
    print("  [4] Tester Binary GT-GWO sur Sac à Dos (1 run)")
    print("  [5] Lancer le PROTOCOLE COMPARATIF COMPLET (120 runs + Wilcoxon + graphiques)")
    print("  [0] Quitter")
    print("-" * 70)


def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        menu()
        try:
            choice = input("Entrez votre choix [0-5] : ").strip()
        except EOFError:
            choice = "5"

    if choice == "1":
        print("\n>>> Exécution de GWO classique sur Rastrigin (D=2, 100 itérations)...")
        pos, fit, _ = classic_gwo(verbose=True)
        print("\nRésultat :")
        print(f"Meilleure position : {pos}")
        print(f"Fitness finale     : {fit:.8e}")

    elif choice == "2":
        print("\n>>> Exécution de GT-GWO (hybride Théorie des Jeux) sur Rastrigin...")
        pos, fit, _, strat_hist = gt_gwo(verbose=True)
        print("\nRésultat :")
        print(f"Meilleure position : {pos}")
        print(f"Fitness finale     : {fit:.8e}")
        print(f"Probabilités finales des stratégies : {strat_hist[-1]}")

    elif choice == "3":
        print("\n>>> Résolution exacte par Programmation Dynamique du Sac à Dos...")
        problem = get_standard_instance()
        val, items = problem.solve_exact_dp()
        total_w = sum(items[i] * problem.weights[i] for i in range(problem.n_items))
        print(f"Capacité max : {problem.capacity}")
        print(f"Valeur optimale théorique : {val}")
        print(f"Poids total consommé      : {total_w} / {problem.capacity}")
        print(f"Objets retenus (binaire)  : {items}")

    elif choice == "4":
        print("\n>>> Exécution de Binary GT-GWO sur le Sac à Dos 0/1...")
        problem = get_standard_instance()
        sol, val, w, _, strat_hist = binary_gt_gwo(problem, verbose=True)
        print("\nRésultat :")
        print(f"Valeur trouvée : {val:.1f}")
        print(f"Poids consommé : {w:.1f} / {problem.capacity}")
        print(f"Probabilités finales : {strat_hist[-1]}")

    elif choice == "5":
        from experiments.experiments_comparison import run_benchmark
        run_benchmark()

    elif choice == "0":
        print("Fin du programme.")
    else:
        print(f"Choix invalide : {choice}")


if __name__ == "__main__":
    main()
