"""
Point d'entrée principal du projet.
Usage :
  python main.py              # Lancement standard (D=30, 10 runs)
  python main.py --quick      # Test rapide (D=10, 3 runs)
  python main.py --dim 20 --runs 15 --fes 500
"""
import argparse
from experiments.benchmark import run_comparison


def main():
    parser = argparse.ArgumentParser(
        description="PSO + Théorie des Jeux Évolutive (SDPSO)"
    )
    parser.add_argument("--dim",   type=int, default=30,  help="Dimension (défaut : 30)")
    parser.add_argument("--runs",  type=int, default=10,  help="Nombre de répétitions (défaut : 10)")
    parser.add_argument("--fes",   type=int, default=1000, help="Évaluations / dimension (défaut : 1000)")
    parser.add_argument("--pop",   type=int, default=40,  help="Taille du swarm (défaut : 40)")
    parser.add_argument("--quick", action="store_true",   help="Mode rapide : D=10, 3 runs, 300 FES/dim")
    args = parser.parse_args()

    if args.quick:
        dim, runs, fes = 10, 3, 300
        print("[Mode rapide]")
    else:
        dim, runs, fes = args.dim, args.runs, args.fes

    results = run_comparison(dim=dim, runs=runs, max_fes=fes * dim, popsize=args.pop)

    # Tableau récapitulatif final
    print("\n" + "─" * 65)
    print(f"{'Fonction':<16} {'PSO moyen':>14} {'SDPSO moyen':>14} {'Δ':>8}")
    print("─" * 65)
    for func_name, data in results.items():
        pso_m  = data["PSO"]["mean"]
        sdp_m  = data["SDPSO"]["mean"]
        delta  = (pso_m - sdp_m) / (pso_m + 1e-300) * 100
        sign   = "▲" if delta > 0 else "▼"
        print(f"{func_name:<16} {pso_m:>14.3e} {sdp_m:>14.3e} {sign}{abs(delta):>6.1f}%")
    print("─" * 65)


if __name__ == "__main__":
    main()
