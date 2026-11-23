"""
=============================================================================
Programme Principal : main.py
Description : Point d'entrée pour l'exécution, l'évaluation et la validation 
              du Devoir Maison de Théorie des Jeux (M2 Informatique Visuelle).
              Implémentation de l'algorithme de Lemke-Howson et vérification 
              des équilibres de Nash en stratégies mixtes.
Auteur : Master 2 Informatique Visuelle
=============================================================================
"""

import sys
import numpy as np
from lemke_howson import lemke_howson, lemke_howson_all_labels
from verifier import verify_nash_equilibrium
from test_suite import get_all_test_games


def format_vector(v: np.ndarray, strat_names: list = None) -> str:
    """Formate un vecteur de probabilités de façon lisible et concise."""
    parts = []
    for i, val in enumerate(v):
        name = strat_names[i] if strat_names and i < len(strat_names) else f"S{i+1}"
        parts.append(f"{name}: {val:6.4f}")
    return "[" + ", ".join(parts) + "]"


def format_matrix(M: np.ndarray, row_names: list = None, col_names: list = None) -> str:
    """Formate une matrice 2D avec alignement soigné."""
    lines = []
    m, n = M.shape
    col_headers = [col_names[j] if col_names and j < len(col_names) else f"C{j+1}" for j in range(n)]
    
    # En-tête
    header = "          " + "  ".join(f"{h:>8}" for h in col_headers)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    
    for i in range(m):
        r_name = row_names[i] if row_names and i < len(row_names) else f"L{i+1}"
        row_str = f"{r_name:>8} |" + "  ".join(f"{M[i, j]:8.2f}" for j in range(n))
        lines.append(row_str)
    return "\n".join(lines)


def run_benchmark(verbose_pivots: bool = False) -> None:
    """
    Exécute tous les jeux de test de la collection et affiche un rapport détaillé.
    """
    games = get_all_test_games()
    
    print("\n" + "=" * 80)
    print(" " * 15 + "DEVOIR MAISON DE THÉORIE DES JEUX (M2 IV)")
    print(" " * 12 + "VALIDATION DE L'ALGORITHME DE LEMKE-HOWSON")
    print("=" * 80)
    print(f"Total de jeux à tester : {len(games)}")
    print("Contraintes : Sans bibliothèque externe, vérification mathématique stricte.")
    print("=" * 80 + "\n")
    
    nb_passed = 0
    nb_failed = 0
    
    for idx, game in enumerate(games, start=1):
        name = game["name"]
        cat = game["category"]
        desc = game["description"]
        A = game["A"]
        B = game["B"]
        s_p1 = game["strategies_P1"]
        s_p2 = game["strategies_P2"]
        m, n = A.shape
        
        print(f"\n>> TEST {idx:02d} [{cat}] : {name}")
        print(f"   Description : {desc}")
        print("   --- Matrice de gains A (Joueur 1) ---")
        print(format_matrix(A, s_p1, s_p2))
        print("   --- Matrice de gains B (Joueur 2) ---")
        print(format_matrix(B, s_p1, s_p2))
        
        try:
            # 1. Résolution par Lemke-Howson avec label par défaut k0=1
            x, y, stats = lemke_howson(A, B, k0=1, verbose=verbose_pivots)
            
            # 2. Vérification mathématique stricte de l'équilibre
            verif = verify_nash_equilibrium(A, B, x, y)
            
            # 3. Affichage des résultats
            print(f"\n   Résolution Lemke-Howson (k0 = {stats['k0']}, {stats['num_steps']} pivots) :")
            print(f"   * Stratégie mixte x (Joueur 1) : {format_vector(x, s_p1)}")
            print(f"   * Stratégie mixte y (Joueur 2) : {format_vector(y, s_p2)}")
            print(f"   * Gain espéré Joueur 1 u_1(x, y) = {verif['u1_expected']:8.4f}")
            print(f"   * Gain espéré Joueur 2 u_2(x, y) = {verif['u2_expected']:8.4f}")
            
            # 4. Détails des conditions de meilleure réponse
            print("   --- Vérification des conditions d'équilibre de Nash ---")
            print(f"   * Gains des actions pures P1 (Ay)  : " + ", ".join(f"{s_p1[i]}: {verif['Ay'][i]:.4f}" for i in range(m)))
            print(f"   * Max gain pur P1 = {verif['u1_max_pure']:.4f} | Écart max support = {verif['dev_p1_support']:.2e} | Déviation hors support = {verif['dev_p1_hors']:.2e}")
            print(f"   * Gains des actions pures P2 (B^Tx) : " + ", ".join(f"{s_p2[j]}: {verif['BTx'][j]:.4f}" for j in range(n)))
            print(f"   * Max gain pur P2 = {verif['u2_max_pure']:.4f} | Écart max support = {verif['dev_p2_support']:.2e} | Déviation hors support = {verif['dev_p2_hors']:.2e}")
            
            # Exploration des équilibres alternatifs (via les autres k0)
            all_eq = lemke_howson_all_labels(A, B)
            if len(all_eq) > 1:
                print(f"   * Note : {len(all_eq)} équilibres distincts découverts en variant k0 dans {{1..{m+n}}}.")
                for e_idx, eq in enumerate(all_eq, start=1):
                    print(f"     -> Équilibre #{e_idx} (k0={eq['found_by_k0']}) : x={np.round(eq['x'], 3)}, y={np.round(eq['y'], 3)}")
            
            if verif["is_nash"]:
                print(f"   => STATUT DU TEST : [ RÉUSSI ] (Équilibre de Nash mathématiquement validé)")
                nb_passed += 1
            else:
                print(f"   => STATUT DU TEST : [ ÉCHOUÉ ] Erreurs : {verif['errors']}")
                nb_failed += 1
                
        except Exception as err:
            print(f"   => STATUT DU TEST : [ ÉCHOUÉ AVEC EXCEPTION ] : {err}")
            nb_failed += 1
            
        print("-" * 80)
        
    print("\n" + "=" * 80)
    print(f"BILAN DES TESTS : {nb_passed}/{len(games)} RÉUSSIS, {nb_failed} ÉCHOUÉS.")
    if nb_failed == 0:
        print("TOUS LES TESTS ONT ÉTÉ VALIDÉS AVEC SUCCÈS !")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    run_benchmark(verbose_pivots=verbose)
