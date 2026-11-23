"""
=============================================================================
DM Théorie des Jeux : Implémentation de l'algorithme de Lemke-Howson
Master 2 Informatique Visuelle
=============================================================================
Ce script implémente l'algorithme de Lemke-Howson sans bibliothèque externe.
Il calcule un équilibre de Nash en stratégies mixtes (x, y) pour un jeu
bimatriciel (A, B) et vérifie mathématiquement le résultat.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Algorithme de Lemke-Howson
# ---------------------------------------------------------------------------

def lemke_howson(A, B, k0=1, verbose=False):
    """
    Calcule un équilibre de Nash (x, y) par l'algorithme de Lemke-Howson.
    
    Notations du cours :
      - A (m x n) : gains du Joueur 1
      - B (m x n) : gains du Joueur 2
      - x (taille m) : stratégie mixte du Joueur 1 (somme = 1, x >= 0)
      - y (taille n) : stratégie mixte du Joueur 2 (somme = 1, y >= 0)
      - k0 : étiquette initiale relâchée (1 <= k0 <= m + n)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape
    total_labels = m + n

    # Étape 1 : Rendre tous les gains strictement positifs (> 0)
    # L'équilibre de Nash est invariant par translation des gains.
    c_A = max(0.0, float(1.0 - np.min(A)))
    c_B = max(0.0, float(1.0 - np.min(B)))
    A_pos = A + c_A
    B_pos = B + c_B

    # Étape 2 : Créer les deux tableaux de Simplexe
    # Tableau 1 (Joueur 1, contrainte B^T z + s = 1) : n lignes, m colonnes z + n colonnes s + RHS
    T1 = np.zeros((n, m + n + 1), dtype=float)
    T1[:, :m] = B_pos.T
    T1[:, m:m+n] = np.eye(n)
    T1[:, -1] = 1.0
    base1 = [m + j for j in range(n)]  # Variables de base initiales : s_1..s_n

    # Tableau 2 (Joueur 2, contrainte A w + r = 1) : m lignes, n colonnes w + m colonnes r + RHS
    T2 = np.zeros((m, n + m + 1), dtype=float)
    T2[:, :n] = A_pos
    T2[:, n:n+m] = np.eye(m)
    T2[:, -1] = 1.0
    base2 = [n + i for i in range(m)]  # Variables de base initiales : r_1..r_m

    # Correspondance des étiquettes (Labels 1 à m+n)
    # - Labels 1..m : stratégies du Joueur 1 (z dans T1, r dans T2)
    # - Labels m+1..m+n : stratégies du Joueur 2 (s dans T1, w dans T2)
    def label_vers_col_T1(lbl):
        return lbl - 1

    def col_vers_label_T1(col):
        return col + 1

    def label_vers_col_T2(lbl):
        return (lbl - m - 1) if lbl > m else (n + lbl - 1)

    def col_vers_label_T2(col):
        return (m + col + 1) if col < n else (col - n + 1)

    # Fonction de pivotage élémentaire (Gauss-Jordan + ratio minimum lexicographique)
    def pivoter(T, base, col_entrante):
        nrows = T.shape[0]
        candidats = [i for i in range(nrows) if T[i, col_entrante] > 1e-12]
        if not candidats:
            raise ValueError("Erreur : pivot non borné.")

        # Test du ratio minimum (avec bris d'égalité lexicographique pour éviter le cyclage)
        meilleure_ligne = candidats[0]
        meilleur_ratio = np.hstack(([T[meilleure_ligne, -1]], T[meilleure_ligne, :-1])) / T[meilleure_ligne, col_entrante]

        for i in candidats[1:]:
            ratio = np.hstack(([T[i, -1]], T[i, :-1])) / T[i, col_entrante]
            diff = ratio - meilleur_ratio
            idx = np.where(np.abs(diff) > 1e-12)[0]
            if len(idx) > 0 and diff[idx[0]] < 0:
                meilleure_ligne = i
                meilleur_ratio = ratio

        ligne_pivot = meilleure_ligne
        var_sortante = base[ligne_pivot]

        # Élimination de Gauss-Jordan
        valeur_pivot = T[ligne_pivot, col_entrante]
        T[ligne_pivot, :] /= valeur_pivot
        for i in range(nrows):
            if i != ligne_pivot:
                T[i, :] -= T[i, col_entrante] * T[ligne_pivot, :]

        base[ligne_pivot] = col_entrante
        return var_sortante

    # Étape 3 : Boucle de cheminement complémentaire
    curr_label = k0
    curr_tab = 1 if k0 <= m else 2
    nb_pivots = 0

    while True:
        nb_pivots += 1
        if curr_tab == 1:
            col_in = label_vers_col_T1(curr_label)
            col_out = pivoter(T1, base1, col_in)
            lbl_out = col_vers_label_T1(col_out)
            next_tab = 2
        else:
            col_in = label_vers_col_T2(curr_label)
            col_out = pivoter(T2, base2, col_in)
            lbl_out = col_vers_label_T2(col_out)
            next_tab = 1

        if verbose:
            print(f"  Pivot {nb_pivots:2d} | Tableau {curr_tab} : entre label {curr_label}, sort label {lbl_out}")

        # Condition d'arrêt : on a récupéré le label de départ k0
        if lbl_out == k0:
            break

        curr_label = lbl_out
        curr_tab = next_tab

    # Étape 4 : Extraction et normalisation des probabilités
    # z = x / v, w = y / u
    z = np.zeros(m)
    for row, col in enumerate(base1):
        if col < m:
            z[col] = max(0.0, T1[row, -1])

    w = np.zeros(n)
    for row, col in enumerate(base2):
        if col < n:
            w[col] = max(0.0, T2[row, -1])

    x = z / np.sum(z)
    y = w / np.sum(w)

    return x, y, nb_pivots


# ---------------------------------------------------------------------------
# 2. Vérification Mathématique de l'Équilibre de Nash
# ---------------------------------------------------------------------------

def verifier_nash(A, B, x, y, tol=1e-5):
    """
    Vérifie les conditions d'équilibre de Nash selon le cours (Slides 75-78) :
      1. x et y sont des distributions valides (somme = 1, >= 0).
      2. Indifférence sur le support : toutes les actions pures jouées avec p > 0
         donnent le même gain maximal.
      3. Non-déviation hors support : les actions pures non jouées (p = 0)
         ne rapportent pas plus que le gain maximal.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape

    # Gains des actions pures
    gains_purs_P1 = A @ y      # (Ay)_i
    gains_purs_P2 = B.T @ x    # (B^T x)_j

    # Gains espérés globaux u_1(x, y) et u_2(x, y)
    u1 = float(x @ gains_purs_P1)
    u2 = float(gains_purs_P2 @ y)

    max_gain_P1 = float(np.max(gains_purs_P1))
    max_gain_P2 = float(np.max(gains_purs_P2))

    # Supports
    supp_P1 = [i for i in range(m) if x[i] > tol]
    supp_P2 = [j for j in range(n) if y[j] > tol]

    # Vérification Joueur 1
    valide_P1 = all(abs(gains_purs_P1[i] - max_gain_P1) <= tol for i in supp_P1) and \
                all(gains_purs_P1[i] <= max_gain_P1 + tol for i in range(m))

    # Vérification Joueur 2
    valide_P2 = all(abs(gains_purs_P2[j] - max_gain_P2) <= tol for j in supp_P2) and \
                all(gains_purs_P2[j] <= max_gain_P2 + tol for j in range(n))

    est_nash = valide_P1 and valide_P2

    return {
        "est_nash": est_nash,
        "u1": u1,
        "u2": u2,
        "gains_purs_P1": gains_purs_P1,
        "gains_purs_P2": gains_purs_P2,
        "supp_P1": supp_P1,
        "supp_P2": supp_P2
    }


# ---------------------------------------------------------------------------
# 3. Fonction d'affichage d'un test
# ---------------------------------------------------------------------------

def tester_jeu(titre, A, B, noms_P1=None, noms_P2=None, k0=1, verbose=False):
    """Affiche proprement les données d'un jeu et le résultat du test."""
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    m, n = A.shape

    if noms_P1 is None: noms_P1 = [f"L{i+1}" for i in range(m)]
    if noms_P2 is None: noms_P2 = [f"C{j+1}" for j in range(n)]

    print("\n" + "=" * 70)
    print(f"JEU : {titre} (Taille {m}x{n})")
    print("=" * 70)

    print("Matrice A (Gains Joueur 1) :")
    for i in range(m):
        row_str = "  ".join(f"{A[i, j]:6.1f}" for j in range(n))
        print(f"  {noms_P1[i]:>10} : [ {row_str} ]")

    print("Matrice B (Gains Joueur 2) :")
    for i in range(m):
        row_str = "  ".join(f"{B[i, j]:6.1f}" for j in range(n))
        print(f"  {noms_P1[i]:>10} : [ {row_str} ]")

    # Résolution
    x, y, nb_pivots = lemke_howson(A, B, k0=k0, verbose=verbose)
    res = verifier_nash(A, B, x, y)

    # Affichage du résultat
    print(f"\nRésultat Lemke-Howson (k0={k0}, convergé en {nb_pivots} pivots) :")
    str_x = ", ".join(f"{noms_P1[i]}: {x[i]:.4f}" for i in range(m))
    str_y = ", ".join(f"{noms_P2[j]}: {y[j]:.4f}" for j in range(n))
    print(f"  * Stratégie mixte x (Joueur 1) : [ {str_x} ]")
    print(f"  * Stratégie mixte y (Joueur 2) : [ {str_y} ]")
    print(f"  * Gain espéré Joueur 1 u_1(x, y) = {res['u1']:.4f}")
    print(f"  * Gain espéré Joueur 2 u_2(x, y) = {res['u2']:.4f}")

    print("Vérification mathématique :")
    g_p1_str = ", ".join(f"{noms_P1[i]}: {res['gains_purs_P1'][i]:.2f}" for i in range(m))
    g_p2_str = ", ".join(f"{noms_P2[j]}: {res['gains_purs_P2'][j]:.2f}" for j in range(n))
    print(f"  * Gains des actions pures P1 (Ay)  : [ {g_p1_str} ]")
    print(f"  * Gains des actions pures P2 (B^Tx) : [ {g_p2_str} ]")

    if res["est_nash"]:
        print("  => RÉSULTAT DU TEST : [ SUCCÈS ] (Équilibre de Nash vérifié)")
    else:
        print("  => RÉSULTAT DU TEST : [ ÉCHEC ]")

    return res["est_nash"]


# ---------------------------------------------------------------------------
# 4. Programme Principal : Exécution des tests demandés par le professeur
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  EXÉCUTION DES TESTS DU DEVOIR MAISON - LEMKE-HOWSON")
    print("#" * 70)

    tous_succes = True

    # -----------------------------------------------------------------------
    # TEST 1 : Jeu 2x2 avec gains négatifs (Matching Pennies - Slide 88)
    # -----------------------------------------------------------------------
    A_mp = [[1.0, -1.0], [-1.0, 1.0]]
    B_mp = [[-1.0, 1.0], [1.0, -1.0]]
    ok1 = tester_jeu("Matching Pennies (2x2, gains négatifs)", A_mp, B_mp, ["Pile", "Face"], ["Pile", "Face"])
    tous_succes = tous_succes and ok1

    # -----------------------------------------------------------------------
    # TEST 2 : Jeu 2x2 avec gains négatifs du cours (Jeu de l'Inspection - Slide 99)
    # -----------------------------------------------------------------------
    # Paramètres du cours : v=5, w=3, g=2, h=1
    A_insp = [[1.0, 1.0], [0.0, 3.0]]
    B_insp = [[1.0, 2.0], [-1.0, -3.0]]
    ok2 = tester_jeu("Jeu de l'Inspection (2x2, Slide 99)", A_insp, B_insp, 
                     ["Travailler", "Tricher"], ["Inspecter", "Non-Inspecter"])
    tous_succes = tous_succes and ok2

    # -----------------------------------------------------------------------
    # TEST 3 : Jeu 3x3 classique (Pierre - Feuille - Ciseaux - Slide 15)
    # -----------------------------------------------------------------------
    A_pfc = [[ 0.0, -1.0,  1.0],
             [ 1.0,  0.0, -1.0],
             [-1.0,  1.0,  0.0]]
    B_pfc = [[ 0.0,  1.0, -1.0],
             [-1.0,  0.0,  1.0],
             [ 1.0, -1.0,  0.0]]
    ok3 = tester_jeu("Pierre - Feuille - Ciseaux (3x3, Slide 15)", A_pfc, B_pfc,
                     ["Pierre", "Feuille", "Ciseaux"], ["Pierre", "Feuille", "Ciseaux"])
    tous_succes = tous_succes and ok3

    # -----------------------------------------------------------------------
    # TEST 4 : Jeu rectangulaire 2x3 du cours (Exemple 1 - Slide 111)
    # -----------------------------------------------------------------------
    A_111 = [[ 2.0, 1.0, 8.0],
             [-1.0, 2.0, 1.0]]
    B_111 = [[4.0, 0.0, 5.0],
             [0.0, 1.0, 4.0]]
    ok4 = tester_jeu("Exemple 1 du cours (Rectangulaire 2x3, Slide 111)", A_111, B_111,
                     ["L", "G"], ["a", "b", "c"])
    tous_succes = tous_succes and ok4

    # -----------------------------------------------------------------------
    # TEST 5 : Jeu rectangulaire 3x4 avec gains négatifs et positifs
    # -----------------------------------------------------------------------
    A_34 = [[ 3.0, -2.0,  5.0,  1.0],
            [-1.0,  4.0, -3.0,  2.0],
            [ 2.0,  0.0,  1.0, -4.0]]
    B_34 = [[ 1.0,  3.0, -1.0,  4.0],
            [ 2.0, -1.0,  5.0,  0.0],
            [-2.0,  4.0,  2.0,  1.0]]
    ok5 = tester_jeu("Jeu rectangulaire quelconque (3x4)", A_34, B_34)
    tous_succes = tous_succes and ok5

    # -----------------------------------------------------------------------
    # TEST 6 : Jeu 4x5 de taille supérieure (montée en charge)
    # -----------------------------------------------------------------------
    A_45 = [[ 4.0, -1.0,  2.0,  0.0,  3.0],
            [ 1.0,  5.0, -2.0,  3.0, -1.0],
            [-3.0,  2.0,  4.0, -1.0,  2.0],
            [ 2.0,  0.0,  1.0,  5.0, -2.0]]
    B_45 = [[ 2.0,  3.0, -2.0,  1.0,  4.0],
            [-1.0,  2.0,  4.0,  0.0, -2.0],
            [ 3.0, -1.0,  1.0,  5.0,  2.0],
            [ 0.0,  4.0, -3.0,  2.0,  1.0]]
    ok6 = tester_jeu("Jeu quelconque de dimension supérieure (4x5)", A_45, B_45)
    tous_succes = tous_succes and ok6

    print("\n" + "=" * 70)
    if tous_succes:
        print("BILAN : TOUS LES TESTS SONT VALIDÉS AVEC SUCCÈS (6/6) !")
    else:
        print("BILAN : CERTAINS TESTS ONT ÉCHOUÉ.")
    print("=" * 70 + "\n")
