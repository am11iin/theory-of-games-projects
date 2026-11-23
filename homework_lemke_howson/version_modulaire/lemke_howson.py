"""
=============================================================================
Module : lemke_howson.py
Description : Implémentation complète 'from scratch' de l'algorithme de 
              Lemke-Howson (1964) pour le calcul d'un équilibre de Nash 
              en stratégies mixtes dans les jeux finis à deux joueurs.
Auteur : Master 2 Informatique Visuelle
=============================================================================
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional


def shift_payoffs(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Assure la positivité stricte des matrices de gains A et B.
    
    Théorie :
    L'algorithme de Lemke-Howson nécessite que les gains maximaux espérés soient 
    strictement positifs (u > 0 et v > 0) pour que les variables d'échelle z = x/v 
    et w = y/u soient bien définies et que les polyèdres contiennent l'origine.
    L'addition d'une constante c > 0 à A et d > 0 à B préserve exactement 
    l'ensemble des équilibres de Nash (invariance par translation affine).
    
    Paramètres :
        A : Matrice des gains du joueur 1 (m x n)
        B : Matrice des gains du joueur 2 (m x n)
        
    Retourne :
        (A_pos, B_pos, c_A, c_B) : matrices strictement positives et décalages appliqués.
    """
    min_A = np.min(A)
    min_B = np.min(B)
    
    # On garantit que le minimum de chaque matrice est au moins égal à 1.0
    c_A = max(0.0, float(1.0 - min_A))
    c_B = max(0.0, float(1.0 - min_B))
    
    A_pos = A.astype(float) + c_A
    B_pos = B.astype(float) + c_B
    
    return A_pos, B_pos, c_A, c_B


def lexicographic_pivot_row(T: np.ndarray, enter_col: int, tol: float = 1e-12) -> int:
    """
    Détermine la ligne sortante selon le test du ratio minimum lexicographique.
    
    Théorie :
    Dans l'algorithme du Simplexe et dans Lemke-Howson, en cas d'égalité stricte 
    des ratios b_i / a_ik (dégénérescence), un choix arbitraire peut entraîner 
    du cyclage. Le critère lexicographique garantit mathématiquement l'unicité 
    du pivot et l'absence totale de cycle en traitant les égalités colonne par colonne.
    
    Paramètres :
        T : Tableau courant de taille (nb_lignes x nb_colonnes)
        enter_col : Indice de la colonne de la variable entrante
        tol : Tolérance numérique pour la positivité du pivot
        
    Retourne :
        pivot_row : Indice de la ligne sélectionnée pour le pivot.
    """
    nrows, ncols = T.shape
    candidates = [i for i in range(nrows) if T[i, enter_col] > tol]
    
    if not candidates:
        raise ValueError(f"Pivot non borné détecté sur la colonne entrante {enter_col}.")
    
    best_row = candidates[0]
    # Vecteur ratio lexicographique : [ratio_b, ratio_col_0, ratio_col_1, ...]
    best_vec = np.hstack(([T[best_row, -1]], T[best_row, :-1])) / T[best_row, enter_col]
    
    for row in candidates[1:]:
        vec = np.hstack(([T[row, -1]], T[row, :-1])) / T[row, enter_col]
        diff = vec - best_vec
        # Recherche du premier indice où la différence est significative
        nz = np.where(np.abs(diff) > tol)[0]
        if len(nz) > 0 and diff[nz[0]] < 0:
            best_row = row
            best_vec = vec
            
    return best_row


def gauss_jordan_pivot(T: np.ndarray, pivot_row: int, enter_col: int) -> None:
    """
    Effectue l'opération de pivotage élémentaire de Gauss-Jordan sur le tableau T.
    
    Paramètres :
        T : Tableau (modifié en place)
        pivot_row : Ligne du pivot
        enter_col : Colonne du pivot
    """
    pivot_val = T[pivot_row, enter_col]
    T[pivot_row, :] /= pivot_val
    
    nrows = T.shape[0]
    for i in range(nrows):
        if i != pivot_row:
            factor = T[i, enter_col]
            if abs(factor) > 1e-15:
                T[i, :] -= factor * T[pivot_row, :]


def lemke_howson(
    A_in: np.ndarray, 
    B_in: np.ndarray, 
    k0: int = 1, 
    max_steps: int = 10000, 
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Exécute l'algorithme de Lemke-Howson pour un label initial k0 donné.
    
    Modélisation :
        Joueur 1 : m stratégies (indices 0..m-1, étiquettes 1..m)
        Joueur 2 : n stratégies (indices 0..n-1, étiquettes m+1..m+n)
        
        Polytope 1 (Joueur 1) : B^T z + s = 1_n
            - Variables z_1..z_m (colonnes 0..m-1), étiquettes 1..m
            - Variables d'écart s_1..s_n (colonnes m..m+n-1), étiquettes m+1..m+n
            - Base initiale : s_1..s_n
            - Étiquettes initiales présentes (hors-base, valant 0) : {1, ..., m}
            
        Polytope 2 (Joueur 2) : A w + r = 1_m
            - Variables w_1..w_n (colonnes 0..n-1), étiquettes m+1..m+n
            - Variables d'écart r_1..r_m (colonnes n..n+m-1), étiquettes 1..m
            - Base initiale : r_1..r_m
            - Étiquettes initiales présentes (hors-base, valant 0) : {m+1, ..., m+n}
            
        Le couple d'origine (z=0, w=0) possède l'ensemble complet des étiquettes {1..m+n}.
        L'algorithme relâche l'étiquette k0, suit le chemin de complémentarité 
        en éliminant à chaque étape l'étiquette dupliquée dans l'autre tableau, 
        jusqu'à ce que l'étiquette k0 soit récupérée.
        
    Paramètres :
        A_in : Matrice des gains du joueur 1 (m x n)
        B_in : Matrice des gains du joueur 2 (m x n)
        k0 : Label initial à relâcher, 1 <= k0 <= m+n (défaut = 1)
        max_steps : Nombre maximal de pivots autorisés
        verbose : Affichage détaillé du déroulement des pivots
        
    Retourne :
        x : Stratégie mixte du Joueur 1 (vecteur de probabilités de taille m)
        y : Stratégie mixte du Joueur 2 (vecteur de probabilités de taille n)
        stats : Dictionnaire contenant l'historique et les statistiques d'exécution
    """
    A = np.asarray(A_in, dtype=float)
    B = np.asarray(B_in, dtype=float)
    
    if A.shape != B.shape:
        raise ValueError(f"Dimensions incompatibles : A {A.shape} != B {B.shape}")
        
    m, n = A.shape
    total_labels = m + n
    
    if not (1 <= k0 <= total_labels):
        raise ValueError(f"k0={k0} invalide. Doit être compris entre 1 et {total_labels}.")
        
    # 1. Translation pour gains strictement positifs
    A_pos, B_pos, c_A, c_B = shift_payoffs(A, B)
    
    # 2. Construction des Tableaux
    # Tableau 1 (dimension n x (m + n + 1)) : [ B^T | I_n | 1_n ]
    T1 = np.zeros((n, m + n + 1), dtype=float)
    T1[:, :m] = B_pos.T
    T1[:, m:m+n] = np.eye(n)
    T1[:, -1] = 1.0
    basis1 = [m + j for j in range(n)] # Variables s_1..s_n en base
    
    # Tableau 2 (dimension m x (n + m + 1)) : [ A | I_m | 1_m ]
    T2 = np.zeros((m, n + m + 1), dtype=float)
    T2[:, :n] = A_pos
    T2[:, n:n+m] = np.eye(m)
    T2[:, -1] = 1.0
    basis2 = [n + i for i in range(m)] # Variables r_1..r_m en base
    
    # Fonctions de conversion entre colonnes et étiquettes
    def col_to_label_T1(c: int) -> int:
        return c + 1  # 0..m-1 -> 1..m ; m..m+n-1 -> m+1..m+n
        
    def label_to_col_T1(lbl: int) -> int:
        return lbl - 1
        
    def col_to_label_T2(c: int) -> int:
        if c < n:
            return m + c + 1  # w_j -> m+1..m+n
        else:
            return c - n + 1  # r_i -> 1..m
            
    def label_to_col_T2(lbl: int) -> int:
        if lbl > m:
            return lbl - m - 1 # w_(lbl-m)
        else:
            return n + lbl - 1 # r_lbl
            
    # 3. Initialisation du chemin
    # Si k0 <= m, k0 est relâché dans T1 (z_k0 entre en base)
    # Si k0 > m, k0 est relâché dans T2 (w_(k0-m) entre en base)
    curr_label = k0
    curr_tab = 1 if k0 <= m else 2
    
    trace = []
    converged = False
    
    for step in range(1, max_steps + 1):
        if curr_tab == 1:
            enter_col = label_to_col_T1(curr_label)
            pivot_row = lexicographic_pivot_row(T1, enter_col)
            leaving_col = basis1[pivot_row]
            leaving_label = col_to_label_T1(leaving_col)
            
            gauss_jordan_pivot(T1, pivot_row, enter_col)
            basis1[pivot_row] = enter_col
            next_tab = 2
        else:
            enter_col = label_to_col_T2(curr_label)
            pivot_row = lexicographic_pivot_row(T2, enter_col)
            leaving_col = basis2[pivot_row]
            leaving_label = col_to_label_T2(leaving_col)
            
            gauss_jordan_pivot(T2, pivot_row, enter_col)
            basis2[pivot_row] = enter_col
            next_tab = 1
            
        trace.append({
            "step": step,
            "tableau": curr_tab,
            "entered_label": curr_label,
            "leaving_label": leaving_label
        })
        
        if verbose:
            print(f"[Pivot {step:2d}] Tableau {curr_tab} : variable étiquetée {curr_label} entre, "
                  f"variable étiquetée {leaving_label} sort.")
            
        if leaving_label == k0:
            converged = True
            if verbose:
                print(f"[Convergence] Étiquette initiale {k0} récupérée en {step} pivots.")
            break
            
        curr_label = leaving_label
        curr_tab = next_tab
        
    if not converged:
        raise RuntimeError(f"Lemke-Howson n'a pas convergé après {max_steps} pivots avec k0={k0}.")
        
    # 4. Extraction des vecteurs non normalisés z et w
    z = np.zeros(m, dtype=float)
    for row_idx, col_idx in enumerate(basis1):
        if col_idx < m:
            z[col_idx] = max(0.0, T1[row_idx, -1])
            
    w = np.zeros(n, dtype=float)
    for row_idx, col_idx in enumerate(basis2):
        if col_idx < n:
            w[col_idx] = max(0.0, T2[row_idx, -1])
            
    sum_z = np.sum(z)
    sum_w = np.sum(w)
    
    if sum_z <= 1e-15 or sum_w <= 1e-15:
        raise RuntimeError("Solution triviale nulle obtenue (anomalie de parcours).")
        
    # 5. Renormalisation des probabilités (x in Delta_m, y in Delta_n)
    x = z / sum_z
    y = w / sum_w
    
    # Nettoyage des petites erreurs numériques d'arrondi
    x[np.abs(x) < 1e-12] = 0.0
    x = x / np.sum(x)
    y[np.abs(y) < 1e-12] = 0.0
    y = y / np.sum(y)
    
    stats = {
        "k0": k0,
        "num_steps": len(trace),
        "trace": trace,
        "z_unnormalized": z,
        "w_unnormalized": w,
        "shift_A": c_A,
        "shift_B": c_B,
        "converged": converged
    }
    
    return x, y, stats


def lemke_howson_all_labels(A: np.ndarray, B: np.ndarray, tol: float = 1e-5) -> List[Dict[str, Any]]:
    """
    Exécute Lemke-Howson pour TOUTES les étiquettes de départ k0 in {1, ..., m+n}
    afin d'explorer les différents équilibres de Nash accessibles du jeu.
    
    Retourne la liste des équilibres distincts trouvés avec les labels correspondants.
    """
    m, n = A.shape
    total_labels = m + n
    unique_equilibria = []
    
    for k0 in range(1, total_labels + 1):
        try:
            x, y, stats = lemke_howson(A, B, k0=k0, verbose=False)
            
            # Vérifier si cet équilibre a déjà été trouvé
            is_new = True
            for eq in unique_equilibria:
                if np.allclose(eq["x"], x, atol=tol) and np.allclose(eq["y"], y, atol=tol):
                    eq["found_by_k0"].append(k0)
                    is_new = False
                    break
                    
            if is_new:
                unique_equilibria.append({
                    "x": x,
                    "y": y,
                    "stats": stats,
                    "found_by_k0": [k0]
                })
        except Exception as e:
            # En cas de jeu dégénéré sans équilibre joignable par cette arête
            pass
            
    return unique_equilibria
