"""
=============================================================================
Module : verifier.py
Description : Vérification mathématique rigoureuse des conditions d'équilibre 
              de Nash en stratégies mixtes selon les définitions et théorèmes 
              du cours (Chapitre 3 THJ).
Auteur : Master 2 Informatique Visuelle
=============================================================================
"""

import numpy as np
from typing import Dict, Any, Tuple


def verify_nash_equilibrium(
    A: np.ndarray, 
    B: np.ndarray, 
    x: np.ndarray, 
    y: np.ndarray, 
    tol: float = 1e-5
) -> Dict[str, Any]:
    """
    Vérifie si le profil de stratégies mixtes (x, y) est un équilibre de Nash 
    pour le jeu bimatriciel défini par (A, B).
    
    Critères mathématiques du cours (Slides 75-78 et 113) :
    1. x et y sont des distributions de probabilités valides :
       x >= 0, sum(x) == 1 et y >= 0, sum(y) == 1.
    2. Condition d'indifférence au support :
       Toute stratégie pure dans le support procure exactement l'utilité maximale :
       - Pour tout i dans supp(x) : (Ay)_i = max_k (Ay)_k = u_1^*
       - Pour tout j dans supp(y) : (B^T x)_j = max_l (B^T x)_l = u_2^*
    3. Condition de non-déviation hors support :
       Toute stratégie pure hors support procure une utilité inférieure ou égale :
       - Pour tout i hors supp(x) : (Ay)_i <= u_1^*
       - Pour tout j hors supp(y) : (B^T x)_j <= u_2^*
       
    Paramètres :
        A : Matrice des gains du joueur 1 (m x n)
        B : Matrice des gains du joueur 2 (m x n)
        x : Stratégie mixte du joueur 1 (taille m)
        y : Stratégie mixte du joueur 2 (taille n)
        tol : Tolérance numérique d'égalité (défaut = 1e-5)
        
    Retourne :
        Un dictionnaire contenant les diagnostics détaillés et le booléen 'is_nash'.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    m, n = A.shape
    errors = []
    
    # 1. Vérification des axiomes de probabilité
    if not (np.all(x >= -tol) and abs(np.sum(x) - 1.0) <= tol):
        errors.append(f"x n'est pas une distribution valide (sum={np.sum(x):.4f}, min={np.min(x):.4e})")
    if not (np.all(y >= -tol) and abs(np.sum(y) - 1.0) <= tol):
        errors.append(f"y n'est pas une distribution valide (sum={np.sum(y):.4f}, min={np.min(y):.4e})")
        
    # 2. Calcul des gains espérés et des vecteurs de réponse pure
    # Pour le joueur 1, jouer l'action pure i rapporte (A y)_i
    Ay = A @ y
    # Pour le joueur 2, jouer l'action pure j rapporte (x^T B)_j = (B^T x)_j
    BTx = B.T @ x
    
    # Utilités espérées globales u_1(x, y) = x^T A y, u_2(x, y) = x^T B y
    u1_expected = float(x @ Ay)
    u2_expected = float(BTx @ y)
    
    # Meilleures réponses pures maximales possibles
    u1_max_pure = float(np.max(Ay))
    u2_max_pure = float(np.max(BTx))
    
    # Supports (actions jouées avec probabilité strictement positive)
    supp_x = [i for i in range(m) if x[i] > tol]
    supp_y = [j for j in range(n) if y[j] > tol]
    
    # 3. Vérification de l'indifférence au support pour le Joueur 1
    dev_p1_support = 0.0
    for i in supp_x:
        diff = abs(Ay[i] - u1_max_pure)
        if diff > dev_p1_support:
            dev_p1_support = diff
        if diff > tol:
            errors.append(f"Joueur 1 : stratégie {i+1} dans le support donne {Ay[i]:.4f} != max {u1_max_pure:.4f}")
            
    # Vérification de non-déviation hors support pour le Joueur 1
    dev_p1_hors = 0.0
    for i in range(m):
        if i not in supp_x:
            excess = Ay[i] - u1_max_pure
            if excess > dev_p1_hors:
                dev_p1_hors = excess
            if excess > tol:
                errors.append(f"Joueur 1 : stratégie {i+1} hors support rapporte {Ay[i]:.4f} > max {u1_max_pure:.4f}")

    # 4. Vérification de l'indifférence au support pour le Joueur 2
    dev_p2_support = 0.0
    for j in supp_y:
        diff = abs(BTx[j] - u2_max_pure)
        if diff > dev_p2_support:
            dev_p2_support = diff
        if diff > tol:
            errors.append(f"Joueur 2 : stratégie {j+1} dans le support donne {BTx[j]:.4f} != max {u2_max_pure:.4f}")
            
    # Vérification de non-déviation hors support pour le Joueur 2
    dev_p2_hors = 0.0
    for j in range(n):
        if j not in supp_y:
            excess = BTx[j] - u2_max_pure
            if excess > dev_p2_hors:
                dev_p2_hors = excess
            if excess > tol:
                errors.append(f"Joueur 2 : stratégie {j+1} hors support rapporte {BTx[j]:.4f} > max {u2_max_pure:.4f}")

    is_nash = len(errors) == 0
    
    return {
        "is_nash": is_nash,
        "u1_expected": u1_expected,
        "u2_expected": u2_expected,
        "u1_max_pure": u1_max_pure,
        "u2_max_pure": u2_max_pure,
        "Ay": Ay,
        "BTx": BTx,
        "supp_x": supp_x,
        "supp_y": supp_y,
        "dev_p1_support": dev_p1_support,
        "dev_p2_support": dev_p2_support,
        "dev_p1_hors": dev_p1_hors,
        "dev_p2_hors": dev_p2_hors,
        "errors": errors
    }
