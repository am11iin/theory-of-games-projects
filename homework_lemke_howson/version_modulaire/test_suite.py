"""
=============================================================================
Module : test_suite.py
Description : Ensemble complet de jeux de test pour valider l'algorithme de 
              Lemke-Howson et la fonction de vérification d'équilibre de Nash.
              Inclut des cas classiques, des exemples issus du cours du 
              professeur, des jeux à gains négatifs et des jeux rectangulaires.
Auteur : Master 2 Informatique Visuelle
=============================================================================
"""

import numpy as np
from typing import List, Dict, Any


def get_all_test_games() -> List[Dict[str, Any]]:
    """
    Construit et retourne la collection des jeux de test ordonnés par dimension.
    """
    games = []
    
    # -------------------------------------------------------------------------
    # 1. JEUX 2x2
    # -------------------------------------------------------------------------
    
    # Jeu 1 : Matching Pennies (Apparier les sous) - Slide 88
    # Jeu à somme nulle, gains positifs et négatifs. Équilibre purement mixte.
    A_mp = np.array([
        [ 1.0, -1.0],
        [-1.0,  1.0]
    ])
    B_mp = -A_mp
    games.append({
        "name": "Matching Pennies (Slide 88)",
        "category": "2x2",
        "description": "Jeu à somme nulle, gains [-1, 1], équilibre purement mixte (1/2, 1/2).",
        "strategies_P1": ["Pile", "Face"],
        "strategies_P2": ["Pile", "Face"],
        "A": A_mp,
        "B": B_mp,
        "expected_support": (2, 2)
    })
    
    # Jeu 2 : La Bataille du couple (Battle of the Sexes) - Slides 17 & 82
    # Jeu de coordination à gains positifs.
    A_bataille = np.array([
        [3.0, 1.0],
        [0.0, 2.0]
    ])
    B_bataille = np.array([
        [2.0, 1.0],
        [0.0, 3.0]
    ])
    games.append({
        "name": "Bataille du couple (Slide 82)",
        "category": "2x2",
        "description": "Jeu de coordination à deux équilibres purs et un mixte.",
        "strategies_P1": ["Théâtre", "Foot"],
        "strategies_P2": ["Théâtre", "Foot"],
        "A": A_bataille,
        "B": B_bataille,
        "expected_support": None
    })
    
    # Jeu 3 : Dilemme du Prisonnier (Prisoner's Dilemma)
    # Gains strictement négatifs (années de prison). Équilibre pur en stratégies dominées.
    A_dp = np.array([
        [-1.0, -10.0],
        [ 0.0,  -8.0]
    ])
    B_dp = np.array([
        [-1.0,   0.0],
        [-10.0, -8.0]
    ])
    games.append({
        "name": "Dilemme du Prisonnier (Gains négatifs)",
        "category": "2x2",
        "description": "Gains négatifs, équilibre unique en stratégies strictement dominées (Avouer, Avouer).",
        "strategies_P1": ["Taire", "Avouer"],
        "strategies_P2": ["Taire", "Avouer"],
        "A": A_dp,
        "B": B_dp,
        "expected_support": (1, 1)
    })
    
    # Jeu 4 : Jeu de l'Inspection (Inspection Game) - Slides 99 à 102
    # Paramètres : v=5, w=3, g=2, h=1 (v > w > g > h > 0)
    # Gains négatifs dans B. Équilibre mixte théorique : x=(2/3, 1/3), y=(2/3, 1/3)
    A_insp = np.array([
        [1.0, 1.0], # T : w-g=1, w-g=1
        [0.0, 3.0]  # NT : 0, w=3
    ])
    B_insp = np.array([
        [ 1.0,  2.0], # T : v-w-h=1, v-w=2
        [-1.0, -3.0]  # NT : -h=-1, -w=-3
    ])
    games.append({
        "name": "Jeu de l'Inspection (Slides 99-102)",
        "category": "2x2",
        "description": "Jeu asymétrique avec gains négatifs dans B, équilibre mixte x=(2/3, 1/3), y=(2/3, 1/3).",
        "strategies_P1": ["Travailler (T)", "Tricher (NT)"],
        "strategies_P2": ["Inspecter (I)", "Ne pas inspecter (NI)"],
        "A": A_insp,
        "B": B_insp,
        "expected_support": (2, 2)
    })

    # Jeu 5 : Jeu du Carrefour - Slide 14
    # Deux voitures au carrefour : Passer (P) ou Stopper (S)
    A_carr = np.array([
        [-1.0, 2.0],
        [ 1.0, 0.0]
    ])
    B_carr = np.array([
        [-1.0, 1.0],
        [ 2.0, 0.0]
    ])
    games.append({
        "name": "Jeu du Carrefour (Slide 14)",
        "category": "2x2",
        "description": "Deux équilibres purs (P, S) et (S, P) et un équilibre mixte.",
        "strategies_P1": ["Passer (P)", "Stopper (S)"],
        "strategies_P2": ["Passer (P)", "Stopper (S)"],
        "A": A_carr,
        "B": B_carr,
        "expected_support": None
    })

    # -------------------------------------------------------------------------
    # 2. JEUX 3x3
    # -------------------------------------------------------------------------

    # Jeu 6 : Pierre - Feuille - Ciseaux (Rock-Paper-Scissors) - Slide 15
    # Somme nulle, équilibre mixte unique (1/3, 1/3, 1/3)
    A_pfc = np.array([
        [ 0.0, -1.0,  1.0],
        [ 1.0,  0.0, -1.0],
        [-1.0,  1.0,  0.0]
    ])
    B_pfc = -A_pfc
    games.append({
        "name": "Pierre - Feuille - Ciseaux (Slide 15)",
        "category": "3x3",
        "description": "Jeu classique à somme nulle, sans équilibre pur, équilibre mixte uniforme (1/3, 1/3, 1/3).",
        "strategies_P1": ["Pierre", "Feuille", "Ciseaux"],
        "strategies_P2": ["Pierre", "Feuille", "Ciseaux"],
        "A": A_pfc,
        "B": B_pfc,
        "expected_support": (3, 3)
    })

    # Jeu 7 : Jeu 3x3 Général avec équilibre mixte partiel
    # Exemple 3.3 du cours Slide 11
    A_ex33 = np.array([
        [6.0, 9.0, 0.0],
        [5.0, 8.0, 4.0],
        [7.0, 7.0, 0.0]
    ])
    B_ex33 = np.array([
        [7.0, 6.0, 6.0],
        [2.0, 2.0, 5.0],
        [7.0, 8.0, 9.0]
    ])
    games.append({
        "name": "Exemple 3.3 du cours (Slide 11)",
        "category": "3x3",
        "description": "Matrice 3x3 générale du cours, exploration des équilibres.",
        "strategies_P1": ["A1", "A2", "A3"],
        "strategies_P2": ["B1", "B2", "B3"],
        "A": A_ex33,
        "B": B_ex33,
        "expected_support": None
    })

    # -------------------------------------------------------------------------
    # 3. JEUX RECTANGULAIRES (m != n)
    # -------------------------------------------------------------------------

    # Jeu 8 : Exemple 1 du cours Slide 111 (Dimension 2x3)
    # A = [[2, 1, 8], [-1, 2, 1]], B = [[4, 0, 5], [0, 1, 4]]
    A_s111 = np.array([
        [ 2.0, 1.0, 8.0],
        [-1.0, 2.0, 1.0]
    ])
    B_s111 = np.array([
        [4.0, 0.0, 5.0],
        [0.0, 1.0, 4.0]
    ])
    games.append({
        "name": "Exemple 1 (Slide 111)",
        "category": "2x3",
        "description": "Jeu bimatriciel rectangulaire 2x3 issu directement de l'introduction computationnelle du cours.",
        "strategies_P1": ["L", "G"],
        "strategies_P2": ["a", "b", "c"],
        "A": A_s111,
        "B": B_s111,
        "expected_support": None
    })

    # Jeu 9 : Variante Bataille du couple Slide 105 (Dimension 2x3)
    # Équilibre mixte théorique calculé en Slide 109 : x=(4/5, 1/5), y=(1/4, 0, 3/4)
    A_s105 = np.array([
        [3.0, 1.0, 0.0],
        [0.0, 2.0, 1.0]
    ])
    B_s105 = np.array([
        [2.0, 1.0, 1.5],
        [0.0, 3.0, 2.0]
    ])
    games.append({
        "name": "Variante Bataille du couple (Slides 105-109)",
        "category": "2x3",
        "description": "Jeu 2x3 avec équilibre mixte x=(4/5, 1/5), y=(1/4, 0, 3/4) résolu pas à pas dans le cours.",
        "strategies_P1": ["a", "b"],
        "strategies_P2": ["A", "B", "C"],
        "A": A_s105,
        "B": B_s105,
        "expected_support": (2, 2)
    })

    # Jeu 10 : Rectangulaire 3x4 avec gains positifs et négatifs
    np.random.seed(123)
    A_34 = np.array([
        [ 3.0, -2.0,  5.0,  1.0],
        [-1.0,  4.0, -3.0,  2.0],
        [ 2.0,  0.0,  1.0, -4.0]
    ])
    B_34 = np.array([
        [ 1.0,  3.0, -1.0,  4.0],
        [ 2.0, -1.0,  5.0,  0.0],
        [-2.0,  4.0,  2.0,  1.0]
    ])
    games.append({
        "name": "Jeu Rectangulaire Asymétrique 3x4",
        "category": "3x4",
        "description": "Jeu rectangulaire 3x4 avec mélange de gains positifs et négatifs.",
        "strategies_P1": [f"P1_S{i+1}" for i in range(3)],
        "strategies_P2": [f"P2_S{j+1}" for j in range(4)],
        "A": A_34,
        "B": B_34,
        "expected_support": None
    })

    # Jeu 11 : Rectangulaire 4x5 (Montée en dimension)
    np.random.seed(456)
    A_45 = np.array([
        [ 4.0, -1.0,  2.0,  0.0,  3.0],
        [ 1.0,  5.0, -2.0,  3.0, -1.0],
        [-3.0,  2.0,  4.0, -1.0,  2.0],
        [ 2.0,  0.0,  1.0,  5.0, -2.0]
    ])
    B_45 = np.array([
        [ 2.0,  3.0, -2.0,  1.0,  4.0],
        [-1.0,  2.0,  4.0,  0.0, -2.0],
        [ 3.0, -1.0,  1.0,  5.0,  2.0],
        [ 0.0,  4.0, -3.0,  2.0,  1.0]
    ])
    games.append({
        "name": "Jeu Rectangulaire 4x5 (Montée en dimension)",
        "category": "4x5",
        "description": "Validation de la robustesse sur des dimensions supérieures 4x5.",
        "strategies_P1": [f"P1_S{i+1}" for i in range(4)],
        "strategies_P2": [f"P2_S{j+1}" for j in range(5)],
        "A": A_45,
        "B": B_45,
        "expected_support": None
    })

    return games
