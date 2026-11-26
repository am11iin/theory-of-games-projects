# GT-GWO : Game Theory Grey Wolf Optimizer
**Hybridation entre la Théorie des Jeux Évolutive et la Métaheuristique Grey Wolf Optimizer**

*Projet universitaire encadré par le Dr Djamila Dahmani.*

---

## 1. Description du projet
Ce projet étudie l'intégration d'un modèle de **Théorie des Jeux Évolutive (EGT)** basé sur l'équation du réplicateur de **Taylor & Jonker (1978)** au sein de la métaheuristique **Grey Wolf Optimizer (GWO)** (Mirjalili et al., 2014).

L'objectif est de remplacer le schéma rigide et statique d'exploration/exploitation de GWO (décroissance linéaire de $a$ de 2 à 0) par une **dynamique adaptative collective** arbitrée par les gains relatifs des stratégies de recherche.

---

## 2. Structure du code

```text
gwo/
├── src/                                  # Code source des algorithmes
│   ├── __init__.py
│   ├── gwo.py                           # GWO classique continu (baseline)
│   ├── binary_gwo.py                    # Binary GWO (BGWO) pour problèmes discrets
│   ├── game_theory.py                   # Modèle EGT : dynamique du réplicateur & 4 stratégies
│   ├── gt_gwo.py                        # Algorithme hybride GT-GWO continu
│   ├── binary_gt_gwo.py                 # Algorithme hybride BGT-GWO discret
│   └── knapsack_problem.py              # Instance sac à dos + solveur exact DP
├── experiments/                          # Protocoles expérimentaux
│   └── experiments_comparison.py        # Benchmark comparatif complet (Wilcoxon, métriques)
├── results/                              # Résultats d'expériences
│   ├── data/                            # Fichiers CSV (tableaux statistiques)
│   └── figures/                         # Courbes de convergence et dynamiques de stratégies
├── main.py                               # Point d'entrée principal interactif
└── requirements.txt                     # Dépendances Python (numpy, matplotlib, scipy)
```

---

## 3. Modélisation de Théorie des Jeux

- **Joueurs :** Les $N$ loups de la meute.
- **Stratégies de chasse :**
  1. `TRIO_COOP` : Guidage standard par le trio $(\alpha, \beta, \delta)$.
  2. `ALPHA_EXPLOIT` : Attaque ciblée et focalisée sur le leader $\alpha$.
  3. `LEVY_EXPLORE` : Saut stochastique de vol de Lévy pour s'échapper des minima locaux.
  4. `SOCIAL_PEER` : Coopération horizontale par apprentissage entre pairs ($X_{r1} - X_{r2}$).
- **Calcul des gains (Payoffs) :**
  $$g_i = \max\left(0, \frac{f_{\text{ancien}} - f_{\text{nouveau}}}{|f_{\text{ancien}}| + \epsilon}\right)$$
- **Dynamique du réplicateur (Taylor & Jonker) :**
  $$p_k(t+1) = p_k(t) + \alpha_{\text{learning}} \cdot p_k(t) \cdot (F_k - \bar{F})$$
- **Seuil de mutation anti-extinction :** $p_k \ge 0.05$.

---

## 4. Synthèse des résultats expérimentaux (120 runs)

| Problème | Algorithme | Best | Worst | Mean | Std | Succès | Temps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rastrigin** (Continu) | GWO classique | `0.0` | `0.1479` | `4.9305e-03` | `2.7006e-02` | 96.7% | 0.27s |
| **Rastrigin** (Continu) | **GT-GWO (Hybride)** | `0.0` | `0.0` | **`0.0000e+00`** | **`0.0000e+00`** | **100.0%** | 0.36s |
| **Sac à Dos** (Discret) | BGWO classique | `573.0` | `570.0` | `572.80` | `0.6103` | 86.7% | 1.86s |
| **Sac à Dos** (Discret) | **BGT-GWO (Hybride)**| `573.0` | `572.0` | **`572.90`** | **`0.3051`** | **90.0%** | **1.11s** |

---

## 5. Exécution rapide

Installer les dépendances :
```bash
pip install -r requirements.txt
```

Lancer le menu interactif :
```bash
python main.py
```

Lancer directement le protocole comparatif complet :
```bash
python experiments/experiments_comparison.py
```
