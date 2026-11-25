# SDPSO — PSO + Théorie des Jeux Évolutive

Projet universitaire de Master — Implémentation Python de l'algorithme **SDPSO** (*Strategy Dynamics Particle Swarm Optimizer*).

> **Référence scientifique :** Ziang Liu & Tatsushi Nishi, *Strategy dynamics particle swarm optimizer*, Information Sciences, vol. 582, pp. 665–703, 2022. [DOI: 10.1016/j.ins.2021.10.028](https://doi.org/10.1016/j.ins.2021.10.028)
> **Code source MATLAB original :** [github.com/zi-ang-liu/SDPSO](https://github.com/zi-ang-liu/SDPSO)

---

## Présentation du projet

L'optimisation par essaim particulaire (**PSO**) est une métaheuristique inspirée du comportement collectif des oiseaux et des poissons. Chaque particule explore l'espace de recherche en s'inspirant de ses propres meilleures positions passées (*pbest*) et de la meilleure position trouvée par l'ensemble du swarm (*gbest*).

Le principal défaut du PSO classique est sa convergence prématurée : toutes les particules utilisent la même stratégie de déplacement, ce qui finit par appauvrir la diversité de l'essaim.

**SDPSO** résout ce problème en intégrant la **Théorie des Jeux Évolutive (EGT)** : quatre stratégies de recherche sont mises en compétition au sein de l'essaim, et leurs probabilités d'adoption évoluent dynamiquement selon les performances observées, via l'**équation du réplicateur**.

---

## La Théorie des Jeux dans SDPSO

### Principe

La population est divisée en deux sous-groupes :
- **30 % des particules** : utilisent en permanence la stratégie CLPSO pour maintenir la diversité.
- **70 % des particules** : choisissent leur stratégie à chaque itération selon un vecteur de probabilités `p = [p₁, p₂, p₃, p₄]`.

### Les 4 stratégies en compétition

| Stratégie | Description |
|-----------|-------------|
| **LIPS** | Voisinage local basé sur la distance euclidienne aux *pbest* (sans *gbest*) |
| **UPSO** | Combinaison de la vitesse globale et locale sur une topologie en anneau |
| **LDWPSO** | PSO standard avec inertie et coefficients d'accélération variables dans le temps |
| **CLPSO** | Apprentissage dimension par dimension à partir des *pbest* d'autres particules |

### L'équation du réplicateur

Toutes les 200 évaluations, les probabilités sont mises à jour selon la formule de **Taylor & Jonker (1978)** :

```
p_k(t+1) = p_k(t) + α · p_k(t) · (f_k - f̄)
```

- `f_k` = gain normalisé de la stratégie k (amélioration relative moyenne de *pbest*)
- `f̄ = Σ p_j · f_j` = gain moyen du swarm
- `α = 0.1` = taux d'apprentissage

Les stratégies qui obtiennent de meilleurs gains que la moyenne voient leur probabilité augmenter ; les autres diminuent.

**Mécanisme de mutation :** `p_k = max(p_k, 0.02)` → aucune stratégie ne peut disparaître définitivement.

**Redémarrage périodique :** à 25 %, 50 % et 75 % du budget d'évaluations, `p` est réinitialisé à `[0.25, 0.25, 0.25, 0.25]` pour éviter la sur-spécialisation.

---

## Architecture du projet

```
SDPSO_projet1/
│
├── src/
│   ├── pso.py          # PSO classique (algorithme de référence)
│   ├── game_theory.py  # Dynamique du réplicateur, gains, mutation, restart
│   └── hybrid_pso.py   # Algorithme SDPSO complet (4 stratégies + EGT)
│
├── experiments/
│   ├── benchmark.py    # Fonctions de test et boucle de comparaison
│   └── visualize.py    # Génération des graphiques
│
├── tests/
│   └── test_pso.py     # Tests unitaires (16 tests)
│
├── results/            # Graphiques et statistiques générés
├── main.py             # Point d'entrée
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation et exécution

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/SDPSO-projet.git
cd SDPSO-projet

# Installer les dépendances (uniquement numpy et matplotlib)
pip install -r requirements.txt

# Lancer les tests unitaires
python -m pytest tests/ -v

# Test rapide (D=10, 3 répétitions, ~10 secondes)
python main.py --quick

# Lancer la comparaison complète (D=30, 10 répétitions)
python main.py --dim 30 --runs 10 --fes 1000
```

### Options disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--dim` | Dimension de l'espace de recherche | 30 |
| `--runs` | Nombre de répétitions Monte Carlo | 10 |
| `--fes` | Budget d'évaluations par dimension | 1000 |
| `--pop` | Taille du swarm | 40 |
| `--quick` | Mode rapide (D=10, 3 runs) | — |

---

## Fonctions benchmark utilisées

| Fonction | Type | Domaine |
|----------|------|---------|
| Sphere | Unimodale, séparable | [-100, 100] |
| Schwefel 2.22 | Unimodale, non-séparable | [-10, 10] |
| Rosenbrock | Vallée en banane | [-30, 30] |
| Rastrigin | Multimodale accidentée | [-5.12, 5.12] |
| Ackley | Multimodale, plateau ext. | [-32, 32] |
| Griewank | Multimodale, non-séparable | [-600, 600] |

---

## Résultats expérimentaux

Comparaison sur D=10, 5 répétitions, 20 000 évaluations (meilleur résultat de chaque algorithme) :

| Fonction | PSO (moy.) | SDPSO (moy.) | Meilleur SDPSO |
|----------|-----------|--------------|----------------|
| Rastrigin | 4.98 | **1.05** | **0.024** ↑ +78.9% |
| Rosenbrock | 28.18 | **25.13** | **4.15** ↑ +10.8% |
| Sphere | 1.08e-11 | 8.84e-8 | — |
| Ackley | 2.11e-6 | 1.55e-4 | — |

> **Note :** Sur les fonctions unimodales simples (Sphere, Ackley), le PSO classique converge plus vite car il n'alloue aucune ressource à l'exploration. SDPSO paie un léger coût exploratoire pour garantir la robustesse sur les paysages multimodaux.

---

## Dépendances

- Python ≥ 3.9
- numpy ≥ 1.24
- matplotlib ≥ 3.7
