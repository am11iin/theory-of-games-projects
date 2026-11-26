# Theory of Games Projects

Academic coursework and implementations of game theory algorithms and applications.

---

## Projets disponibles

### 1. [SDPSO — PSO + Théorie des Jeux Évolutive](./SDPSO)
Application de la théorie des jeux évolutive (EGT) dans une métaheuristique d'optimisation par essaim particulaire (SDPSO).
- **Algorithmes :** PSO classique vs SDPSO (dynamique du réplicateur de Taylor & Jonker, 4 stratégies de recherche LIPS, UPSO, LDWPSO, CLPSO).
- **Tests & Benchmarks :** Fonctions Sphere, Schwefel 2.22, Rosenbrock, Rastrigin, Ackley, Griewank.
- **Référence :** Liu & Nishi (2022), *Information Sciences*.

### 2. [Homework Lemke–Howson](./homework_lemke_howson)
Implémentation de l'algorithme de Lemke–Howson pour le calcul des équilibres de Nash dans les jeux bimatriciels à somme non nulle.

### 3. [GT-GWO — Grey Wolf Optimizer + Théorie des Jeux Évolutive](./gwo)
Application de la théorie des jeux évolutive (EGT) dans la métaheuristique Grey Wolf Optimizer (GT-GWO).
- **Algorithmes :** GWO classique vs GT-GWO (dynamique du réplicateur de Taylor & Jonker, 4 stratégies de chasse TRIO_COOP, ALPHA_EXPLOIT, LEVY_EXPLORE, SOCIAL_PEER).
- **Variantes :** Problème continu (Rastrigin, Sphere) et problème combinatoire binaire (Sac à dos 0/1).
- **Tests & Benchmarks :** 120 exécutions indépendantes, analyse statistique de variance et tests de Wilcoxon.

