# Fiche de Soutenance M2 : Lemke–Howson
*(Aide-mémoire simple, direct et utile pour défendre le DM devant le professeur)*

---

## 1. Comment résumer ton projet en 1 minute au prof ?

> *"Monsieur, pour ce devoir maison, nous avons implémenté l'algorithme de Lemke–Howson **from scratch**, sans aucune bibliothèque d'optimisation externe.*  
> *Le script transforme le jeu bimatriciel $(A, B)$ en un **Problème de Complémentarité Linéaire (LCP)** avec deux tableaux de type Simplexe (un pour chaque joueur).*  
> *On part du sommet artificiel à l'origine $(0, 0)$ où toutes les stratégies pures sont non jouées (étiquettes $1$ à $m+n$). En relâchant une étiquette $k_0$, l'algorithme suit un chemin déterministe de pivots complémentaires le long des arêtes des polyèdres jusqu'à récupérer $k_0$.*  
> *Le sommet final est complètement étiqueté et non nul : en renormalisant les variables $z$ et $w$, on obtient directement l'équilibre de Nash $(x, y)$.*  
> *Nous avons validé l'algorithme sur des jeux $2 \times 2$, $3 \times 3$, des jeux rectangulaires asymétriques ($2 \times 3$, $3 \times 4$, $4 \times 5$) et avec des gains négatifs."*

---

## 2. Les 5 questions pièges du prof et tes réponses

### Question 1 : *"Pourquoi Lemke–Howson et pas l'énumération des supports (Slide 115) ?"*
* **Réponse :** L'énumération des supports teste à l'aveugle tous les sous-ensembles possibles de stratégies et résout un système linéaire pour chacun, ce qui donne une complexité exponentielle en $\mathcal{O}(4^n)$. Lemke–Howson est un algorithme de **pivotage complémentaire** : chaque pivot est polynomial ($\mathcal{O}((m+n)^2)$) et suit un chemin continu garanti d'aboutir à un équilibre.

### Question 2 : *"Pourquoi décaler les matrices de gains (Slide 110) ?"*
* **Réponse :** Pour poser les variables d'échelle $z = \frac{x}{v}$ et $w = \frac{y}{u}$, il faut que les gains maximaux $u$ et $v$ soient strictement positifs ($>0$). Ajouter une constante positive $c_A$ à $A$ et $c_B$ à $B$ rend tous les gains strictement positifs sans changer l'équilibre de Nash (propriété d'invariance affine).

### Question 3 : *"C'est quoi les étiquettes (Labels) ?"*
* **Réponse :** Il y a $m+n$ étiquettes au total.
  * Pour le Joueur 1, l'étiquette $i \in \{1, \dots, m\}$ signifie soit que $x_i = 0$ (l'action $i$ n'est pas jouée), soit que l'action $i$ est une meilleure réponse face à $y$ ($(Ay)_i = u$).
  * Pour le Joueur 2, l'étiquette $m+j \in \{m+1, \dots, m+n\}$ signifie soit que $y_j = 0$, soit que l'action $j$ est une meilleure réponse face à $x$ ($(B^T x)_j = v$).
  * Un équilibre de Nash correspond à un point qui possède **toutes les étiquettes de $1$ à $m+n$** (complémentarité).

### Question 4 : *"Comment êtes-vous sûr que votre algorithme ne boucle jamais ?"*
* **Réponse :** C'est une conséquence du **Lemme des poignées de main** en théorie des graphes. Dans le graphe d'étiquetage, chaque sommet presque complètement étiqueté a un degré au plus 2. L'origine $(0, 0)$ a un degré 1 (une seule arête possible pour $k_0$ fixé). Le graphe n'étant composé que de cycles et de chemins simples, le chemin qui part de $(0, 0)$ ne peut pas bifurquer ni cycler : il doit obligatoirement se terminer sur un autre sommet de degré 1, qui est nécessairement un équilibre de Nash.

### Question 5 : *"Comment gérez-vous la dégénérescence ?"*
* **Réponse :** En cas d'égalité sur le ratio minimum $\frac{b_i}{a_{ik}}$, un choix arbitraire pourrait provoquer du cyclage. Nous avons implémenté le **ratio minimum lexicographique** (`lexicographic_pivot_row`), qui compare les lignes colonne par colonne pour garantir un bris d'égalité unique et déterministe.

---

## 3. Comment tester n'importe quel jeu dans le script ?

Dans [dm_lemke_howson.py](file:///c:/Users/HP%20EliteBook%20840%20G6/Desktop/dm_thj/dm_lemke_howson.py), ajoute simplement à la fin :

```python
A = [[3, 0], 
     [5, 1]]

B = [[1, 2], 
     [0, 1]]

tester_jeu("Mon jeu perso", A, B)
```

Et lance dans le terminal :
```bash
python dm_lemke_howson.py
```
Le script affichera directement les matrices, $x$, $y$, les gains espérés, et validera si c'est bien un équilibre de Nash !
