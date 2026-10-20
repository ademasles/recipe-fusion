# Plan du projet recipe-fusion

Version du 20 octobre 2026. Ce fichier fixe le cadre et les exigences. Les choix de modèles, de
fusion et de protocole sont à écrire ici par l'auteur du projet, puis à faire relire **avant**
de lancer les calculs.

## L'énoncé

Projet 1 : classification multimodale. Prétraitement, extraction de représentations par
modalité, fusion, entraînement, évaluation comparative.

Date de rendu, format du rapport, travail seul ou en groupe : **[à compléter]**.

## La tâche

Prédire la classe (101 plats) d'une paire (photo, titre de la page web) d'UPMC Food-101
(Wang et al., 2015). Mesure principale : exactitude top-1 sur le jeu de test officiel
(22 716 paires) **[à confirmer]**.

## Exigences minimales du protocole

1. **Références unimodales** : un modèle image seule et un modèle texte seul, entraînés et
   évalués dans les mêmes conditions que les modèles multimodaux.
2. **Au moins trois stratégies de fusion**, comparées à budget comparable.
3. **Plusieurs graines** (au moins trois) pour chaque configuration ; rapporter moyenne et
   écart-type.
4. **Ablations par modalité** : que devient un modèle multimodal quand on retire ou dégrade une
   modalité au test ?
5. **Jeu de validation** prélevé dans l'entraînement, stratifié par classe et figé une fois pour
   toutes ; le jeu de test ne sert qu'au tableau final.

## Faits sur les données qui pèsent sur le plan

- Les titres sont courts : 7 mots en médiane, 15 au 95e centile.
- 73 % des titres contiennent le nom de la classe en toutes lettres (« Apple Pie Recipe »).
  La référence texte seul sera donc forte, et l'apport de l'image doit être mesuré sur ce fond.
- 39 % des titres de test se retrouvent à l'identique (en minuscules) dans l'entraînement.
  Doublons de pages ou titres génériques ? À examiner avant de conclure quoi que ce soit.
- 14 % des titres contiennent des entités HTML.

## À écrire avant de lancer les calculs

| Point | Choix | Justification |
|---|---|---|
| Prétraitement du texte | **[à compléter]** | |
| Prétraitement des images | **[à compléter]** | |
| Encodeur image | **[à compléter]** | |
| Encodeur texte | **[à compléter]** | |
| Encodeurs gelés ou affinés | **[à compléter]** | |
| Stratégies de fusion (≥ 3) | **[à compléter]** | |
| Protocole d'ablation | **[à compléter]** | |
| Sélection des hyperparamètres | **[à compléter]** | |
| Mesures et tests statistiques | **[à compléter]** | |

## Répartition (proposition, à valider par le groupe)

Même groupe que scene-to-score : `ademasles`, `Argodm`, `OriGiNaLGet9`. La répartition est pensée
sur les deux projets à la fois : chacun tient un rôle « données et évaluation » dans l'un et un
rôle « modèle » dans l'autre, et chacun écrit une fusion ici, puisque la fusion est le cœur du
cours.

| Personne | recipe-fusion (projet 1) | scene-to-score (projet 2) |
|---|---|---|
| ademasles | Données, évaluation et ablations ; fusion intermédiaire (attention croisée) | Encodeurs et extraction ; génération et démo |
| Argodm **[à confirmer]** | Branche texte et sa référence ; fusion précoce | Données et évaluation |
| OriGiNaLGet9 **[à confirmer]** | Branche image et sa référence ; fusion tardive | Projection et entraînement |

Contenu des rôles du projet 1 :

- **Données, évaluation et ablations** : jeu de validation figé, examen des doublons de titres
  entre entraînement et test, script d'évaluation commun, lancement à plusieurs graines,
  ablations par modalité, tableau final.
- **Branche texte** : prétraitement des titres, encodeur texte, référence texte seul,
  représentations mises en cache.
- **Branche image** : prétraitement des images, encodeur image, référence image seule,
  représentations mises en cache.
- **Fusions** : chacun conçoit, entraîne et justifie la sienne, sur les représentations en
  cache et avec le script d'évaluation commun. Les familles ci-dessus sont une proposition ;
  l'architecture précise revient à chacun.

À fixer en premier, avant tout travail en parallèle (rôle données et évaluation) :

1. Le fichier du jeu de validation, commité.
2. Le format du cache de représentations : un fichier par modalité et par découpage, lignes dans
   l'ordre des fichiers de titres, avec les identifiants d'images.
3. L'interface du script d'évaluation : prédictions en entrée, mesures en sortie.

Les deux collaborateurs n'ont pas de GPU NVIDIA. Les encodeurs gelés tournent une fois, au tech
lab ou sur le portable d'ademasles ; les têtes et les fusions s'entraînent ensuite sur processeur
à partir du cache. Affiner un encodeur se fait au tech lab.

## Calcul

Portable : RTX 4070, 8 Go. Ce qui dépasse va au tech lab. Estimation du temps par
configuration : **[à compléter]** après une première mesure.

## Terminé quand

- Le tableau références unimodales × fusions existe, à trois graines, sur le jeu de test.
- Les ablations par modalité sont faites.
- Le dépôt se relance de zéro avec les commandes du README.
- Le rapport est rendu.
