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

## Calcul

Portable : RTX 4070, 8 Go. Ce qui dépasse va au tech lab. Estimation du temps par
configuration : **[à compléter]** après une première mesure.

## Terminé quand

- Le tableau références unimodales × fusions existe, à trois graines, sur le jeu de test.
- Les ablations par modalité sont faites.
- Le dépôt se relance de zéro avec les commandes du README.
- Le rapport est rendu.
