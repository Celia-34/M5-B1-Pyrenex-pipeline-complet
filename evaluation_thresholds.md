# Seuils d'évaluation continue — Pyrenex scoring v2

> Doit être lisible par Sophie Léger (Lead Data) et le DPO. **Chaque seuil
> est justifié** par une raison chiffrée.

Stratégie retenue : **hybride** (plancher absolu + baisse max vs golden run).
Le plancher absolu garantit un niveau de service minimum indépendant du golden
run ; la baisse max protège contre une dégradation progressive du modèle
tant que le plancher absolu n'est pas encore atteint.
Jeu de référence : `data/reference_set.csv` (sous-échantillon figé du holdout M1).

**⚠️ Point de vigilance:** le jeu de référence (golden run) est un sous échantillon de 500 lignes représentatif de la distribution des classes en production avec 18% de défauts. L'objectif est double : avoir une référence de la performance du modèle mais également des données. Si le taux de défauts venait à varier dans les données de production, cela serait détecté.

## Deux baselines, à ne pas confondre

| | Mesurée sur | Sert à |
|---|---|---|
| **Baseline communiquée** (`metrics_holdout`) | le holdout M1 complet | ce qu'on a annoncé au client |
| **Golden run** (`data/reference_baseline.json`) | **votre** jeu de référence, au gel | **arbitrer les releases** |

⚠️ Le garde-fou compare au **golden run**, jamais à la baseline communiquée :
les deux jeux n'ont ni la même taille ni la même composition, donc l'écart
entre eux mesure une **différence de population**, pas une dégradation du
modèle.

_**Les seuils ont été mesurés et évalués dans le fichier `thresholds.ipynb`**_

| Métrique | Golden run | Plancher absolu | Baisse max vs golden run | Justification |
|---|---|---|---|---|
| F1 macro | 0.5934 | 0.50 | 0.05 | 2σ mesuré = 0.0475 < 0.05 retenu : marge confortable, ~2.0% de fausses alertes sur le bootstrap. |
| F1 défaut | 0.4170 | 0.30 | 0.08 | 2σ mesuré = 0.0742 < 0.08 retenu : ~2.0% de fausses alertes. |
| ROC-AUC | 0.7248 | 0.65 | 0.04 | ⚠️ 2σ mesuré = 0.0553 > 0.04 retenu : tolérance sous le bruit, ~7.2% de fausses alertes sur le bootstrap. Assumé temporairement (métrique jugée moins critique pour la décision métier que le recall défaut), à relever à ≥ 0.06 si ces faux positifs de garde-fou deviennent gênants en CI. |
| Recall défaut | 0.6556 | 0.50 | 0.10 | 2σ mesuré = 0.1028 ≈ 0.10 retenu : tolérance alignée sur le bruit (justifiée par le faible nombre de défauts, ~90, dans le jeu de référence), ~2.6% de fausses alertes. |

> **Comment dimensionner la colonne « baisse max »** : mesurez le bruit de
> votre jeu de référence (bootstrap, cf. mini-cours 08), et prenez **au moins
> 2 σ**. Une tolérance sous le bruit se déclenche toute seule. Reportez ici le
> σ mesuré — c'est ce qui rend le seuil défendable devant Sophie Léger.

Bootstrap : 2000 ré-échantillonnages avec remise du jeu de référence (500 lignes,
90 défauts, 18%), `random_state=42` — voir `notebook/thresholds.ipynb`.

| Métrique | σ bootstrap mesuré | 2 σ | Tolérance retenue | Fausses alertes (bootstrap) |
|---|---|---|---|---|
| F1 macro | 0.0237 | 0.0475 | 0.05 | 2.0% |
| F1 défaut | 0.0371 | 0.0742 | 0.08 | 2.0% |
| ROC-AUC | 0.0276 | 0.0553 | 0.04 | 7.2% |
| Recall défaut | 0.0514 | 0.1028 | 0.10 | 2.6% |

## Procédure de mise à jour des seuils

- **Qui** : Sophie Léger (Lead Data), en lien avec le DPO pour la partie garde-fou données.
- **Quand** : à chaque changement de golden run (nouveau modèle validé en v2.x et meilleur) ou si le
  jeu de référence change (nouveau tirage depuis le holdout M1).
- **Comment** : garder `THRESHOLDS` dans `scripts/evaluate_model.py` ET ce fichier
  cohérents ; si le jeu de référence change, **regeler le golden run**
  (`python scripts/evaluate_model.py --freeze-baseline`) puis relancer le
  bootstrap (`notebook/thresholds.ipynb`) pour revalider les tolérances.
