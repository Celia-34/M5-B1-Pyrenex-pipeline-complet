# Seuils d'évaluation continue — Pyrenex scoring v2

Stratégie retenue **hybride** : un seul seuil relatif ne protège pas contre une dérive lente. Un seuil absolu ne détecte pas une régression brutale qui resterait au-dessus du plancher métier. Les deux ensembles couvrent les deux risques.
Jeu de référence : `data/reference_set.csv` (sous-échantillon figé du holdout M1).

## Deux baselines, à ne pas confondre

| | Mesurée sur | Sert à |
|---|---|---|
| **Baseline communiquée** (`metrics_holdout`) | le holdout M1 complet | ce qu'on a annoncé au client |
| **Golden run** (`data/reference_baseline.json`) |  jeu de référence gelé | **arbitrer les releases** |

⚠️ Le garde-fou compare au **golden run**, jamais à la baseline communiquée :
les deux jeux n'ont ni la même taille ni la même composition, donc l'écart
entre eux mesure une **différence de population**, pas une dégradation du
modèle. Vérifié en pratique : `--release-tag ok` sur modèle inchangé donne
`violations: []`, écart exactement nul au golden run.
 
| Métrique | Golden run | Plancher absolu | Baisse max vs golden run | Justification |
|---|---|---|---|---|
| F1 macro | 0.6728 | 0.50 | 0.043 | σ bootstrap = 0.0216 → 2σ = 0.043 : sous ce seuil, la baisse ne peut plus s'expliquer par le bruit d'échantillonnage de notre jeu. Plancher 0.50 = point où le tri perd sa valeur discriminante réelle. |
| F1 défaut | 0.6316 | 0.35 | 0.055 | σ bootstrap = 0.0273 → 2σ = 0.055. Plancher 0.35 = seuil où le modèle rate ou signale à tort trop de dossiers de défaut pour rester utile en tri manuel. |
| ROC-AUC | 0.7726 | 0.65 | 0.041 | σ bootstrap = 0.0204 → 2σ = 0.041. Plancher 0.65 = seuil classique en scoring de risque en dessous duquel le modèle discrimine à peine mieux qu'un tirage aléatoire pondéré. |
| Recall défaut | 0.552 | 0.45 | 0.065 | σ bootstrap = 0.0325 (la plus bruitée des 4, cohérent car mesurée sur la classe qu'on a choisi de mieux échantillonner) → 2σ = 0.065. Plancher 0.45 = seuil où plus d'un défaut sur deux échappe au tri, la métrique la plus directement liée au risque métier pour Pyrenex. |
 
> **Comment dimensionner la colonne « baisse max »** : mesurez le bruit de
> votre jeu de référence (bootstrap, cf. mini-cours 08), et prenez **au moins
> 2 σ**. Une tolérance sous le bruit se déclenche toute seule. Reportez ici le
> σ mesuré — c'est ce qui rend le seuil défendable devant Sophie Léger.
 
| Métrique | σ bootstrap mesuré | 2 σ | Tolérance retenue |
|---|---|---|---|
| F1 macro | 0.0216 | 0.0432 | 0.043 |
| F1 défaut | 0.0273 | 0.0547 | 0.055 |
| ROC-AUC | 0.0204 | 0.0409 | 0.041 |
| Recall défaut | 0.0325 | 0.0650 | 0.065 |
 
*(500 tirages avec remise, `random_state=42`, sur `data/reference_set.csv`.)*
 
## Procédure de mise à jour des seuils
 
- **Qui** : la personne qui modifie le modèle ou le jeu de référence (dev + revue par Sophie Léger avant merge sur `main`).
- **Quand** : à chaque changement du modèle en profondeur (nouvelle version majeure, ex. v3.0) ou du jeu de référence lui-même — jamais à la légère pour faire passer une release au vert.
- **Comment** : garder `THRESHOLDS` dans le script ET ce fichier cohérents ; si le jeu de référence change, **regeler le golden run** — `--freeze-baseline` — puis relancer le bootstrap pour requantifier le bruit, sans quoi on recompare deux populations différentes.