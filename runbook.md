# Runbook d'astreinte — Pyrenex Prod (À COMPLÉTER)

> Pour l'équipe SRE Pyrenex. Rédigez **4 procédures**. Chacune doit suivre
> le même format : **déclenchement → actions → qui appeler → ce qu'on NE
> fait PAS**. Mini-cours : `05_Runbook_astreinte_essentiel.md`.

---

## 1. Service KO (un conteneur down)

**Déclenchement** : Vérification manuelle `docker ps` montre un conteneur en état **DOWN** (ou alerte Grafana si disponible).

**Actions** :

- Immédiatement : `docker logs <conteneur> | tail -100` pour identifier l'erreur de sortie
- `docker-compose restart <service>` pour relancer le service
- Attendre 30s et vérifier la reprise : `docker ps` + vérifier les logs pour traces d'erreur récurrente

**Qui appeler** : **DevOps/Infra** (infrastructure + orchestration, gestion des conteneurs, orchestration).

**On NE fait PAS** :

- ❌ Redémarrer le nœud complet sans d'abord sauvegarder les logs
- ❌ Supprimer les volumes Docker (risque perte de données persistantes)
- ❌ Modifier les variables d'env en prod directement sans plan de rollback

---

## 2. Latence p95 dégradée

**Déclenchement** : Seuil chiffré : **p95 > 2000ms (2 secondes)** détecté dans Grafana (panel « Vitesse »).

**Actions** (investigation parallèle) :

- Afficher le dashboard Prometheus/Grafana et identifier quel endpoint/requête spike
- Vérifier les ressources du host : CPU, RAM, I/O disque (potentiel saturation)
- Note : faire des snapshots Grafana pour post-mortem

**Qui appeler** : **Lead Dev API FastAPI** si la logique métier est lente

**On NE fait PAS** :

- ❌ Scale up les ressources CPU/RAM sans métriques justificatives (coût inutile)
- ❌ Monter les timeouts API pour masquer le problème
- ❌ Redémarrer le pod (une solution plus fine est préférable)

---

## 3. Métrique modèle qui s'écarte (distribution des prédictions anormale)

**Déclenchement** : Au moins un signal déclenche :

- Écart-type des prédictions > 2σ vs période stable (on constate une dérive comportementale dans les prédictions)
- KL-divergence (train vs production) > 0.5 (on vérifie que la distribution des classes est cohérentes avec l'entrainement)
- Pourcentage d'outliers sur scores > 5% vs baseline (il y a beaucoup plus de valeurs aberrantes qu'à la baseline)

**Actions — Phase 1 (audit données)** :

- Vérifier qu'aucune source n'a changé de schéma ou de contenu
- Comparer min/max/mean des features actuelles vs. période stable (via notebook ou job)
- Vérifier qu'aucun NaN/NULL aberrant n'a été introduit dans l'ETL

**Actions — Phase 2 (audit modèle, si données OK)** :

- Vérifier la **version du modèle** déployée (hash, tag git)
- Relancer une inférence manuelle sur un sample de test → comparer avec sortie attendue
- Vérifier les versions des dépendances ML (transformers, numpy, torch — elles doivent matcher `requirements.txt`)

**Qui appeler** : **Data Engineer / Data Scientist** pour investigation data puis modèle + potentiel réentraînement en sandbox.

**On NE fait PAS** :

- ❌ Réentraîner et déployer un modèle directement en prod (doit passer validation test d'abord)
- ❌ Minimiser les alertes 
- ❌ Modifier le seuil d'alerte pour éteindre le symptôme (masquage)

---

## 4. Rollback de release

**Déclenchement** : Au moins l'un des trois critères (à discrétion du Lead de garde) :

- Déploiement échoué (tests failing, health checks DOWN après 2 min)
- Métrique business/technique s'écroule en < 10 min post-deploy (ex. : p95 × 5, erreurs API > 5%)
- Support/client rapporte une régression fonctionnelle nouvelle

**Vérifications PRÉ-ROLLBACK** (« 4 yeux ») :

- ✅ Confirmer que la version précédente existe en registry Docker
- ✅ Vérifier qu'aucune migration BD **irréversible** n'a été appliquée (ou plan de rollback en place)
- ✅ Si possible, simuler le rollback en staging d'abord

**Exécution du rollback** :

- **Autorisation** : Lead Dev valide, Manager informe, astreinte exécute
- **Duo 4 yeux obligatoire** (sauf si perte économique directe directe → astreinte peut agir seule)
- Commande type : `docker-compose up -d --no-build <version_précédente>` ou `kubectl rollout undo ...`
- Vérifier les logs post-rollback : traces d'erreur ? santé retrouvée ?
- Création d'un ticket post-incident pour enquete ultérieure

**Qui appeler** : **Lead Dev** (validation), **DevOps/Infra** (exécution).

**On NE fait PAS** :

- ❌ Réessayer la version cassée avec « un petit fix » (itération dev en prod)
- ❌ Faire le rollback en silence (communication client/support obligatoire)
- ❌ Oublier de créer un ticket post-incident pour débugger pourquoi ça a cassé (retrospective)
