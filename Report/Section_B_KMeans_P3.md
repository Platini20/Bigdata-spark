# Partie B – Segmentation K-Means (P3)

Code : `src/segmentation.py` (fonction `segmenter`), tests : `src/tester_P3.py`.
Tous les chiffres de cette section proviennent de `results/kmeans_P3_test.json`,
produit par `tester_P3.py --reel` en `local[*]` (voir « Limites »).

## B.1 Variables et standardisation

La segmentation utilise les quatre variables RFM étendues produites par
`analyse_sql.construire_rfm` pour chacun des 12 000 clients (0 ligne retirée
pour valeur manquante) :

| Variable | Signification | Étendue observée |
|---|---|---|
| `recency_days` | jours entre le dernier achat et le 1er novembre 2026 | 1 à 328 jours |
| `frequency` | nombre de transactions distinctes | 4 à 35 |
| `amount` | montant total dépensé (quantité × prix unitaire) | 58,24 $ à 5 730,34 $ |
| `product_diversity` | nombre de catégories distinctes achetées | 1 à 8 |

K-Means affecte chaque client au centre le plus proche selon la distance
euclidienne. Sans mise à l'échelle, le montant (des milliers de dollars)
écraserait la diversité (quelques unités) : un écart de 100 $ pèserait autant
que 100 catégories. Les variables sont donc assemblées avec `VectorAssembler`,
puis centrées-réduites avec `StandardScaler(withMean=True, withStd=True)`.
Chaque variable a alors une moyenne de 0 et un écart-type de 1, et contribue à
la distance à égalité. Le scaler est ajusté une seule fois. Le DataFrame
standardisé est mis en cache pendant les 15 entraînements, puis libéré.

## B.2 Comparaison de K = 2 à 6

Pour chaque K, K-Means (`maxIter=50`) est entraîné avec trois seeds (42, 7 et
2026). Pour chaque K, on relève :

- le **coût** (`trainingCost`, seed 42) ;
- la **silhouette moyenne** sur les trois seeds (distance euclidienne au
  carré, variables standardisées) ;
- l'**ARI moyen** (indice de Rand ajusté) entre le seed 42 et chacun des deux
  autres seeds. C'est la mesure de stabilité : 1 signifie des partitions
  identiques ;
- la part du **plus petit cluster**, la plus faible observée sur les trois
  seeds.

| K | Coût (seed 42) | Silhouette moyenne | ARI moyen | Plus petit cluster | Décision |
|---|---|---|---|---|---|
| 2 | 18 973,52 | 0,7028 | 0,6671 | 26,12 % | Écarté : instable (ARI < 0,80) |
| 3 | 7 594,46 | **0,7815** | **1,0000** | 25,05 % | Retenu par la règle, remplacé après revue |
| 4 | 4 555,94 | 0,7493 | 0,8195 | 7,69 % | **Retenu après revue humaine** |
| 5 | 4 098,76 | 0,7399 | 0,8236 | 7,45 % | Écarté : silhouette inférieure |
| 6 | 3 012,20 | 0,7125 | 0,8905 | 7,45 % | Écarté : silhouette inférieure |

Écarts-types de la silhouette entre les seeds : 0,0267 (K=2), 0,0000 (K=3),
0,0390 (K=4), 0,0290 (K=5) et 0,0328 (K=6).

## B.3 Choix de K : règle automatique, puis revue humaine

**Règle automatique** (codée dans `segmenter` et reproduite dans le JSON) :

1. Les candidats ont un ARI moyen ≥ 0,80 et un plus petit cluster ≥ 2 % des
   clients.
2. Parmi les candidats, on prend la silhouette moyenne la plus haute. Si un K
   plus petit est à moins de 0,01, on préfère ce K plus petit.
3. S'il n'y a aucun candidat, on prend la silhouette moyenne la plus haute.

K = 2 est écarté parce qu'il est instable : son ARI moyen de 0,67 montre que
la partition change selon le seed. Les candidats sont K = 3, 4, 5 et 6, et
**la règle retient K = 3** : c'est la meilleure silhouette (0,78) et la
partition est parfaitement stable (ARI 1,00, écart-type de la silhouette
nul).

**Revue des profils.** L'interprétabilité fait partie des critères de choix,
et la règle ne peut pas la mesurer. La revue des quartiles de K = 3 montre que
le cluster intermédiaire (5 946 clients, environ la moitié de la base) est
**bimodal en fréquence** : 2 952 clients font 7 ou 8 achats, 2 994 en font 16
à 18, et un seul se situe entre les deux. Sa fréquence moyenne (13) ne
correspond donc à aucun client réel, et aucun nom métier unique ne décrit ce
groupe.

Un tableau croisé K = 3 × K = 4 (seed 42) confirme que K = 4 sépare ces deux
sous-groupes sans déstabiliser le reste :

- 95,9 % du groupe à 7–8 achats et 93,6 % du groupe à 16–18 achats tombent
  dans deux clusters distincts de K = 4 ;
- les champions (99,8 %) et les inactifs (98,8 %) restent dans le même
  cluster.

**K = 4 est donc retenu après revue humaine** (`k_force=4`, valeur par défaut
`K_APRES_REVUE` du module). Ce choix a un coût mesuré :

- la silhouette moyenne passe de 0,7815 à 0,7493 (−0,03). Avec le seed 42
  seul, elle est de 0,7718 ;
- l'ARI moyen passe de 1,00 à 0,82, ce qui reste au-dessus du seuil de 0,80 ;
- le plus petit cluster vaut 7,69 % sur l'ensemble des seeds : un des seeds
  converge vers un minimum local moins bon. Le modèle final (seed 42) a quatre
  clusters de 24,4 % à 25,5 %.

Nous acceptons cette baisse modérée de séparation contre quatre segments
cohérents, sur lesquels on peut agir.

## B.4 Profil des clusters (K = 4, seed 42, variables originales)

Les valeurs sont la médiane (`percentile_approx`), puis la moyenne entre
parenthèses.

| | Cluster 1 | Cluster 3 | Cluster 0 | Cluster 2 |
|---|---|---|---|---|
| **Nom métier** | Clients champions | Clients réguliers | Acheteurs occasionnels | Clients inactifs / à risque |
| Effectif | 3 000 | 2 927 | 3 061 | 3 012 |
| Part | 25,00 % | 24,39 % | 25,51 % | 25,10 % |
| Récence (jours) | 2 (2,5) | 6 (9,2) | 35 (40,2) | 196 (202,4) |
| Fréquence | 35 (34,9) | 18 (17,6) | 8 (8,6) | 5 (5,0) |
| Montant ($) | 4 217,95 (4 217,93) | 1 302,59 (1 277,25) | 320,87 (383,44) | 143,82 (148,19) |
| Diversité (catégories) | 6 (6,0) | 4 (4,5) | 3 (2,6) | 2 (2,1) |
| Centre z (R, F, M, D) | −0,74 ; 1,58 ; 1,65 ; 1,21 | −0,66 ; 0,09 ; −0,14 ; 0,36 | −0,28 ; −0,67 ; −0,68 ; −0,65 | 1,66 ; −0,98 ; −0,82 ; −0,90 |

Actions marketing proposées :

- **Clients champions** (achat récent, fréquence et montant les plus élevés) :
  programme de fidélité haut de gamme, accès anticipé aux nouveautés. On
  cherche à les garder, pas à leur donner des rabais.
- **Clients réguliers** (actifs, 18 achats, environ 1 300 $) : ventes croisées
  vers de nouvelles catégories et paliers de fidélité, pour les faire monter
  vers le segment champions.
- **Acheteurs occasionnels** (8 achats, environ 320 $, dernier achat il y a
  environ 5 semaines) : relances ciblées et offres de réachat, pour augmenter
  la fréquence.
- **Clients inactifs / à risque** (dernier achat il y a plus de 6 mois) :
  campagne de réactivation limitée dans le temps. Ceux qui ne réagissent pas
  sortent des envois coûteux.

Les noms sont générés par des règles sur les centres en z-scores (seuil de
±0,5 écart-type), puis validés en lisant les profils ci-dessus.

## B.5 Pourquoi le coût seul ne peut pas choisir K

Le coût est la somme des distances au carré entre chaque client et son centre.
Il diminue presque toujours quand K augmente : ici, il passe de 18 974 (K=2) à
3 012 (K=6), et il vaudrait 0 avec un cluster par client. Le minimiser revient
donc à choisir le plus grand K testé. La « méthode du coude » ne règle pas le
problème, car elle est subjective. Les baisses successives sont de −60 %
(2→3), −40 % (3→4), −10 % (4→5) et −26,5 % (5→6). On peut y voir un coude à 3
ou à 4, et la reprise entre 5 et 6 vient d'un minimum local du seed 42 pour
K = 5. Enfin, le coût mesure la compacité à l'intérieur des clusters, pas leur
séparation. C'est pourquoi on le combine avec la silhouette (séparation),
l'ARI (stabilité entre seeds) et la lecture des profils (interprétabilité).

## B.6 Limites

- **Montant asymétrique.** Le montant s'étend de 58 $ à 5 730 $. Dans le
  cluster des acheteurs occasionnels, la moyenne (383 $) dépasse nettement la
  médiane (321 $), signe d'une queue à droite. K-Means utilise des moyennes et
  est sensible à ces valeurs extrêmes. Une transformation logarithmique du
  montant n'a pas été testée.
- **Noms métier.** Le seuil de ±0,5 écart-type utilisé pour nommer les
  clusters est un choix de l'équipe, pas une valeur de référence. Les noms ont
  été vérifiés à la main sur les profils.
- **Revue humaine.** Le passage de K = 3 à K = 4 repose sur une lecture des
  quartiles. Il est documenté dans le JSON (`k_regle`, `k_retenu`,
  `revue_humaine`). Environ 5 % du cluster bimodal (313 clients) est affecté
  « de l'autre côté » de la coupe.
- **Environnement.** Ces chiffres ont été obtenus en `local[*]` dans l'image
  `inf6303-spark-mllib:4.2.0`, et non sur le cluster. Les chiffres officiels
  du rapport (dont les durées) sont ceux de `resultat_1_worker.json` et
  `resultat_3_workers.json`, produits par P5. K-Means est déterministe à seed
  fixe, mais le coût et la silhouette peuvent varier légèrement selon le
  partitionnement.

## Note d'intégration pour P5

- Dans `src/devoir2_spark.py`, remplacer le stub `segmenter` par
  `from segmentation import segmenter`. L'appel `segmenter(rfm)` reste le même
  et retourne K = 4 (valeur par défaut `k_force=K_APRES_REVUE`). Le
  dictionnaire va dans `result["kmeans"]`. `segmenter(rfm, k_force=None)`
  applique la règle seule (K = 3). `rfm` doit venir de
  `analyse_sql.construire_rfm`.
- Durée : `segmenter` prend environ 1,5 min en local (de 84 à 111 s mesurées
  sur 4 exécutions, 93,6 s pour le JSON ci-dessus). Prévoir plus sur 1 worker
  à 1 cœur. La durée est dans `result["kmeans"]["duree_secondes"]` pour la
  Partie D.
- Il n'y a pas de `.gitignore`, et des fichiers `src/__pycache__/*.pyc` sont
  versionnés. Lancer Python dans le conteneur réécrit ces fichiers à travers le
  volume monté. En attendant, utiliser `PYTHONDONTWRITEBYTECODE=1`.
  `results/kmeans_P3_test.json` est un fichier de test, à ne pas commiter.
- Le port 8080 (Master UI) peut être occupé sur certaines machines, ce qui
  fait échouer `docker compose up`. Il faut le libérer avant de lancer
  `executer_devoir2.ps1`.
- En `local[*]`, Spark 4.2 a écrit une fois une `NullPointerException` dans le
  thread `executor-heartbeater` (`PartitionKeyedAccumulator`). Le job s'est
  terminé normalement. À surveiller sur le cluster.
