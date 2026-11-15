#### **INF6303 — Devoir 2 : Pipeline analytique distribué avec Apache Spark**



1. ###### **Présentation du projet**



Ce projet a été réalisé dans le cadre du cours INF6303 — Techniques d'analyses des mégadonnées.



Il consiste à concevoir et à exécuter un pipeline analytique distribué avec Apache Spark et PySpark, dans un environnement conteneurisé avec Docker.



Le pipeline couvre cinq composantes :



* P1 — Préparation des données : chargement, validation et nettoyage.
* P2 — Analyse SQL : construction des indicateurs RFM et analyses complémentaires.
* P3 — Segmentation : regroupement des clients par K-Means.
* P4 — Recommandation : recommandations de produits avec ALS.
* P5 — Intégration et évaluation : orchestration, tests et comparaison des performances distribuées.



###### **2. Architecture technique**



L'environnement repose sur Docker Compose et comprend :



* Un conteneur Spark Master.
* Un conteneur Spark Client.
* Un ou trois conteneurs Spark Worker.



Le Spark Master coordonne les ressources du cluster.



Le Spark Client soumet les applications PySpark.



Les Workers exécutent les tâches distribuées.



\### **Configuration**



| Paramètre | Valeur |

|---|---|

| Apache Spark | 4.2.0 |

| Image Docker | spark:4.2.0-python3 |

| NumPy | 2.2.6 |

| Nombre de Workers | 1 ou 3 |

| Cœurs par Worker | 1 |

| Mémoire par Worker | 1 400 Mo |

| Mémoire par Executor | 1 100 Mo |

| Mémoire du Driver | 1 Go |

| Partitions Shuffle | 24 |

| Allocation dynamique | Désactivée |

| Mode de déploiement | Client |



###### **3. Organisation du projet**



```text

Devoir2\_Spark/

│

├── compose.yaml

├── README.md

├── .gitignore

│

├── docker/

│   └── Dockerfile

│

├── data/

│   ├── catalogue.csv

│   ├── clients.csv.gz

│   ├── transactions.csv.gz

│   └── MANIFESTE.json

│

├── src/

│   ├── devoir2\_spark.py

│   ├── devoir2\_spark\_P1.py

│   ├── analyse\_sql.py

│   ├── segmentation.py

│   ├── als\_config.py

│   ├── als\_recommandation.py

│   └── analyser\_plan\_spark.py

│

├── scripts/

│   ├── executer\_devoir2.ps1

│   └── arreter\_cluster.ps1

│

├── tests/

│   ├── test\_integration\_p1\_p2.py

│   └── test\_integration\_p1\_p2\_p3.py

│

├── results/

│   ├── resultat\_1\_worker.json

│   ├── resultat\_3\_workers.json

│   ├── synthese\_comparaison\_workers.json

│   ├── test\_integration\_p1\_p2\_p3.json

│   ├── plan\_execution\_rfm.txt

│   └── sauvegardes/

│

└── Report/

```



Cette arborescence présente les principaux fichiers utilisés dans la version intégrée du projet.



###### **4. Données utilisées**



Le pipeline exploite trois sources principales :



| Source | Description |

|---|---|

| transactions.csv.gz | Historique des transactions |

| clients.csv.gz | Informations sur les clients |

| catalogue.csv | Catalogue des produits |



Les expérimentations ont porté sur :



\- 197 677 transactions.

\- 12 000 clients.

\- 800 produits.



###### **5. Fonctionnement du pipeline**



**P1 — Préparation des données**



Cette étape assure le chargement, la validation et le nettoyage des données.



Elle prépare les trois sources pour les traitements analytiques suivants.



**P2 — Analyse SQL et RFM**



Cette étape utilise les opérations SQL de Spark pour construire les indicateurs clients.



Les quatre variables utilisées pour la segmentation sont :



\- `recency\_days` : récence des achats.

\- `frequency` : fréquence des achats.

\- `amount` : montant des achats.

\- `product\_diversity` : diversité des produits achetés.



**P3 — Segmentation K-Means**



La segmentation repose sur les quatre variables RFM standardisées.



Les valeurs de K étudiées sont :



2, 3, 4, 5 et 6.



La sélection automatique repose notamment sur la silhouette et la stabilité des partitions.



La règle automatique privilégie K=3.



Cependant, l'analyse des profils a conduit à retenir K=4 après revue humaine.



Les quatre profils métier obtenus sont :



1\. Clients inactifs ou à risque.

2\. Clients réguliers.

3\. Clients champions.

4\. Acheteurs occasionnels.



**P4 — Recommandation ALS**



Le module de recommandation utilise l'algorithme Alternating Least Squares (ALS) de Spark MLlib.



Deux configurations sont comparées :



| Paramètre | Configuration A | Configuration B |

|---|---|---|

| Rank | 10 | 20 |

| maxIter | 10 | 10 |

| regParam | 0,1 | 0,1 |

| alpha | 40 | 40 |

| Seed | 42 | 42 |



Le protocole utilise une séparation entraînement/test de 80 % / 20 %.



L'indicateur principal est le Recall@10.



**P5 — Intégration et performances**



Cette étape assure :



\- L'orchestration des traitements.

\- La vérification des modules.

\- L'exécution sur différentes configurations du cluster.

\- La collecte des résultats.

\- L'analyse comparative des performances.

\- L'inspection du plan d'exécution Spark.



###### **6. Installation et exécution**



**### Prérequis**



\- Docker Desktop.

\- Docker Compose.

\- Windows PowerShell.

\- Git.



**### Démarrer l'environnement**



Depuis le répertoire racine du projet :



```powershell

docker compose up -d --build

```



**### Vérifier les conteneurs**



```powershell

docker compose ps

```



**### Exécuter avec 1 Worker**



```powershell

powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\executer\_devoir2.ps1 -Workers 1 -Partitions 24

```



**### Exécuter avec 3 Workers**



```powershell

powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\executer\_devoir2.ps1 -Workers 3 -Partitions 24

```



Le script vérifie la disponibilité du nombre de Workers demandé avant de soumettre le traitement Spark.



**Attention** : une nouvelle exécution remplace le fichier JSON correspondant. Il est recommandé de sauvegarder les résultats précédents avant de relancer une expérimentation.



**### Arrêter le cluster**



```powershell

docker compose down

```



###### **7. Tests d'intégration**



Deux tests d'intégration ont été réalisés :



\- P1 + P2.

\- P1 + P2 + P3.



Les deux tests ont été exécutés avec succès.



Le test P1 + P2 + P3 a notamment confirmé :



\- La disponibilité des données.

\- La construction des indicateurs RFM.

\- L'exécution de K-Means.

\- La sélection finale de quatre segments.



Les résultats sont conservés dans le répertoire `results/`.



###### **8. Résultats expérimentaux**



**### Comparaison des configurations du cluster**



| Indicateur | 1 Worker | 3 Workers |

|---|---:|---:|

| Transactions | 197 677 | 197 677 |

| Clients | 12 000 | 12 000 |

| Produits | 800 | 800 |

| K retenu | 4 | 4 |

| Recall@10 ALS A | 0,081404 | 0,081404 |

| Recall@10 ALS B | 0,077534 | 0,077534 |

| Durée totale initiale | 275,315 s | 297,553 s |



**### Accélération et efficacité parallèle**



L'accélération est calculée par :



Speedup = T1 / T3



Soit :



Speedup = 275,315 / 297,553 = 0,9253



L'efficacité parallèle est calculée par :



Efficacité = Speedup / 3 × 100



Soit :



Efficacité = 30,84 %



**### Interprétation**



L'exécution avec trois Workers est plus lente que celle avec un Worker dans les conditions expérimentales étudiées.



L'augmentation du nombre de Workers entraîne une hausse de la durée totale de 8,08 %.



Ce résultat peut s'expliquer par les coûts de coordination, de communication et d'ordonnancement associés à l'exécution distribuée.



Ces facteurs constituent des hypothèses explicatives et ne sont pas isolés expérimentalement.



Les résultats analytiques comparés sont restés identiques.



Une seconde exécution avec un Worker a produit une durée de 277,039 secondes, soit une variation de 0,63 % par rapport à la première mesure.



\## 9. Analyse du plan d'exécution Spark



Le projet comprend une analyse du plan physique d'exécution de la construction RFM.



Les informations générées sont conservées dans :



```text

results/plan\_execution\_rfm.txt

```



Cette analyse permet d'examiner les opérations physiques planifiées par Spark et d'identifier les mécanismes utilisés pour exécuter les transformations.



L'interprétation détaillée du plan est présentée dans la documentation technique du projet.



###### **10. Reproductibilité et limites**



Les principaux mécanismes favorisant la reproductibilité sont :



\- Utilisation d'une image Docker définie.

\- Paramétrage explicite des ressources Spark.

\- Fixation des graines aléatoires pour les algorithmes concernés.

\- Conservation des paramètres ALS.

\- Exportation des résultats au format JSON.

\- Scripts PowerShell d'exécution.

\- Tests d'intégration.



**### Limites**



Les résultats de performance concernent uniquement les configurations testées.



La comparaison repose sur un volume de données limité et deux tailles de cluster.



Les durées mesurées peuvent varier selon les ressources disponibles et la charge de l'environnement d'exécution.



Une évaluation plus complète de la scalabilité nécessiterait plusieurs répétitions expérimentales et des volumes de données supplémentaires.



###### **11. Documentation complémentaire**



Le répertoire `Report/` contient les documents techniques et les rapports produits par les membres du groupe.



Le répertoire `results/` conserve les résultats expérimentaux et les informations nécessaires à leur vérification.



