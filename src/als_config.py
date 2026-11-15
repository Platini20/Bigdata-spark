"""Configuration de la partie C : recommandation ALS (Devoir 2, INF6303).

Tous les réglages de l'ALS sont ici. Pour comparer deux configurations,
on ne modifie qu'un ensemble limité de paramètres (ici seulement `rank`).
"""

SEED = 42                # graine unique : séparation train/test et ALS reproductibles
TOP_K = 10               # taille de la liste de recommandations (Top-10 -> Recall@10)
POURCENTAGE_TRAIN = 80   # 80 % des triplets pour l'entraînement, 20 % pour le test

# Deux configurations ALS (mode implicite). Seul `rank` change entre A et B :
#   rank     : nombre de facteurs latents (taille des vecteurs client et produit)
#   maxIter  : nombre d'itérations de l'algorithme
#   regParam : régularisation (évite le surapprentissage)
#   alpha    : poids de confiance accordé aux achats observés (feedback implicite)
CONFIGS_ALS = {
    "A_rank10": {"rank": 10, "maxIter": 10, "regParam": 0.1, "alpha": 40.0},
    "B_rank20": {"rank": 20, "maxIter": 10, "regParam": 0.1, "alpha": 40.0},
}
