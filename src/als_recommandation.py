"""Partie C : recommandation ALS implicite + Recall@10 (Devoir 2, INF6303).

Entrée attendue : un DataFrame Spark de transactions NETTOYÉES avec au moins les colonnes
    client_id, produit_id, quantite
Utilisation (dans devoir2_spark.py) :
    from als_recommandation import recommander
    als_result = recommander(transactions, top_k=10)
La configuration (graine, configurations ALS, pourcentage train) est dans als_config.py.
"""
import time

from pyspark.ml.recommendation import ALS
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

from als_config import CONFIGS_ALS, POURCENTAGE_TRAIN, SEED, TOP_K


def _top_n_sans_vus(modele, train, clients_test, n):
    """Top-N par client après retrait des produits déjà vus à l'entraînement."""
    nb_items = train.select("iid").distinct().count()
    max_vus = (train.join(clients_test, "uid").groupBy("uid").count()
               .agg(F.max("count")).first()[0] or 0)
    k = min(n + max_vus, nb_items)      # assez de candidats pour qu'il en reste n après retrait
    recos = (modele.recommendForUserSubset(clients_test, k)
             .select("uid", F.explode("recommendations").alias("rec"))
             .select("uid", F.col("rec.iid").alias("iid"), F.col("rec.rating").alias("score")))
    recos = recos.join(train.select("uid", "iid"), ["uid", "iid"], "left_anti")   # retire les vus
    w = Window.partitionBy("uid").orderBy(F.col("score").desc(), F.col("iid"))
    return recos.withColumn("rang", F.row_number().over(w)).filter(F.col("rang") <= n)


def _recall_at_n(top_n, test):
    """Recall@N = moyenne sur les clients de (hits / nb d'articles pertinents du test).
    Retourne aussi le nombre de clients ayant au moins un hit."""
    pertinents = test.groupBy("uid").agg(F.count("*").alias("nb_pertinents"))
    hits = (test.select("uid", "iid").join(top_n.select("uid", "iid"), ["uid", "iid"], "inner")
            .groupBy("uid").agg(F.count("*").alias("hits")))
    par_client = (pertinents.join(hits, "uid", "left").fillna(0, subset=["hits"])
                  .select((F.col("hits") / F.col("nb_pertinents")).alias("recall"),
                          (F.col("hits") > 0).cast("int").alias("a_hit")))
    r = par_client.agg(F.avg("recall").alias("recall"), F.sum("a_hit").alias("n_hit")).first()
    return float(r["recall"]), int(r["n_hit"])


def _top_n_populaires_sans_vus(train, clients_test, n):
    """Référence sans apprentissage : les N produits les plus achetés (nombre de clients
    distincts dans le train) que le client n'a pas déjà vus. Sert à situer ALS."""
    max_vus = (train.join(clients_test, "uid").groupBy("uid").count()
               .agg(F.max("count")).first()[0] or 0)
    populaires = (train.groupBy("iid").agg(F.count("*").alias("nb_acheteurs"))
                  .orderBy(F.col("nb_acheteurs").desc(), F.col("iid")).limit(n + max_vus))
    recos = (clients_test.crossJoin(broadcast(populaires))
             .join(train.select("uid", "iid"), ["uid", "iid"], "left_anti"))
    w = Window.partitionBy("uid").orderBy(F.col("nb_acheteurs").desc(), F.col("iid"))
    return recos.withColumn("rang", F.row_number().over(w)).filter(F.col("rang") <= n)


def recommander(transactions, top_k=TOP_K):
    """Triplets implicites (client, produit, strength = somme des quantités), séparation
    train/test reproductible sans fuite, ALS implicite pour chaque configuration de
    als_config.CONFIGS_ALS, Top-N sans articles déjà vus, Recall@N, comptes liés au cold start.
    Retourne un dictionnaire sérialisable en JSON."""
    triplets = (transactions.groupBy("client_id", "produit_id")
                .agg(F.sum("quantite").cast("float").alias("strength")))

    # ALS exige des identifiants ENTIERS : on indexe clients et produits (petites tables, broadcast)
    idx_clients = (triplets.select("client_id").distinct()
                   .withColumn("uid", F.row_number().over(Window.orderBy("client_id"))))
    idx_produits = (triplets.select("produit_id").distinct()
                    .withColumn("iid", F.row_number().over(Window.orderBy("produit_id"))))
    triplets = (triplets.join(broadcast(idx_clients), "client_id")
                .join(broadcast(idx_produits), "produit_id")).cache()

    # Séparation train/test par HACHAGE du couple (client, produit) : un couple est soit en train,
    # soit en test (pas de fuite) et le résultat ne dépend ni du nombre de workers ni de l'ordre.
    h = F.expr(f"pmod(hash(client_id, produit_id, {SEED}), 100)")
    train = triplets.filter(h < POURCENTAGE_TRAIN).select("uid", "iid", "strength").cache()
    test_brut = triplets.filter(h >= POURCENTAGE_TRAIN).select("uid", "iid")

    # Cold start : un client/produit absent du train n'a pas de vecteur appris -> non évaluable.
    # On le retire du test (en comptant le retrait) pour mesurer la qualité du modèle.
    test = (test_brut.join(train.select("uid").distinct(), "uid")
            .join(train.select("iid").distinct(), "iid")).cache()
    clients_test = test.select("uid").distinct().cache()

    nb_clients_total = idx_clients.count()
    stats_split = {
        "methode": f"hachage du couple (client_id, produit_id), {POURCENTAGE_TRAIN} % train / "
                   f"{100 - POURCENTAGE_TRAIN} % test",
        "triplets_total": triplets.count(), "triplets_train": train.count(),
        "triplets_test_brut": test_brut.count(), "triplets_test_retenus": test.count(),
        "clients_total": nb_clients_total,
        "clients_sans_historique_train": nb_clients_total - train.select("uid").distinct().count(),
        "clients_evalues": clients_test.count(),
    }
    stats_split["triplets_test_retires_cold_start"] = (
        stats_split["triplets_test_brut"] - stats_split["triplets_test_retenus"])

    configurations = {}
    meilleur_top = None
    for nom, p in CONFIGS_ALS.items():
        debut = time.perf_counter()
        modele = ALS(userCol="uid", itemCol="iid", ratingCol="strength", implicitPrefs=True,
                     rank=p["rank"], maxIter=p["maxIter"], regParam=p["regParam"], alpha=p["alpha"],
                     coldStartStrategy="drop", nonnegative=True, seed=SEED).fit(train)
        top = _top_n_sans_vus(modele, train, clients_test, top_k).cache()
        rappel, n_hit = _recall_at_n(top, test)
        configurations[nom] = {"parametres": p, f"recall_at_{top_k}": round(rappel, 6),
                               "clients_avec_au_moins_un_hit": n_hit,
                               "duree_secondes": round(time.perf_counter() - debut, 3)}
        if meilleur_top is None or rappel > configurations[meilleur_top[0]][f"recall_at_{top_k}"]:
            meilleur_top = (nom, top)

    rappel_pop, n_hit_pop = _recall_at_n(_top_n_populaires_sans_vus(train, clients_test, top_k), test)
    exemple = (meilleur_top[1].join(idx_clients, "uid").join(idx_produits, "iid")
               .select("client_id", "produit_id", "rang", F.round("score", 4).alias("score"))
               .orderBy("client_id", "rang").limit(3 * top_k))
    return {"configurations": configurations, "meilleure_configuration": meilleur_top[0],
            "reference_popularite": {f"recall_at_{top_k}": round(rappel_pop, 6),
                                     "clients_avec_au_moins_un_hit": n_hit_pop},
            "split": stats_split,
            "exemple_top10_meilleure_configuration": [r.asDict() for r in exemple.collect()]}
