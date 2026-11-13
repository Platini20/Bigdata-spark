from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--partitions", type=int, default=24)
    return p.parse_args()
    
    


def charger_et_nettoyer(spark, partitions):
    """P1 - Lecture, validation, nettoyage et contrôle des trois sources.

    Contrat de sortie :
      transactions(transaction_id, client_id, produit_id,
                   date_transaction, quantite, prix_unitaire)
      clients(client_id, region, date_inscription)
      catalogue(produit_id, categorie, marque)
    Les identifiants client et produit restent numériques pour les jointures
    et pour ALS. Les fichiers CSV sources ne sont jamais modifiés.
    """
    from pyspark.sql.types import (StructType, StructField, StringType,
                                   IntegerType, DoubleType, DateType)

    if partitions <= 0:
        raise ValueError("Le nombre de partitions doit etre strictement positif")

    transactions_schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("client_id", IntegerType(), True),
        StructField("produit_id", IntegerType(), True),
        StructField("date_transaction", DateType(), True),
        StructField("quantite", IntegerType(), True),
        StructField("prix_unitaire", DoubleType(), True),
    ])
    clients_schema = StructType([
        StructField("client_id", IntegerType(), True),
        StructField("region", StringType(), True),
        StructField("date_inscription", DateType(), True),
    ])
    catalogue_schema = StructType([
        StructField("produit_id", IntegerType(), True),
        StructField("categorie", StringType(), True),
        StructField("marque", StringType(), True),
    ])

    def lire_csv(nom, schema):
        return (spark.read.option("header", "true")
                .option("mode", "FAILFAST")
                .option("dateFormat", "yyyy-MM-dd")
                .schema(schema).csv(f"/workspace/data/{nom}"))

    t_brutes = lire_csv("transactions.csv.gz", transactions_schema)
    c_bruts = lire_csv("clients.csv.gz", clients_schema)
    p_brut = lire_csv("catalogue.csv", catalogue_schema)

    # Suppression des doublons exacts.
    t_uniques = t_brutes.dropDuplicates()
    c_uniques = c_bruts.dropDuplicates()
    p_uniques = p_brut.dropDuplicates()

    # Elimination des lignes invalides.
    # Les identifiants egaux a zero sont conserves.
    ok_t = (F.col("transaction_id").isNotNull() &
            (F.trim(F.col("transaction_id")) != "") &
            F.col("client_id").isNotNull() &
            F.col("produit_id").isNotNull() &
            F.col("date_transaction").isNotNull() &
            (F.col("quantite") > 0) &
            (F.col("prix_unitaire") > 0) &
            (~F.isnan(F.col("prix_unitaire"))))

    ok_c = (F.col("client_id").isNotNull() &
            F.col("region").isNotNull() &
            (F.trim(F.col("region")) != "") &
            F.col("date_inscription").isNotNull())

    ok_p = (F.col("produit_id").isNotNull() &
            F.col("categorie").isNotNull() &
            (F.trim(F.col("categorie")) != "") &
            F.col("marque").isNotNull() &
            (F.trim(F.col("marque")) != ""))

    t_valides = t_uniques.filter(ok_t)
    c_valides = c_uniques.filter(ok_c)
    p_valides = p_uniques.filter(ok_p)

    # Controle des references clients et produits.
    t_rejetees_client = t_valides.join(
        c_valides.select("client_id"), "client_id", "left_anti")

    t_rejetees_produit = t_valides.join(
        p_valides.select("produit_id"), "produit_id", "left_anti")

    # Conservation des 24 partitions imposees.
    transactions = t_valides.repartition(partitions, "client_id")
    clients = c_valides
    catalogue = p_valides

    # Audit du nettoyage avec Spark.
    audit = {
        "transactions_brutes": t_brutes.count(),

        "transactions_doublons_exacts":
            t_brutes.count() - t_uniques.count(),

        "transactions_quantite_invalide_apres_dedoublonnage":
            t_uniques.filter(
                F.col("quantite").isNull() |
                (F.col("quantite") <= 0)
            ).count(),

        "transactions_prix_invalide_apres_dedoublonnage":
            t_uniques.filter(
                F.col("prix_unitaire").isNull() |
                (F.col("prix_unitaire") <= 0) |
                F.isnan("prix_unitaire")
            ).count(),

        "transactions_nettoyees": transactions.count(),
        "clients_bruts": c_bruts.count(),
        "clients_nettoyes": clients.count(),
        "catalogue_brut": p_brut.count(),
        "catalogue_nettoye": catalogue.count(),

        "transactions_sans_client_correspondant":
            t_rejetees_client.count(),

        "transactions_sans_produit_correspondant":
            t_rejetees_produit.count(),

        "partitions_transactions":
            transactions.rdd.getNumPartitions(),
    }

    print("AUDIT_P1=" + json.dumps(
        audit, ensure_ascii=False, sort_keys=True
    ))

    return transactions, clients, catalogue





"""
def construire_rfm(transactions, catalogue):
    """TODO 2 - Jointure et agregations Spark SQL/DataFrame.
    Produire: client_id, recency_days, frequency, amount, product_diversity.
    Date de reference imposee: 2026-11-01.
    """
    raise NotImplementedError("TODO 2")
"""

def segmenter(rfm):
    """TODO 3 - Assembler, standardiser, comparer K=2..6 avec silhouette,
    choisir K et profiler les clusters sur les variables originales.
    Retourner un dictionnaire serialisable.
    """
    raise NotImplementedError("TODO 3")


# ---------------------------------------------------------------------------
# PARTIE C - ALS implicite + Recall@10
# ---------------------------------------------------------------------------
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


def recommander(transactions, top_k=10):
    """Triplets implicites (client, produit, strength=somme des quantités), séparation
    train/test reproductible sans fuite, ALS implicite pour 2 configurations,
    Top-10 sans articles déjà vus, Recall@10, comptes liés au cold start."""
    triplets = (transactions.groupBy("client_id", "produit_id")
                .agg(F.sum("quantite").cast("float").alias("strength")))

    # ALS exige des identifiants ENTIERS : on indexe clients et produits (petites tables, broadcast)
    idx_clients = (triplets.select("client_id").distinct()
                   .withColumn("uid", F.row_number().over(Window.orderBy("client_id"))))
    idx_produits = (triplets.select("produit_id").distinct()
                    .withColumn("iid", F.row_number().over(Window.orderBy("produit_id"))))
    triplets = (triplets.join(broadcast(idx_clients), "client_id")
                .join(broadcast(idx_produits), "produit_id")).cache()

    # Séparation 80/20 par HACHAGE du couple (client, produit) : un couple est soit en train,
    # soit en test (pas de fuite) et le résultat ne dépend ni du nombre de workers ni de l'ordre.
    h = F.expr(f"pmod(hash(client_id, produit_id, {SEED}), 100)")
    train = triplets.filter(h < 80).select("uid", "iid", "strength").cache()
    test_brut = triplets.filter(h >= 80).select("uid", "iid")

    # Cold start : un client/produit absent du train n'a pas de vecteur appris -> non évaluable.
    # On le retire du test (en comptant le retrait) pour mesurer la qualité du modèle.
    test = (test_brut.join(train.select("uid").distinct(), "uid")
            .join(train.select("iid").distinct(), "iid")).cache()
    clients_test = test.select("uid").distinct().cache()

    nb_clients_total = idx_clients.count()
    stats_split = {
        "methode": "hachage du couple (client_id, produit_id), 80 % train / 20 % test",
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


def main():
    args = arguments()
    spark = (SparkSession.builder.appName(f"INF6303-Devoir2-{args.workers}-workers")
             .config("spark.sql.shuffle.partitions", str(args.partitions)).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    started = time.perf_counter()
    transactions, clients, catalogue = charger_et_nettoyer(spark, args.partitions)
    rfm = construire_rfm(transactions, catalogue)
    # TODO 5 - Afficher explain("formatted") sur au moins un plan pertinent.
    kmeans_result = segmenter(rfm)
    als_result = recommander(transactions, top_k=10)
    result = {
        "environment": {"master": spark.sparkContext.master, "spark_version": spark.version,
                        "workers": args.workers, "default_parallelism": spark.sparkContext.defaultParallelism,
                        "shuffle_partitions": args.partitions},
        "counts": {"transactions": transactions.count(), "clients": clients.count(), "catalogue": catalogue.count()},
        "kmeans": kmeans_result,
        "als": als_result,
        "total_duration_seconds": round(time.perf_counter() - started, 3),
    }
    suffix = "worker" if args.workers == 1 else "workers"
    output = Path(f"/workspace/results/resultat_{args.workers}_{suffix}.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
