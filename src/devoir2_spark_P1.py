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

    # Les doublons strictement identiques sont supprimés en premier.
    t_uniques = t_brutes.dropDuplicates()
    c_uniques = c_bruts.dropDuplicates()
    p_uniques = p_brut.dropDuplicates()

    # Puis élimination des lignes invalides et des identifiants manquants.
    # On garde les ID égaux à 0, car ce sont des ID valides du jeu fourni.
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

    # Vérifier également les références aux tables clients/catalogue.
    # La table conservée reste la source transactionnelle pour P2 et P4.
    # Les jointures de contrôle ne modifient pas les colonnes retournées.
    t_rejetees_client = t_valides.join(
        c_valides.select("client_id"), "client_id", "left_anti")
    t_rejetees_produit = t_valides.join(
        p_valides.select("produit_id"), "produit_id", "left_anti")

    # Conservation de la même partition logique pour 1 et 3 workers.
    transactions = t_valides.repartition(partitions, "client_id")
    clients = c_valides
    catalogue = p_valides

    # Contrôles Spark (et non simple copie des effectifs du manifeste).
    audit = {
        "transactions_brutes": t_brutes.count(),
        "transactions_doublons_exacts": t_brutes.count() - t_uniques.count(),
        "transactions_quantite_invalide_apres_dedoublonnage":
            t_uniques.filter(F.col("quantite").isNull() | (F.col("quantite") <= 0)).count(),
        "transactions_prix_invalide_apres_dedoublonnage":
            t_uniques.filter(F.col("prix_unitaire").isNull() |
                             (F.col("prix_unitaire") <= 0) |
                             F.isnan("prix_unitaire")).count(),
        "transactions_nettoyees": transactions.count(),
        "clients_bruts": c_bruts.count(),
        "clients_nettoyes": clients.count(),
        "catalogue_brut": p_brut.count(),
        "catalogue_nettoye": catalogue.count(),
        "transactions_sans_client_correspondant": t_rejetees_client.count(),
        "transactions_sans_produit_correspondant": t_rejetees_produit.count(),
        "partitions_transactions": transactions.rdd.getNumPartitions(),
    }
    print("AUDIT_P1=" + json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return transactions, clients, catalogue


def construire_rfm(transactions, catalogue):
    """TODO 2 - Jointure et agregations Spark SQL/DataFrame.
    Produire: client_id, recency_days, frequency, amount, product_diversity.
    Date de reference imposee: 2026-11-01.
    """
    raise NotImplementedError("TODO 2")


def segmenter(rfm):
    """TODO 3 - Assembler, standardiser, comparer K=2..6 avec silhouette,
    choisir K et profiler les clusters sur les variables originales.
    Retourner un dictionnaire serialisable.
    """
    raise NotImplementedError("TODO 3")


def recommander(transactions, top_k=10):
    """TODO 4 - Construire les triplets implicites, separer entrainement/test,
    entrainer ALS, retirer les articles deja vus et calculer Recall@10.
    Gerer le cold start et retourner un dictionnaire serialisable.
    """
    raise NotImplementedError("TODO 4")


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
