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


