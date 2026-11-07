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
    """TODO 1 - Lire les trois fichiers avec des schemas explicites.
    Retirer doublons, quantites <= 0, prix <= 0 et identifiants manquants.
    Retourner transactions, clients et catalogue.
    """
    raise NotImplementedError("TODO 1")


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
