import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "/workspace/src")

from pyspark.sql import SparkSession

from devoir2_spark_P1 import charger_et_nettoyer
from analyse_sql import construire_rfm
from segmentation import segmenter


def main():

    spark = (
        SparkSession.builder
        .appName("INF6303-Test-Integration-P1-P2-P3")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    debut = time.perf_counter()

    try:

        print("\n===== ETAPE P1 : PREPARATION =====")

        transactions, clients, catalogue = charger_et_nettoyer(
            spark, 24
        )

        print("Transactions :", transactions.count())
        print("Clients :", clients.count())
        print("Catalogue :", catalogue.count())

        print("\n===== ETAPE P2 : CONSTRUCTION RFM =====")

        rfm = construire_rfm(transactions, catalogue)

        print("Nombre de clients RFM :", rfm.count())

        print("\n===== ETAPE P3 : SEGMENTATION K-MEANS =====")

        resultat_kmeans = segmenter(rfm)

        print("\n===== RESULTATS K-MEANS =====")

        print(
            json.dumps(
                resultat_kmeans,
                ensure_ascii=False,
                indent=2
            )
        )

        print("\n===== ENREGISTREMENT DES RESULTATS =====")

        chemin = Path(
            "/workspace/results/test_integration_p1_p2_p3.json"
        )

        chemin.parent.mkdir(parents=True, exist_ok=True)

        chemin.write_text(
            json.dumps(
                resultat_kmeans,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print("Fichier :", chemin)

        duree = time.perf_counter() - debut

        print("\n===== BILAN =====")
        print("TEST_P1_P2_P3_REUSSI")
        print("Duree totale :", round(duree, 3), "secondes")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()