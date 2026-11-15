import sys
import time

sys.path.insert(0, "/workspace/src")

from pyspark.sql import SparkSession

from devoir2_spark_P1 import charger_et_nettoyer
from analyse_sql import construire_rfm


def main():

    spark = (
        SparkSession.builder
        .appName("INF6303-Test-Integration-P1-P2")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    debut = time.perf_counter()

    try:
        print("\n===== TEST P1 : CHARGEMENT ET NETTOYAGE =====")

        transactions, clients, catalogue = charger_et_nettoyer(
            spark, 24
        )

        print("Transactions :", transactions.count())
        print("Clients :", clients.count())
        print("Produits :", catalogue.count())

        print("\n===== TEST P2 : CONSTRUCTION RFM =====")

        rfm = construire_rfm(transactions, catalogue)

        print("Nombre de clients RFM :", rfm.count())

        print("\n===== SCHEMA RFM =====")
        rfm.printSchema()

        print("\n===== APERCU RFM =====")
        rfm.show(10, truncate=False)

        print("\n===== PLAN PHYSIQUE SPARK =====")
        rfm.explain(mode="formatted")

        duree = time.perf_counter() - debut

        print("\n===== RESULTAT =====")
        print("TEST_P1_P2_REUSSI")
        print("Duree totale :", round(duree, 3), "secondes")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()