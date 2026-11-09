"""Test autonome P1, sans exécuter les fonctions TODO 2/3/4."""
from pyspark.sql import SparkSession
from devoir2_spark_P1 import charger_et_nettoyer

spark = (SparkSession.builder.appName("INF6303-Test-P1")
         .config("spark.sql.shuffle.partitions", "24").getOrCreate())
spark.sparkContext.setLogLevel("WARN")
try:
    transactions, clients, catalogue = charger_et_nettoyer(spark, 24)
    assert transactions.count() == 197677, "Vérifier l'audit P1 et la politique de nettoyage"
    assert clients.count() == 12000
    assert catalogue.count() == 800
    assert transactions.rdd.getNumPartitions() == 24
    print("TEST P1 OK : 3 tables disponibles et comptages conformes")
    transactions.printSchema()
finally:
    spark.stop()
