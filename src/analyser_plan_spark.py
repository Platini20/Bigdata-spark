from pyspark.sql import SparkSession

from devoir2_spark import charger_et_nettoyer
from analyse_sql import construire_rfm

spark = (
    SparkSession.builder
    .appName("INF6303-Analyse-Catalyst")
    .master("spark://spark-master:7077")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\n" + "=" * 60)
print("ANALYSE DU PLAN D'EXECUTION SPARK - RFM")
print("=" * 60)

# Chargement des données
transactions, clients, catalogue = charger_et_nettoyer(
    spark, 24
)

# Construction du DataFrame RFM
rfm = construire_rfm(transactions, catalogue)

# Affichage du plan physique
rfm.explain(mode="formatted")

print("\nFIN DE L'ANALYSE CATALYST")

spark.stop()