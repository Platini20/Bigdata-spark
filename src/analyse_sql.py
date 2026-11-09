from pyspark.sql import functions as F
from pyspark.sql import Window
def joindre_catalogue(transactions, catalogue):
    return transactions.join(
        catalogue.select("produit_id", "categorie"),
        on="produit_id",
        how="left",
    )
    
def construire_rfm(transactions, catalogue):
    date_reference = F.lit("2026-11-01").cast("date")

    achats = joindre_catalogue(transactions, catalogue).withColumn(
        "montant_ligne", F.col("quantite") * F.col("prix_unitaire")
    )

    return achats.groupBy("client_id").agg(
        F.datediff(date_reference, F.max("date_transaction")).alias("recency_days"),
        F.countDistinct("transaction_id").alias("frequency"),
        F.sum("montant_ligne").alias("amount"),
        F.countDistinct("categorie").alias("product_diversity"),
    )
    
def produit_dominant_par_region(transactions, clients):
    ventes_par_produit = (
        transactions
        .join(clients.select("client_id", "region"), on="client_id", how="left")
        .filter(F.col("region").isNotNull())
        .withColumn("montant_ligne", F.col("quantite") * F.col("prix_unitaire"))
        .groupBy("region", "produit_id")
        .agg(F.sum("montant_ligne").alias("montant_total"))
    )

    classement = Window.partitionBy("region").orderBy(
        F.desc("montant_total"),
        F.asc("produit_id"),
    )

    return (
        ventes_par_produit
        .withColumn("rang", F.row_number().over(classement))
        .filter(F.col("rang") == 1)
        .drop("rang")
    )