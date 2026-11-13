"""Interactions implicites, séparation sans fuite, ALS et Recall@10."""

from __future__ import annotations

import time

from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from configuration import CONFIGURATIONS_ALS, GRAINE


def construire_interactions(transactions: DataFrame) -> DataFrame:
    """Agrège les achats répétés : strength est la quantité totale, strictement positive."""
    interactions = transactions.groupBy("client_id", "produit_id").agg(
        F.sum("quantite").alias("strength")
    )
    invalides = interactions.filter(F.col("strength").isNull() | (F.col("strength") <= 0) | F.isnan("strength"))
    if invalides.limit(1).count():
        raise ValueError("Des interactions non positives ont survécu au nettoyage.")
    return interactions


def encoder_identifiants(interactions: DataFrame) -> tuple[DataFrame, dict[str, DataFrame], dict]:
    """Construit un index dense et ne l'applique que si l'identifiant source a des trous.

    Les tables de correspondance restent disponibles pour revenir aux identifiants métier.
    """
    def correspondance(df: DataFrame, colonne: str, index: str) -> DataFrame:
        return df.select(colonne).distinct().withColumn(
            index,
            F.row_number().over(Window.orderBy(F.col(colonne))) - 1,
        )

    clients = correspondance(interactions, "client_id", "client_idx").cache()
    produits = correspondance(interactions, "produit_id", "produit_idx").cache()
    ecart_clients = int(clients.filter(F.col("client_id") != F.col("client_idx")).count())
    ecart_produits = int(produits.filter(F.col("produit_id") != F.col("produit_idx")).count())
    strategie = "rang_dense" if ecart_clients or ecart_produits else "identite"
    codees = (
        interactions.join(clients, "client_id")
        .join(produits, "produit_id")
        .select(
            F.col("client_idx").cast("int").alias("client_id"),
            F.col("produit_idx").cast("int").alias("produit_id"),
            "strength",
        )
    )
    meta = {
        "strategie": strategie,
        "clients_non_identiques_au_rang": ecart_clients,
        "produits_non_identiques_au_rang": ecart_produits,
        "clients": int(clients.count()),
        "produits": int(produits.count()),
        "note": (
            "Chaque identifiant source est comparé à son rang dense. "
            "S'ils coïncident, ALS reçoit l'identifiant métier. "
            "Sinon, le rang dense est utilisé et la correspondance permet de revenir au code d'origine. "
            "Les deux colonnes ne sont jamais interchangeées dans une même table."
        ),
    }
    return codees, {"clients": clients, "produits": produits}, meta


def identifiants_metier(df: DataFrame, correspondances: dict[str, DataFrame]) -> DataFrame:
    """Remplace les index ALS par les identifiants d'origine."""
    clients = correspondances["clients"].select(
        F.col("client_idx").alias("client_id"),
        F.col("client_id").alias("client_source"),
    )
    produits = correspondances["produits"].select(
        F.col("produit_idx").alias("produit_id"),
        F.col("produit_id").alias("produit_source"),
    )
    return (
        df.join(clients, "client_id", "inner")
        .join(produits, "produit_id", "inner")
        .drop("client_id", "produit_id")
        .withColumnRenamed("client_source", "client_id")
        .withColumnRenamed("produit_source", "produit_id")
    )


def separer_entrainement_test(interactions: DataFrame, fraction_test: float = 0.2) -> tuple[DataFrame, DataFrame, dict]:
    """Réserve des produits par client, sans retirer son dernier achat d'entraînement.

    Le rang repose sur xxhash64(client, produit, graine), donc sur une clé stable.
    Un client qui n'a qu'un produit reste entièrement dans l'entraînement :
    il n'a pas de vérité de test et ne sera pas inclus dans le Recall@10.
    """
    fenetre = Window.partitionBy("client_id").orderBy(
        F.xxhash64(F.col("client_id"), F.col("produit_id"), F.lit(GRAINE)),
        F.col("produit_id"),
    )
    classe = interactions.withColumn("rang", F.row_number().over(fenetre)).withColumn(
        "n_produits", F.count("*").over(Window.partitionBy("client_id"))
    )
    n_test = F.least(
        F.greatest(F.floor(F.col("n_produits") * F.lit(fraction_test)).cast("int"), F.lit(1)),
        F.col("n_produits") - F.lit(1),
    )
    classe = classe.withColumn(
        "est_test",
        (F.col("n_produits") >= 2) & (F.col("rang") <= n_test),
    )
    test = classe.filter(F.col("est_test")).select("client_id", "produit_id", "strength")
    train = classe.filter(~F.col("est_test")).select("client_id", "produit_id", "strength")

    chevauchement = int(
        train.join(test, ["client_id", "produit_id"], "inner").count()
    )
    if chevauchement:
        raise RuntimeError("Fuite détectée : une interaction est à la fois en entraînement et en test.")

    clients_test = test.select("client_id").distinct()
    clients_test_sans_train = int(
        clients_test.join(train.select("client_id").distinct(), "client_id", "left_anti").count()
    )
    produits_test_sans_train = int(
        test.select("produit_id").distinct().join(
            train.select("produit_id").distinct(), "produit_id", "left_anti"
        ).count()
    )
    meta = {
        "fraction_visee": fraction_test,
        "graine": GRAINE,
        "interactions_entrainement": int(train.count()),
        "interactions_test": int(test.count()),
        "chevauchement": chevauchement,
        "clients_evalues": int(clients_test.count()),
        "clients_test_absents_entrainement": clients_test_sans_train,
        "produits_test_absents_entrainement": produits_test_sans_train,
        "clients_un_seul_produit_non_evalues": int(
            classe.filter(F.col("n_produits") == 1).select("client_id").distinct().count()
        ),
        "strategie": (
            "Pour chaque client ayant au moins deux produits, environ 20 % des produits "
            "(au moins un, au plus n-1) vont au test selon un rang haché. "
            "Le Recall@10 ne compare jamais une recommandation à une interaction d'entraînement."
        ),
    }
    return train.cache(), test.cache(), meta


def filtrer_deja_vus(recommandations: DataFrame, entrainement: DataFrame, k: int = 10) -> DataFrame:
    """Retire les produits vus à l'entraînement puis conserve les k meilleurs scores."""
    vus = entrainement.select("client_id", "produit_id")
    inedits = recommandations.join(vus, ["client_id", "produit_id"], "left_anti")
    fenetre = Window.partitionBy("client_id").orderBy(F.desc("score"), F.asc("produit_id"))
    return (
        inedits.withColumn("rang", F.row_number().over(fenetre))
        .filter(F.col("rang") <= k)
        .select("client_id", "produit_id", "score", "rang")
    )


def rappel_at_k(recommandes: list, pertinents: list, k: int = 10) -> float | None:
    """Recall@k = |intersection| / |pertinents|. None si aucun produit pertinent."""
    verite = list(dict.fromkeys(pertinents))
    if not verite:
        return None
    retenus = list(dict.fromkeys(recommandes))[:k]
    return len(set(retenus) & set(verite)) / len(verite)


def _recall_spark(top: DataFrame, test: DataFrame, k: int) -> dict:
    pertinents = test.groupBy("client_id").agg(F.collect_set("produit_id").alias("pertinents"))
    proposes = (
        top.groupBy("client_id")
        .agg(F.sort_array(F.collect_list(F.struct("rang", "produit_id"))).alias("liste"))
        .select("client_id", F.expr("transform(liste, x -> x.produit_id)").alias("recommandes"))
    )
    compare = pertinents.join(proposes, "client_id", "left").withColumn(
        "recommandes",
        F.coalesce(F.col("recommandes"), F.expr("array()")),
    )
    # slice SQL est 1-indexé et borne la liste même si elle contient moins de k produits.
    compare = compare.withColumn("retenus", F.expr(f"slice(recommandes, 1, {k})")).withColumn(
        "hits", F.size(F.array_intersect(F.col("retenus"), F.col("pertinents")))
    ).withColumn("n_pertinents", F.size(F.col("pertinents"))).withColumn(
        "recall", F.col("hits") / F.col("n_pertinents")
    )
    resume = compare.agg(
        F.count("*").alias("clients"),
        F.round(F.avg("recall"), 6).alias("recall_at_10"),
        F.sum((F.size("retenus") < k).cast("int")).alias("clients_moins_de_k"),
        F.min(F.size("retenus")).alias("minimum_recommandations"),
    ).collect()[0]
    return {
        "clients_evalues": int(resume["clients"]),
        "recall_at_10": float(resume["recall_at_10"] or 0.0),
        "clients_avec_moins_de_k_recommandations": int(resume["clients_moins_de_k"] or 0),
        "minimum_recommandations": int(resume["minimum_recommandations"] or 0),
        "definition": (
            "Moyenne, sur les clients qui ont au moins un produit dans le test, "
            "de |Top-10 hors produits d'entraînement ∩ test| / |test du client|. "
            "Ce n'est pas un taux de classification : l'ordre des dix produits compte, "
            "et un produit pertinent absent du Top-10 est un raté même si le modèle lui donnait un score."
        ),
    }


def _entrainer(train: DataFrame, config: dict):
    als = ALS(
        userCol="client_id",
        itemCol="produit_id",
        ratingCol="strength",
        implicitPrefs=True,
        coldStartStrategy="drop",
        seed=GRAINE,
        rank=config["rank"],
        maxIter=config["maxIter"],
        regParam=config["regParam"],
        alpha=config["alpha"],
        nonnegative=False,
    )
    return als.fit(train)


def recommander(transactions: DataFrame, catalogue: DataFrame | None = None, top_k: int = 10) -> dict:
    """Entraîne deux ALS implicites et compare leur Recall@10 hors produits déjà vus."""
    debut_preparation = time.perf_counter()
    interactions = construire_interactions(transactions)
    codees, correspondances, encodage = encoder_identifiants(interactions)
    colonnes_als = codees.select("client_id", "produit_id", "strength")
    train, test, separation = separer_entrainement_test(colonnes_als)
    train.count()
    test.count()
    duree_preparation = time.perf_counter() - debut_preparation

    max_vus = train.groupBy("client_id").count().agg(F.max("count")).collect()[0][0] or 0
    n_produits = int(train.select("produit_id").distinct().count())
    # Demander assez de candidats pour qu'il reste top_k produits après retrait des vus.
    n_candidats = int(min(max(n_produits, top_k), int(max_vus) + top_k))

    configurations = []
    echantillon = None
    top_precedent = None
    duree_entrainement = 0.0
    duree_recommandations = 0.0
    for config in CONFIGURATIONS_ALS:
        debut = time.perf_counter()
        modele = _entrainer(train, config)
        duree_fit = time.perf_counter() - debut
        duree_entrainement += duree_fit

        debut_reco = time.perf_counter()
        brutes = modele.recommendForAllUsers(n_candidats)
        explodees = brutes.select(
            "client_id", F.posexplode("recommendations").alias("position", "rec")
        ).select(
            "client_id",
            F.col("rec.produit_id").alias("produit_id"),
            F.col("rec.rating").alias("score"),
        )
        if top_precedent is not None:
            top_precedent.unpersist()
        top = filtrer_deja_vus(explodees, train, top_k).cache()
        top_precedent = top
        top.count()
        metrique = _recall_spark(top, test, top_k)
        duree_reco = time.perf_counter() - debut_reco
        duree_recommandations += duree_reco
        configurations.append(
            {
                **config,
                "coldStartStrategy": "drop",
                "seed": GRAINE,
                "nonnegative": False,
                "candidats_demandes": n_candidats,
                "duree_entrainement_secondes": round(duree_fit, 3),
                "duree_recommandations_secondes": round(duree_reco, 3),
                "recall": metrique,
            }
        )
        if echantillon is None or metrique["recall_at_10"] >= echantillon["recall_at_10"]:
            # Matérialisé tout de suite : le cache du Top-k est libéré à la configuration suivante.
            echantillon = {
                "configuration": config["nom"],
                "recall_at_10": metrique["recall_at_10"],
                "clients": _echantillon(identifiants_metier(top, correspondances), catalogue),
            }

    return {
        "encodage": encodage,
        "separation": separation,
        "cold_start": {
            "clients_sans_interaction_apres_nettoyage": int(
                transactions.select("client_id").distinct().count() - encodage["clients"]
            ),
            "clients_du_test_absents_de_l_entrainement": separation["clients_test_absents_entrainement"],
            "produits_du_test_absents_de_l_entrainement": separation["produits_test_absents_entrainement"],
            "clients_non_evalues_car_un_seul_produit": separation["clients_un_seul_produit_non_evalues"],
            "traitement": (
                "ALS ne recommande pas un client absent de l'entraînement (coldStartStrategy=drop). "
                "Dans cette séparation, chaque client évalué conserve au moins une interaction d'entraînement. "
                "Un produit présent seulement dans le test ne peut pas être recommandé. "
                "Les clients qui n'ont qu'un produit ne contribuent pas au Recall@10."
            ),
        },
        "parametre_modifie": "alpha",
        "configurations": configurations,
        "echantillon_top10": {
            "configuration": echantillon["configuration"],
            "clients": echantillon["clients"],
        },
        "duree_preparation_interactions_secondes": round(duree_preparation, 3),
        "duree_entrainement_secondes": round(duree_entrainement, 3),
        "duree_recommandations_secondes": round(duree_recommandations, 3),
    }


def _echantillon(top: DataFrame, catalogue: DataFrame | None, n_clients: int = 3) -> list[dict]:
    clients = [r["client_id"] for r in top.select("client_id").distinct().orderBy("client_id").limit(n_clients).collect()]
    if not clients:
        return []
    extrait = top.filter(F.col("client_id").isin(clients))
    if catalogue is not None:
        extrait = extrait.join(catalogue, "produit_id", "left")
    colonnes = ["client_id", "rang", "produit_id", "score"]
    if "categorie" in extrait.columns:
        colonnes.extend(["categorie", "marque"])
    lignes = []
    for row in extrait.select(*colonnes).orderBy("client_id", "rang").collect():
        item = {
            "client_id": int(row["client_id"]),
            "rang": int(row["rang"]),
            "produit_id": int(row["produit_id"]),
            "score": round(float(row["score"]), 6),
        }
        if "categorie" in colonnes:
            item["categorie"] = row["categorie"]
            item["marque"] = row["marque"]
        lignes.append(item)
    return lignes
