"""P3 - Partie B : segmentation K-Means des clients à partir du RFM étendu.

Module importé uniquement par le driver : aucune UDF Python, seulement des
opérations DataFrame et MLlib exécutées par la JVM. Il n'a donc pas besoin
d'être envoyé aux exécuteurs (pas de --py-files).

Contrat d'entrée (sortie de analyse_sql.construire_rfm) :
  rfm(client_id, recency_days, frequency, amount, product_diversity)
"""
from __future__ import annotations

import json
import math
import statistics
import time

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import functions as F

FEATURES = ["recency_days", "frequency", "amount", "product_diversity"]

# Paramètres de la règle de sélection de K (fixés par l'équipe).
ARI_MIN = 0.80
PART_MIN_CLUSTER = 0.02
TOLERANCE_SILHOUETTE = 0.01
MAX_ITER = 50

# K retenu après revue humaine des profils, utilisé par défaut pour k_force.
# La règle automatique retient K=3, mais son cluster intermédiaire est bimodal.
K_APRES_REVUE = 4
MOTIF_REVUE = (
    "Revue des quartiles (seed 42, K=3) : le cluster intermédiaire (5 946 "
    "clients) est bimodal en fréquence, 2 952 clients à 7-8 achats et 2 994 "
    "à 16-18 achats, un seul entre les deux. Aucun nom métier unique ne le "
    "décrit. K=4 le coupe en deux (95,9 % et 93,6 % de chaque groupe dans "
    "un cluster distinct) et garde les champions (99,8 %) et les inactifs "
    "(98,8 %) presque identiques. K=4 reste candidat selon la règle (ARI "
    "moyen >= 0,80, plus petit cluster >= 2 %)."
)

# Seuil (en écarts-types) utilisé pour nommer les clusters. Convention de
# l'équipe, à revoir avec les profils : ce n'est pas une valeur de référence.
SEUIL_Z = 0.5

REGLE_SELECTION = (
    "1) Candidats : ARI moyen (seed de référence vs autres seeds) >= 0,80 "
    "et plus petit cluster >= 2 % des clients pour tous les seeds. "
    "2) Parmi les candidats, on prend la silhouette moyenne la plus haute; "
    "si un K plus petit est à moins de 0,01 de cette silhouette, on prend "
    "le plus petit K, plus facile à interpréter. "
    "3) Sans candidat, on prend la silhouette moyenne la plus haute. "
    "4) k_force impose un K après revue humaine des profils."
)

POURQUOI_PAS_LE_COUT_SEUL = (
    "Le coût (somme des distances au carré aux centres) diminue presque "
    "toujours quand K augmente, jusqu'à 0 avec un cluster par client : il "
    "favorise mécaniquement le plus grand K. Il faut donc le combiner à une "
    "mesure de séparation (silhouette), à la stabilité entre seeds et à "
    "l'interprétabilité métier."
)


def _r(x, n=4):
    """Arrondit un nombre en float Python natif (None conservé)."""
    return None if x is None else round(float(x), n)


def _ari(contingence):
    """Indice de Rand ajusté calculé en Python pur sur la table de contingence.

    contingence : liste de tuples (pred_a, pred_b, effectif).
    """
    lignes, colonnes = {}, {}
    somme_nij = 0
    n = 0
    for a, b, c in contingence:
        lignes[a] = lignes.get(a, 0) + c
        colonnes[b] = colonnes.get(b, 0) + c
        somme_nij += math.comb(c, 2)
        n += c
    somme_a = sum(math.comb(v, 2) for v in lignes.values())
    somme_b = sum(math.comb(v, 2) for v in colonnes.values())
    total = math.comb(n, 2)
    attendu = somme_a * somme_b / total if total else 0.0
    maximum = (somme_a + somme_b) / 2
    if maximum == attendu:
        # Partitions triviales identiques (ex. un seul cluster) : accord parfait.
        return 1.0
    return (somme_nij - attendu) / (maximum - attendu)


def _choisir_k(comparaison):
    """Applique REGLE_SELECTION; retourne (k, justification, ensemble des K candidats)."""
    candidats = [c for c in comparaison
                 if c["ari_moyen"] >= ARI_MIN and c["part_min_cluster"] >= PART_MIN_CLUSTER]
    if candidats:
        meilleure = max(c["silhouette_moyenne"] for c in candidats)
        retenu = min((c for c in candidats
                      if meilleure - c["silhouette_moyenne"] < TOLERANCE_SILHOUETTE),
                     key=lambda c: c["k"])
        justification = (
            f"Candidats stables et équilibrés : {[c['k'] for c in candidats]}. "
            f"Meilleure silhouette moyenne parmi eux : {meilleure:.4f}. "
            f"K={retenu['k']} retenu (silhouette moyenne "
            f"{retenu['silhouette_moyenne']:.4f}, ARI moyen {retenu['ari_moyen']:.4f}, "
            f"plus petit cluster {retenu['part_min_cluster']:.2%})."
        )
    else:
        retenu = max(comparaison, key=lambda c: c["silhouette_moyenne"])
        justification = (
            "Aucun K ne respecte à la fois ARI moyen >= 0,80 et plus petit "
            f"cluster >= 2 % : repli sur la silhouette moyenne maximale, K={retenu['k']} "
            f"({retenu['silhouette_moyenne']:.4f})."
        )
    return retenu["k"], justification, {c["k"] for c in candidats}


def _fr(x, n=2):
    """Formate un nombre avec la virgule décimale française."""
    return f"{x:.{n}f}".replace(".", ",")


def _decision(c, retenu, candidats):
    """Texte court pour la colonne « Décision » du tableau du rapport."""
    if c["k"] == retenu["k"]:
        return "retenu"
    raisons = []
    if c["ari_moyen"] < ARI_MIN:
        raisons.append(f"instable (ARI moyen {_fr(c['ari_moyen'])} < 0,80)")
    if c["part_min_cluster"] < PART_MIN_CLUSTER:
        raisons.append(f"cluster trop petit ({_fr(100 * c['part_min_cluster'], 1)} % < 2 %)")
    if c["k"] in candidats:
        if retenu["silhouette_moyenne"] - c["silhouette_moyenne"] >= TOLERANCE_SILHOUETTE:
            raisons.append(f"candidat, silhouette moyenne inférieure "
                           f"({_fr(c['silhouette_moyenne'], 4)} vs "
                           f"{_fr(retenu['silhouette_moyenne'], 4)})")
        else:
            raisons.append("candidat, silhouette équivalente mais K plus grand")
    return "écarté : " + "; ".join(raisons)


def _nommer(centres_z):
    """Propose un nom métier unique par cluster à partir des centres en z-scores.

    centres_z : {cluster: {variable: z}}. Les règles sont évaluées dans l'ordre;
    la première qui s'applique donne le nom de base.
    """
    s = SEUIL_Z

    def nom_base(z):
        r, f, m, d = (z[v] for v in FEATURES)
        if r <= -s and (f >= s or m >= s):
            return "Clients champions"
        if r >= s:
            return "Clients inactifs / à risque"
        if d >= s:
            return "Explorateurs multi-catégories"
        if f >= s or m >= s:
            return "Clients fidèles à forte valeur"
        if f <= -s and m <= -s:
            return "Acheteurs occasionnels"
        if r <= -s and f > -s:
            return "Clients réguliers"
        if r <= -s:
            return "Clients récents / actifs"
        return "Clients standards"

    libelles = {"recency_days": "récence", "frequency": "fréquence",
                "amount": "montant", "product_diversity": "diversité"}
    noms = {c: nom_base(z) for c, z in centres_z.items()}
    # Unicité : en cas de doublon, on qualifie par la variable la plus marquée.
    for c in sorted(noms):
        if list(noms.values()).count(noms[c]) > 1:
            z = centres_z[c]
            v = max(FEATURES, key=lambda x: abs(z[x]))
            sens = "élevée" if z[v] > 0 else "faible"
            noms[c] = f"{noms[c]} ({libelles[v]} {sens})"
    for c in sorted(noms):
        if list(noms.values()).count(noms[c]) > 1:
            noms[c] = f"{noms[c]} n°{c}"
    return noms


def segmenter(rfm, k_values=(2, 3, 4, 5, 6), seeds=(42, 7, 2026), k_force=K_APRES_REVUE):
    """P3 - Segmentation K-Means sur les quatre variables RFM standardisées.

    Étapes : conversion en double et retrait des nulls, VectorAssembler,
    StandardScaler (moyenne 0, écart-type 1, ajusté une seule fois), puis
    K-Means pour chaque K et chaque seed (maxIter=50).

    Mesures par modèle : coût d'entraînement, silhouette (distance euclidienne
    au carré, sur les variables standardisées) et part du plus petit cluster.
    Stabilité : ARI entre le premier seed (référence) et chacun des autres,
    calculé sur une table de contingence K×K ramenée sur le driver.

    Règle de sélection de K :
      1) Candidats : ARI moyen >= 0,80 et plus petit cluster >= 2 % des
         clients (pour tous les seeds).
      2) Parmi les candidats, silhouette moyenne la plus haute; si un K plus
         petit est à moins de 0,01 de cette silhouette, on prend le plus petit
         K (plus facile à interpréter).
      3) Sans candidat : silhouette moyenne la plus haute.
      4) k_force (doit faire partie de k_values) impose K après revue humaine.

    Par défaut, k_force vaut K_APRES_REVUE (4) : segmenter(rfm) retourne K=4,
    retenu après revue des profils (MOTIF_REVUE). Passer k_force=None
    explicitement applique la règle seule (K=3 sur les données du devoir).
    Le dictionnaire contient toujours k_regle, k_retenu et revue_humaine.

    Le modèle final est celui du K retenu avec le seed de référence. Les
    profils sont calculés sur les variables originales (non standardisées).
    Retourne un dictionnaire sérialisable en JSON (types Python natifs).
    """
    debut = time.perf_counter()
    k_values = sorted(int(k) for k in k_values)
    seeds = [int(s) for s in seeds]
    if len(seeds) < 2:
        raise ValueError("Au moins deux seeds sont nécessaires pour mesurer la stabilité")
    if k_force is not None and int(k_force) not in k_values:
        raise ValueError(f"k_force={k_force} doit faire partie de k_values={k_values}")
    seed_ref = seeds[0]

    # 1. Préparation : conversion en double et retrait des lignes incomplètes.
    typees = rfm.select(*[F.col(c).cast("double").alias(c) for c in FEATURES])
    complet = typees.dropna(subset=FEATURES)
    lignes_totales = typees.count()

    # 2. Assemblage puis standardisation (scaler ajusté une seule fois).
    assembleur = VectorAssembler(inputCols=FEATURES, outputCol="features_brutes")
    scaler = StandardScaler(inputCol="features_brutes", outputCol="features",
                            withMean=True, withStd=True)
    assemble = assembleur.transform(complet)
    modele_scaler = scaler.fit(assemble)
    standardise = (modele_scaler.transform(assemble)
                   .select(*FEATURES, "features").cache())
    try:
        n = standardise.count()
        lignes_retirees = lignes_totales - n

        # 3. Entraînement pour chaque K et chaque seed.
        evaluateur = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                                         metricName="silhouette",
                                         distanceMeasure="squaredEuclidean")
        mesures = {}   # (k, seed) -> (coût, silhouette, part_min)
        modeles = {}   # (k, seed) -> modèle, réutilisé pour l'ARI et le modèle final
        for k in k_values:
            for seed in seeds:
                modele = KMeans(k=k, seed=seed, maxIter=MAX_ITER,
                                featuresCol="features").fit(standardise)
                resume = modele.summary
                silhouette = evaluateur.evaluate(modele.transform(standardise))
                part_min = min(resume.clusterSizes) / n
                mesures[(k, seed)] = (resume.trainingCost, silhouette, part_min)
                modeles[(k, seed)] = modele

        # 4. Stabilité : ARI entre le seed de référence et chacun des autres.
        comparaison = []
        for k in k_values:
            ref = modeles[(k, seed_ref)]
            aris = []
            for seed in seeds[1:]:
                autre = modeles[(k, seed)]
                paires = autre.transform(
                    ref.transform(standardise, {ref.predictionCol: "pred_a"}),
                    {autre.predictionCol: "pred_b"})
                contingence = [(r["pred_a"], r["pred_b"], r["count"])
                               for r in paires.groupBy("pred_a", "pred_b").count().collect()]
                aris.append(_ari(contingence))
            couts = [mesures[(k, s)][0] for s in seeds]
            sils = [mesures[(k, s)][1] for s in seeds]
            comparaison.append({
                "k": k,
                "cout": _r(mesures[(k, seed_ref)][0]),
                "cout_moyen": _r(statistics.mean(couts)),
                "silhouette": _r(mesures[(k, seed_ref)][1]),
                "silhouette_moyenne": _r(statistics.mean(sils)),
                "silhouette_ecart_type": _r(statistics.stdev(sils)),
                "ari_moyen": _r(statistics.mean(aris)),
                "part_min_cluster": _r(min(mesures[(k, s)][2] for s in seeds)),
            })

        # 5. Choix de K.
        k_regle, justification, candidats = _choisir_k(comparaison)
        k_retenu = k_regle if k_force is None else int(k_force)
        revue = None
        if k_force is not None:
            revue = {
                "k_force": k_retenu,
                "motif": (MOTIF_REVUE if k_retenu == K_APRES_REVUE
                          else "K imposé par k_force après revue humaine."),
            }
        # Les décisions décrivent la règle; la revue est signalée à part.
        retenu_regle = next(c for c in comparaison if c["k"] == k_regle)
        for c in comparaison:
            c["decision"] = _decision(c, retenu_regle, candidats)
            if k_retenu != k_regle:
                if c["k"] == k_regle:
                    c["decision"] = f"retenu par la règle, remplacé après revue par K={k_retenu}"
                elif c["k"] == k_retenu:
                    c["decision"] = f"retenu après revue humaine (règle : {c['decision']})"

        # 6. Modèle final et profils sur les variables originales.
        final = modeles[(k_retenu, seed_ref)]
        aggs = [F.count(F.lit(1)).alias("effectif")]
        for v in FEATURES:
            aggs.append(F.avg(v).alias(f"moy_{v}"))
            aggs.append(F.percentile_approx(v, 0.5).alias(f"med_{v}"))
        lignes_profil = (final.transform(standardise)
                         .groupBy("prediction").agg(*aggs).collect())
        centres = final.clusterCenters()
        centres_z = {int(c): {v: float(centres[c][i]) for i, v in enumerate(FEATURES)}
                     for c in range(len(centres))}

        # 7. Noms métier proposés à partir des centres en z-scores.
        noms = _nommer(centres_z)

        profils = []
        for r in sorted(lignes_profil, key=lambda r: r["prediction"]):
            c = int(r["prediction"])
            profils.append({
                "cluster": c,
                "nom_propose": noms[c],
                "effectif": int(r["effectif"]),
                "part_pct": _r(100.0 * r["effectif"] / n),
                "moyenne": {v: _r(r[f"moy_{v}"]) for v in FEATURES},
                "mediane": {v: _r(r[f"med_{v}"]) for v in FEATURES},
                "centre_z": {v: _r(z) for v, z in centres_z[c].items()},
            })
    finally:
        standardise.unpersist()

    # 8. Durée totale (servira à la Partie D).
    resultat = {
        "features": list(FEATURES),
        "standardisation": "StandardScaler(withMean=True, withStd=True), ajusté une seule fois",
        "lignes_utilisees": int(n),
        "lignes_retirees_nulles": int(lignes_retirees),
        "seeds": seeds,
        "comparaison": comparaison,
        "k_regle": k_regle,
        "k_retenu": k_retenu,
        "regle_selection": REGLE_SELECTION,
        "justification": justification,
        "revue_humaine": revue,
        "profils": profils,
        "noms_metier": (f"Propositions générées par règles sur les centres en z-scores "
                        f"(seuil ±{SEUIL_Z} écart-type); à valider en lisant les profils."),
        "pourquoi_pas_le_cout_seul": POURQUOI_PAS_LE_COUT_SEUL,
        "duree_secondes": _r(time.perf_counter() - debut, 3),
    }

    # 10. Ligne de résumé, sur le modèle de l'audit P1.
    print("KMEANS_P3=" + json.dumps(resultat, ensure_ascii=False, separators=(",", ":")))
    return resultat
