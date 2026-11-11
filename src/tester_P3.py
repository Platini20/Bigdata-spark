"""Tests de la Partie B (P3) : segmentation K-Means de segmentation.py.

Deux modes :
  - par défaut : mini-DataFrame synthétique (300 clients, 3 groupes bien
    séparés) et test unitaire de l'ARI; ne lit pas data/.
  - --reel : chaîne complète P1 -> P2 -> P3 sur les données fournies; le
    résultat est écrit dans results/kmeans_P3_test.json (à ne pas commiter).

Le master n'est pas fixé dans le code : il vient de spark-submit.

Lancement en local[*] dans l'image du projet (depuis la racine du dépôt,
Git Bash; PYTHONDONTWRITEBYTECODE évite d'écrire des .pyc dans src/) :
  MSYS_NO_PATHCONV=1 docker run --rm -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$(pwd -W):/workspace" -w /tmp inf6303-spark-mllib:4.2.0 \
    /opt/spark/bin/spark-submit --master "local[*]" /workspace/src/tester_P3.py [--reel]

Lancement sur le cluster (port 8080 libre, après docker compose up -d) :
  docker compose exec -e PYTHONDONTWRITEBYTECODE=1 spark-client \
    /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
    --conf spark.driver.host=spark-client /workspace/src/tester_P3.py [--reel]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pyspark.sql import SparkSession

from segmentation import _ari, segmenter

K_VALUES = [2, 3, 4, 5, 6]


def tester_ari():
    """ARI : partition identique (à une permutation près) -> 1,0; exemple classique -> 0,5714."""
    assert _ari([(0, 1, 50), (1, 0, 50)]) == 1.0
    # [0, 0, 1, 1] contre [0, 0, 1, 2]
    assert round(_ari([(0, 0, 2), (1, 1, 1), (1, 2, 1)]), 4) == 0.5714
    print("TEST ARI OK")


def rfm_synthetique(spark, n_par_groupe=100, graine=6303):
    """Trois groupes de clients bien séparés, avec les colonnes de construire_rfm."""
    alea = random.Random(graine)
    # (récence, fréquence, montant, diversité) au centre de chaque groupe
    groupes = [(5, 35, 4000.0, 6), (200, 5, 150.0, 2), (40, 15, 900.0, 3)]
    lignes = []
    for g, (r, f, m, d) in enumerate(groupes):
        for i in range(n_par_groupe):
            lignes.append((g * n_par_groupe + i,
                           max(1, r + alea.randint(-3, 3)),
                           max(1, f + alea.randint(-1, 1)),
                           round(m * alea.uniform(0.95, 1.05), 2),
                           max(1, d + alea.randint(-1, 1)) if g != 1 else d))
    return spark.createDataFrame(
        lignes, "client_id int, recency_days int, frequency long, "
                "amount double, product_diversity long")


def verifier(res):
    """Contrôles de forme communs aux deux modes."""
    json.dumps(res)  # échoue si un type non natif subsiste
    assert [c["k"] for c in res["comparaison"]] == K_VALUES
    for c in res["comparaison"]:
        assert -1.0 <= c["silhouette"] <= 1.0
        assert -1.0 <= c["silhouette_moyenne"] <= 1.0
    assert 2 <= res["k_retenu"] <= 6
    assert len(res["profils"]) == res["k_retenu"]
    somme_parts = sum(p["part_pct"] for p in res["profils"])
    assert abs(somme_parts - 100.0) < 0.01, somme_parts
    noms = [p["nom_propose"] for p in res["profils"]]
    assert len(set(noms)) == len(noms), noms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reel", action="store_true", help="données réelles (P1 -> P2 -> P3)")
    p.add_argument("--k-force", type=int, default=None,
                   help="K imposé (par défaut : règle seule en synthétique, "
                        "K_APRES_REVUE en --reel, comme l'appel de P5)")
    p.add_argument("--regle-seule", action="store_true",
                   help="--reel avec k_force=None : la règle automatique seule")
    args = p.parse_args()

    spark = (SparkSession.builder.appName("INF6303-Test-P3")
             .config("spark.sql.shuffle.partitions", "24").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        print("MASTER=" + spark.sparkContext.master)
        if args.reel:
            from analyse_sql import construire_rfm
            from devoir2_spark import charger_et_nettoyer
            transactions, _, catalogue = charger_et_nettoyer(spark, 24)
            rfm = construire_rfm(transactions, catalogue)
            if args.regle_seule:
                res = segmenter(rfm, k_force=None)
            elif args.k_force is not None:
                res = segmenter(rfm, k_force=args.k_force)
            else:
                res = segmenter(rfm)  # appel prévu par P5 : k_force=K_APRES_REVUE
            verifier(res)
            assert 2 <= res["k_regle"] <= 6
            if res["k_retenu"] != res["k_regle"]:
                assert res["revue_humaine"] is not None
            sortie = Path(__file__).resolve().parent.parent / "results" / "kmeans_P3_test.json"
            sortie.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"TEST P3 REEL OK : K règle={res['k_regle']}, K retenu={res['k_retenu']}, "
                  f"résultat écrit dans {sortie}")
        else:
            tester_ari()
            res = segmenter(rfm_synthetique(spark), k_force=args.k_force)
            verifier(res)
            if args.k_force is None:
                assert res["k_retenu"] == res["k_regle"] and res["revue_humaine"] is None
            print(f"TEST P3 SYNTHETIQUE OK : K={res['k_retenu']}")
        for c in res["comparaison"]:
            print(f"  K={c['k']} silhouette_moy={c['silhouette_moyenne']} "
                  f"ari={c['ari_moyen']} part_min={c['part_min_cluster']} | {c['decision']}")
        for pr in res["profils"]:
            print(f"  cluster {pr['cluster']} {pr['nom_propose']} : {pr['part_pct']} %")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
