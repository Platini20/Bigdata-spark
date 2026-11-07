# Commandes à copier-coller - Devoir 2

```powershell
docker info
docker compose build --pull
.\scripts\executer_devoir2.ps1 -Workers 1 -Partitions 24
.\scripts\executer_devoir2.ps1 -Workers 3 -Partitions 24
Get-Content .\results\resultat_1_worker.json
Get-Content .\results\resultat_3_workers.json
.\scripts\arreter_cluster.ps1
```

Interfaces pendant le calcul : Master UI http://localhost:8080 ; Spark UI http://localhost:4040.
