param(
    [Parameter(Mandatory=$true)][ValidateSet(1,3)][int]$Workers,
    [int]$Partitions = 24
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    docker info | Out-Null
    docker compose up -d --scale spark-worker=$Workers
    $deadline=(Get-Date).AddMinutes(2)
    do {
        Start-Sleep -Seconds 2
        try {$state=Invoke-RestMethod "http://localhost:8080/json/" -TimeoutSec 3; $alive=@($state.workers|Where-Object {$_.state -eq "ALIVE"}).Count} catch {$alive=0}
    } while($alive -ne $Workers -and (Get-Date) -lt $deadline)
    if($alive -ne $Workers){throw "Workers attendus: $Workers; detectes: $alive"}
    $suffix=if($Workers -eq 1){"worker"}else{"workers"}
    $result="results/resultat_${Workers}_${suffix}.json"
    if(Test-Path $result){Remove-Item $result -Force}
    docker compose exec spark-client /opt/spark/bin/spark-submit `
      --master spark://spark-master:7077 --deploy-mode client `
      --executor-cores 1 --total-executor-cores $Workers --executor-memory 1100m --driver-memory 1g `
      --conf spark.driver.host=spark-client --conf spark.driver.bindAddress=0.0.0.0 `
      --conf spark.driver.port=7078 --conf spark.driver.blockManager.port=7079 `
      --conf spark.dynamicAllocation.enabled=false --conf spark.sql.shuffle.partitions=$Partitions `
      /workspace/src/devoir2_spark.py --workers $Workers --partitions $Partitions
    if(-not(Test-Path $result)){throw "Fichier de resultat absent: $result"}
    Write-Host "Resultat: $result" -ForegroundColor Green
} finally {Pop-Location}
