$ErrorActionPreference = "Stop"
 
$ROOT = "E:\phd\exp1"

$SCRIPTS = Join-Path $ROOT "scripts"

$RESULTS = Join-Path $ROOT "results"
 
New-Item -ItemType Directory -Force -Path $RESULTS | Out-Null
 
$env:HF_HOME = Join-Path $ROOT "hf_home"

$env:TRANSFORMERS_CACHE = Join-Path $ROOT "hf_cache"
 
$LOGFILE = Join-Path $RESULTS ("run_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

Start-Transcript -Path $LOGFILE
 
try {

    Set-Location $SCRIPTS
 
    Write-Host "===== RUNNING SINGLE AGENT ====="

    python .\single_agent_eval.py
 
    Write-Host "===== RUNNING CONSENSUS AGENT ====="

    python .\multi_agent_consensus.py
 
    Write-Host "===== RUNNING TWO AGENT COLLAB ====="

    python .\two_agent_collab.py
 
    Write-Host "===== GENERATING TRUST + EXCEL ANALYSIS ====="

    python .\analyze_results.py
 
    Write-Host "===== DONE. Check E:\phd\exp1\results ====="

}

finally {

    Stop-Transcript

}

 