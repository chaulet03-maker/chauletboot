# run_simulador_examples.ps1
param(
  [string]$RepoRoot = ".",
  [string]$HistDir = ".\full\data\hist"
)
python "$RepoRoot\simulador_final.py" --csv1h "$HistDir\BTCUSDT_USDT_1h_usdm" --csv4h "$HistDir\BTCUSDT_USDT_4h_usdm" --modes TENDENCIA,RANGO
python "$RepoRoot\simulador_final.py" --csv1h "$HistDir\BNBUSDT_USDT_1h_usdm" --csv4h "$HistDir\BNBUSDT_USDT_4h_usdm" --modes TENDENCIA,RANGO
