# Auditoría de sintaxis

Se realizó una compilación en frío de todos los archivos Python del repositorio utilizando `compile()` para verificar errores de sintaxis.

## Resultados por archivo

- bot/config_validator.py — ✅ Sin errores de sintaxis
- bot/core/indicators.py — ✅ Sin errores de sintaxis
- bot/core/market_regime.py — ✅ Sin errores de sintaxis
- bot/core/regime.py — ✅ Sin errores de sintaxis
- bot/core/risk.py — ✅ Sin errores de sintaxis
- bot/core/strategy.py — ✅ Sin errores de sintaxis
- bot/engine.py — ✅ Sin errores de sintaxis
- bot/exchanges/binance_client.py — ✅ Sin errores de sintaxis
- bot/exchanges/paper.py — ✅ Sin errores de sintaxis
- bot/exchanges/real.py — ✅ Sin errores de sintaxis
- bot/execution/trader.py — ✅ Sin errores de sintaxis
- bot/health/endpoint.py — ✅ Sin errores de sintaxis
- bot/logger.py — ✅ Sin errores de sintaxis
- bot/risk/guards.py — ✅ Sin errores de sintaxis
- bot/risk/trailing.py — ✅ Sin errores de sintaxis
- bot/settings.py — ✅ Sin errores de sintaxis
- bot/state.py — ✅ Sin errores de sintaxis
- bot/storage/csv_store.py — ✅ Sin errores de sintaxis
- bot/storage/sqlite_store.py — ✅ Sin errores de sintaxis
- bot/telemetry/__init__.py — ✅ Sin errores de sintaxis
- bot/telemetry/formatter.py — ✅ Sin errores de sintaxis
- bot/telemetry/notifier.py — ✅ Sin errores de sintaxis
- bot/telemetry/reporting.py — ✅ Sin errores de sintaxis
- bot/telemetry/telegram_bot.py — ✅ Sin errores de sintaxis
- bot/telemetry/telegram_commands.py — ✅ Sin errores de sintaxis
- bot/telemetry/webhooks.py — ✅ Sin errores de sintaxis
- bot/trader.py — ✅ Sin errores de sintaxis
- descargar_datos.py — ✅ Sin errores de sintaxis
- optimizador.py — ✅ Sin errores de sintaxis
- pro_defaults.py — ✅ Sin errores de sintaxis
- pro_tools/autopatch.py — ✅ Sin errores de sintaxis
- pro_tools/data_ccxt.py — ✅ Sin errores de sintaxis
- pro_tools/fee_aware.py — ✅ Sin errores de sintaxis
- pro_tools/leverage_policy.py — ✅ Sin errores de sintaxis
- pro_tools/paper_futures_engine.py — ✅ Sin errores de sintaxis
- pro_tools/parity.py — ✅ Sin errores de sintaxis
- pro_tools/risk_sizer.py — ✅ Sin errores de sintaxis
- pro_tools/rounding.py — ✅ Sin errores de sintaxis
- pro_tools/run_walkforward.py — ✅ Sin errores de sintaxis
- pro_tools/strategy_cots.py — ✅ Sin errores de sintaxis
- pro_tools/walkforward.py — ✅ Sin errores de sintaxis
- scripts/backtest.py — ✅ Sin errores de sintaxis
- scripts/montecarlo.py — ✅ Sin errores de sintaxis
- scripts/optimize.py — ✅ Sin errores de sintaxis
- simulador_final.py — ✅ Sin errores de sintaxis
- start.py — ✅ Sin errores de sintaxis
- systemd/start.py — ✅ Sin errores de sintaxis

## Metodología

El análisis se ejecutó con el siguiente comando:

```bash
python - <<'PY'
import compileall, pathlib
root = pathlib.Path('.')
files = sorted(root.rglob('*.py'))
for path in files:
    try:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    except SyntaxError as exc:
        print(f"{path}: ERROR {exc.msg} at line {exc.lineno}")
    else:
        print(f"{path}: OK")
PY
```

No se detectaron errores de sintaxis en ninguno de los archivos inspeccionados.
