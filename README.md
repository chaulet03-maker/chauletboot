# BOT_BESTIA_ULTRA — Futures Bot Enterprise 🐺⚡

**Meta:** Bot autónomo con **entradas significativas**, **apalancamiento dinámico (x5–x15)**, **fees neteados**, 
**circuit breakers** (día/semana/global), **VaR/Kelly** operativo, **límite de exposición/correlación**, 
**detección de régimen**, **DCA + TPs escalonados + trailing ATR**, **hedge táctico BTC/ETH**, 
**microestructura** (spread/liquidez, slippage guard), **persistencia de estado**, **reintentos/backoff**,
**webhooks** para Grafana/TradingView, **backtesting real**, **Monte Carlo** y **optimizer**.

> Disclaimer: uso educativo. Probá primero en **PAPER**.

---

## Puesta en marcha (Windows PowerShell)

```powershell
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
copy config\config.example.yaml config\config.yaml
py start.py
```

## Telegram (comandos)
- `ayuda` → lista y breve explicación de comandos.
- `precio` → último BTC/USDT y ETH/USDT.
- `estado` → PnL neto día/semana/30d + equity.
- `posiciones` → lado/qty/lev/entry/stop/TPs.
- `cerrar todo` | `cerra btc` | `cerra eth` → cierre manual.
- `bot on` | `bot off` → habilita/pausa nuevas entradas.
- `diag` → **diagnóstico** del porqué **no abre** (últimos motivos por símbolo).
- `test` → test de conectividad.

## Mensajes (apertura/cierre)
- **Apertura**: flecha ↑ LONG / ↓ SHORT, margen/lev/notional, entrada, TP1/TP2/SL con PnL esperado, orden y latencia.
- **Cierre**: ✅/❌, qty, entrada/salida, PnL **neto**, % y holding.

## Backtesting / Monte Carlo / Optimizer
- `python scripts/backtest.py --csv data/hist/BTCUSDT_1m.csv --symbol BTC/USDT --timeframe 1m`
- `python scripts/montecarlo.py data/backtests/RESULTS.csv`
- `python scripts/optimize.py --symbol BTC/USDT`

## Diferenciales
- **Circuit breakers** día/semana/global.
- **VaR & Kelly** operativos desde historial de trades.
- **Exposición/correlación** por símbolo. Hedge táctico BTC/ETH.
- **Persistencia** real (state.json): posiciones, allow_new_entries, equity snapshot.
- **Command Queue** via `data/cmd_queue.json` (Telegram → Engine).
- **Webhooks** en `telemetry/webhooks.py`.
