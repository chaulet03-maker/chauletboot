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


---

## Cambios aplicados en esta versión
# Cambios aplicados (ULTRA)
- Telegram ahora **acepta comandos** (`/start`, `/help`, etc.) y **responde a texto y comandos**.
- Si falta `TELEGRAM_BOT_TOKEN`, se **loggea error** en vez de fallar silencioso.
- Carga de `.env` más robusta: busca en el **cwd** y en el **root del proyecto**.
- Mensaje de **startup** ya estaba implementado en `engine.py` y se enviará si `TELEGRAM_CHAT_ID` y token son válidos.
- Agregados:
  - `scripts/run_forever.sh`: reinicia el bot si se cae (modo "siempre encendido").
  - `systemd/bot-bestia-ultra.service`: unidad para systemd (install opcional).
  
## Cómo usar
1. Crear `.env` en la raíz del proyecto (misma carpeta donde está `bot/`):
```
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=123456789
```
2. Iniciar:
```
python3 -m bot.engine
```
Deberías recibir: `🤖 BESTIA ULTRA online (PAPER/LIVE)` en Telegram.

### Siempre encendido (opciones)
- **Bash**: `./scripts/run_forever.sh`
- **systemd** (copiar repo a `~/bestia-ultra` o ajustar `WorkingDirectory`):
```
cp systemd/bot-bestia-ultra.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now bot-bestia-ultra.service
```
