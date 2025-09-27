# === PRO BOOTSTRAP (defaults + logging) ===
try:
    from pro_defaults import apply_defaults as _pro_apply_defaults
    _pro_apply_defaults()
except Exception as _e:
    import logging; logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger(__name__).warning("pro_defaults not applied: %s", _e)

import logging, os, sys, asyncio
logging.basicConfig(level=getattr(logging, os.environ.get("LOG_LEVEL","INFO").upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
def _pro_excepthook(exc_type, exc, tb):
    logging.getLogger("UNCAUGHT").exception("Uncaught exception", exc_info=(exc_type, exc, tb))
sys.excepthook = _pro_excepthook
try:
    loop = asyncio.get_event_loop()
    def _pro_async_handler(loop, context):
        logging.getLogger("ASYNC").error("Async exception: %s", context.get("message"), exc_info=context.get("exception"))
    loop.set_exception_handler(_pro_async_handler)
except Exception:
    pass
# === END PRO BOOTSTRAP ===

import asyncio, os, sys
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
os.chdir(BASE)
sys.path.insert(0, str(BASE))

# ✅ Imports del proyecto
from bot.settings import load_config
from bot.engine import TradingApp
from bot.telemetry.telegram_bot import start_telegram_bot
# ⬇️ IMPORTANTE: comandos de Telegram
from bot.telemetry.telegram_commands import CommandBot

try:
    from bot.telemetry.reporting import ReportingScheduler
except Exception:
    ReportingScheduler = None

def _normalize_symbols(cfg: dict) -> dict:
    """
    Acepta symbols como lista o como dict con 'whitelist'.
    Normaliza a lista simple (engine-friendly).
    """
    syms = cfg.get("symbols")
    if isinstance(syms, dict):
        syms = syms.get("whitelist", [])
        cfg["symbols"] = syms
    return cfg

# === ADAPTADOR DE CONFIG (evita claves ignoradas por el engine) ===
def _adapt_cfg(cfg: dict) -> dict:
    out = dict(cfg)

    # 1) timeframe: subirlo del bloque strategy al top-level
    tf = (cfg.get("strategy") or {}).get("timeframe")
    if tf:
        out["timeframe"] = tf

    # 2) limits: mapear nombres usados por el engine
    lim_in = cfg.get("limits", {}) or {}
    lim_out = out.setdefault("limits", {})
    if "max_open_total" in lim_in:
        lim_out["max_total_positions"] = lim_in["max_open_total"]
    if "max_open_per_symbol" in lim_in:
        lim_out["max_per_symbol"] = lim_in["max_open_per_symbol"]
    if "no_hedge" in lim_in:
        lim_out["no_hedge"] = lim_in["no_hedge"]

    # 3) persistence -> storage
    pers = cfg.get("persistence", {}) or {}
    sto = out.setdefault("storage", {})
    if pers.get("dir"):
        sto["csv_dir"] = pers["dir"]
    if pers.get("sqlite_file"):
        sto["sqlite_path"] = pers["sqlite_file"]

    # 4) slippage: execution -> paper
    exe = cfg.get("execution", {}) or {}
    pap = out.setdefault("paper", {})
    if "slippage_bps" in exe:
        pap["slippage_bps"] = exe["slippage_bps"]

    # 5) fees por si faltan (evita KeyError y asegura realismo)
    if "fees" not in out:
        out["fees"] = {"taker": 0.0005, "maker": 0.0002}

    return out
# === FIN ADAPTADOR ===

def _log_boot_info(cfg: dict):
    log = logging.getLogger("BOOT")
    syms = cfg.get("symbols", [])
    tf = cfg.get("timeframe", (cfg.get("strategy") or {}).get("timeframe"))
    fees = (cfg.get("fees") or {})
    taker = fees.get("taker", "n/a")
    maker = fees.get("maker", "n/a")
    slipp = ((cfg.get("paper") or {}).get("slippage_bps")
             or (cfg.get("execution") or {}).get("slippage_bps"))
    mode = cfg.get("mode", os.getenv("MODE", "paper"))
    log.info("Mode: %s | Timeframe: %s | Symbols: %s", mode, tf, ", ".join(syms))
    log.info("Fees: taker=%s maker=%s | Slippage bps=%s", taker, maker, slipp)

async def main():
    # Algunos repos esperan load_config() sin args; otros reciben el modo.
    try:
        cfg = load_config()
    except TypeError:
        cfg = load_config(os.getenv("MODE"))

    cfg = _normalize_symbols(cfg)
    cfg = _adapt_cfg(cfg)
    _log_boot_info(cfg)

    app = TradingApp(cfg)
    await app.start()

    # 🚀 Iniciar bot de Telegram (alertas + reportes)
    try:
        asyncio.create_task(start_telegram_bot(app, cfg))
    except Exception as e:
        logging.getLogger(__name__).warning("telegram bot not started: %s", e)

    # 🚀 Iniciar listener de comandos (texto plano)
    try:
        asyncio.create_task(CommandBot(app).run())
    except Exception as e:
        logging.getLogger(__name__).warning("telegram commands not started: %s", e)

    # (Opcional) Reporting programado, si está disponible
    if ReportingScheduler is not None:
        try:
            asyncio.create_task(ReportingScheduler(app, cfg).run())
        except Exception as e:
            logging.getLogger(__name__).warning("reporting not started: %s", e)

    await app.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

# === PRO PARITY AUTOPATCH ===
try:
    from pro_tools.autopatch import autopatch_if_enabled as _pro_parity_autopatch
    _pro_parity_autopatch()
except Exception as _e:
    import logging; logging.getLogger(__name__).warning("Parity autopatch not applied: %s", _e)
# === END PRO PARITY AUTOPATCH ===
