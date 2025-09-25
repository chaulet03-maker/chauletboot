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

# ✅ Imports correctos del proyecto
from bot.settings import load_config
from bot.engine import TradingApp
from bot.telemetry.telegram_bot import start_telegram_bot
try:
    from bot.telemetry.reporting import ReportingScheduler
except Exception:
    ReportingScheduler = None

def _normalize_symbols(cfg: dict) -> dict:
    """
    Admite symbols como lista o como dict con 'whitelist'.
    """
    syms = cfg.get("symbols")
    if isinstance(syms, dict):
        syms = syms.get("whitelist", [])
        cfg["symbols"] = syms
    return cfg

async def main():
    # Algunos repos esperan load_config() sin args; otros reciben el modo.
    try:
        cfg = load_config()
    except TypeError:
        cfg = load_config(os.getenv("MODE"))

    cfg = _normalize_symbols(cfg)

    app = TradingApp(cfg)
    await app.start()

    # 🚀 Iniciar bot de Telegram (no duplicar si ya estuviera iniciándose en otro lado)
    try:
        asyncio.create_task(start_telegram_bot(app, cfg))
    except Exception as e:
        logging.getLogger(__name__).warning("telegram bot not started: %s", e)

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
