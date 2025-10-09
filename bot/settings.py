import os, yaml
from dotenv import load_dotenv, find_dotenv

# Carga robusta de .env sin tocar tus valores
try:
    from dotenv import load_dotenv, find_dotenv
    _p = find_dotenv(usecwd=True)
    if _p:
        load_dotenv(_p)
    else:
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parents[1] / ".env"
        if _root.exists():
            load_dotenv(str(_root))
except Exception:
    pass
    

def load_config():
    cfg = os.getenv("CONFIG_PATH","config/config.yaml")
    if not os.path.exists(cfg):
        cfg = "config/config.example.yaml"
    with open(cfg,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

MODE = CONFIG.get("mode","PAPER").upper()
SYMBOLS = CONFIG.get("symbols",["BTC/USDT","ETH/USDT"])
TIMEFRAME = CONFIG.get("timeframe","1m")
LOOP_SECONDS = int(CONFIG.get("loop_seconds",60))

EXCONF = CONFIG.get("exchange",{})
FEES = CONFIG.get("fees",{"taker":0.0005,"maker":0.0002,"prefer":"taker"})
RISK = CONFIG.get("risk",{})
LEV = CONFIG.get("leverage",{"min":5,"max":15})
ORDER = CONFIG.get("order",{})
FILTERS = CONFIG.get("filters",{})
INDICATORS = CONFIG.get("indicators",{})
STRAT = CONFIG.get("strategy",{})
REGIME = CONFIG.get("regime",{})
FUNDING = CONFIG.get("funding",{"enabled":True,"extreme_abs_annual_bps":30})
WEBHOOKS = CONFIG.get("webhooks",{"enabled":False,"url":"","token":""})
TELEGRAM_CONF = CONFIG.get("telegram",{"enabled":True})

LOG_LEVEL = os.getenv("LOG_LEVEL","INFO")
DEBUG_DECISIONS = os.getenv("DEBUG_DECISIONS","0") == "1"

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_NOTIFY_ERRORS = os.getenv("TELEGRAM_NOTIFY_ERRORS","0") == "1"

WEBHOOK_URL = os.getenv("WEBHOOK_URL", WEBHOOKS.get("url",""))
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", WEBHOOKS.get("token",""))
