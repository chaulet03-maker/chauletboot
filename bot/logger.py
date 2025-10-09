import logging, os, json, time
from logging.handlers import RotatingFileHandler

LOG_DIR="logs"; os.makedirs(LOG_DIR, exist_ok=True)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base={"ts":int(time.time()*1000),"lvl":record.levelname,"name":record.name,"msg":record.getMessage()}
        if record.exc_info: base["exc"]=self.formatException(record.exc_info)
        if hasattr(record,"extra"):
            try: base.update(record.extra)
            except Exception: pass
        return json.dumps(base, ensure_ascii=False)

def build_logger(name, level, file):
    lg=logging.getLogger(name); lg.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not lg.handlers:
        fh=RotatingFileHandler(file, maxBytes=3_000_000, backupCount=4, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        sh=logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        lg.addHandler(fh); lg.addHandler(sh)
    return lg

BOT_LOGGER = build_logger("bot", os.getenv("LOG_LEVEL","INFO"), "logs/bot.log")
DEC_LOGGER = build_logger("decisions", os.getenv("LOG_LEVEL","INFO"), "logs/decisions.log")

def decision_event(code:str, message:str, **context):
    from bot.settings import DEBUG_DECISIONS
    payload={"code":code,"message":message,"context":context}
    DEC_LOGGER.info(message, extra={"extra": payload})

def log_exception(logger, message: str, **context):
    try: logger.exception(message, extra={"extra":{"context":context}})
    except Exception: logger.exception(message)
