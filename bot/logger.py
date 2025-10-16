import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

from logging_setup import LOG_DIR, RedactTokenFilter, setup_logging


setup_logging()


class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": int(time.time() * 1000),
            "lvl": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            try:
                base.update(record.extra)
            except Exception:
                pass
        return json.dumps(base, ensure_ascii=False)


def _managed_handler_exists(logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "_managed", False):
            return True
    return False


def _attach_file_handler(logger: logging.Logger, path: str) -> None:
    os.makedirs(os.path.dirname(path) or LOG_DIR, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=3_000_000, backupCount=4, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactTokenFilter())
    handler._managed = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


def build_logger(name: str, level: str, file_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if file_path and not _managed_handler_exists(logger):
        _attach_file_handler(logger, file_path)
    return logger


BOT_LOGGER = build_logger("bot", os.getenv("LOG_LEVEL", "INFO"), None)
BOT_LOGGER.propagate = True

DEC_LOGGER = build_logger(
    "decisions",
    os.getenv("LOG_LEVEL", "INFO"),
    os.path.join(LOG_DIR, "decisions.log"),
)

def decision_event(code:str, message:str, **context):
    from bot.settings import DEBUG_DECISIONS
    payload={"code":code,"message":message,"context":context}
    DEC_LOGGER.info(message, extra={"extra": payload})

def log_exception(logger, message: str, **context):
    try: logger.exception(message, extra={"extra":{"context":context}})
    except Exception: logger.exception(message)
