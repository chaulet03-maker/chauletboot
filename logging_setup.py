import logging
import os
import re
import time
from collections import deque
from logging.handlers import RotatingFileHandler

logging.Formatter.default_msec_format = "%s,%03d"

LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
os.makedirs(LOG_DIR, exist_ok=True)


class RedactTokenFilter(logging.Filter):
    _pat = re.compile(
        r"(api\.telegram\.org/bot)[A-Za-z0-9:_-]+|bot\d+:[A-Za-z0-9_-]{20,}"
    )

    @classmethod
    def _replace(cls, match: re.Match) -> str:
        prefix = match.group(1)
        if prefix:
            return f"{prefix}***REDACTED***"
        return "bot***:***"

    @classmethod
    def _sanitize(cls, value):
        if isinstance(value, str):
            return cls._pat.sub(cls._replace, value)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._sanitize(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._sanitize(v) for k, v in record.args.items()}
            else:
                record.args = tuple(self._sanitize(arg) for arg in record.args)
        return True


class RateLimitFilter(logging.Filter):
    """Filter that throttles repeated INFO/WARNING records."""

    def __init__(self, min_interval_sec: float = 300) -> None:
        super().__init__()
        self.min_interval = float(min_interval_sec)
        self._last: dict[tuple[int, str], float] = {}

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - timing based
        if not (logging.INFO <= record.levelno <= logging.WARNING):
            return True
        key = (record.levelno, record.msg)
        now = time.time()
        last = self._last.get(key, 0.0)
        if now - last >= self.min_interval:
            self._last[key] = now
            return True
        return False


def setup_logging() -> logging.Logger:
    logger = logging.getLogger()
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(RedactTokenFilter())

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(RedactTokenFilter())

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.addFilter(RateLimitFilter())

    class DequeHandler(logging.Handler):
        def __init__(self, maxlen: int = 5000) -> None:
            super().__init__()
            self.buf: deque[str] = deque(maxlen=maxlen)
            self.setFormatter(fmt)

        def emit(self, record: logging.LogRecord) -> None:
            self.buf.append(self.format(record))

    mem_handler = DequeHandler()
    logger.addHandler(mem_handler)
    logger._memh = mem_handler  # type: ignore[attr-defined]
    logger.info("Logger inicializado. Archivo: %s", LOG_FILE)
    for name in ["httpx", "httpcore", "urllib3", "telegram"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("telegram.http").setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.ERROR)
    logging.getLogger("apscheduler.scheduler").setLevel(logging.ERROR)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.ERROR)
    logging.getLogger("apscheduler.jobstores").setLevel(logging.ERROR)
    logging.getLogger("apscheduler.executors").setLevel(logging.ERROR)
    logging.captureWarnings(True)
    return logger
