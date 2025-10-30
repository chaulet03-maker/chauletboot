# notifier.py
import logging
import os
import time
from typing import Iterable, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4096  # límite de Telegram

def _load_token_chat() -> Tuple[Optional[str], Optional[str]]:
    """
    Intenta primero con config.py (si existe),
    luego con variables de entorno.
    """
    token = None
    chat = None
    try:
        # si tenés config.py con getters (como en endpoint.py)
        from config import get_telegram_token, get_telegram_chat_id  # type: ignore
        token = get_telegram_token() or None
        chat = get_telegram_chat_id() or None
    except Exception:
        pass

    # fallback a ENV
    token = token or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN")
    chat = chat or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT_ID")

    # normalización simple
    if chat is not None:
        chat = str(chat).strip()
    if token is not None:
        token = str(token).strip()

    return token, chat

_TELEGRAM_BOT_TOKEN, _TELEGRAM_CHAT_ID = _load_token_chat()

def _chunks(text: str, limit: int = TELEGRAM_MAX_LEN) -> Iterable[str]:
    """Parte el texto en trozos <= limit, cortando en '\n' cuando sea posible."""
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    out = []
    s = text
    while s:
        if len(s) <= limit:
            out.append(s)
            break
        cut = s.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(s[:cut])
        s = s[cut:].lstrip("\n")
    return out

def notify(
    msg: str,
    *,
    parse_mode: Optional[str] = None,      # "MarkdownV2" | "HTML" | None
    disable_preview: bool = True,
    timeout: Tuple[float, float] = (3.0, 7.0),  # (connect, read)
    retries: int = 3,
) -> bool:
    """
    Envía msg a Telegram. Devuelve True/False.
    - Reintenta con backoff (maneja 429 retry_after).
    - Corta mensajes >4096.
    - Loguea warnings si falla (no rompe el proceso).
    """
    token = _TELEGRAM_BOT_TOKEN
    chat = _TELEGRAM_CHAT_ID
    if not token or not chat:
        logger.debug("Notifier: faltan TELEGRAM_BOT_TOKEN/CHAT_ID. No se envía.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok_all = True

    for part in _chunks(str(msg), TELEGRAM_MAX_LEN):
        payload = {
            "chat_id": chat,
            "text": part,
            "disable_web_page_preview": disable_preview,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        delay = 1.0
        for attempt in range(1, max(1, retries) + 1):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                if r.status_code == 429:
                    # Rate limit: respetar retry_after si viene
                    try:
                        data = r.json()
                        retry_after = (
                            data.get("parameters", {}).get("retry_after")
                            or data.get("retry_after")
                        )
                        delay = float(retry_after) if retry_after else max(2.0 * delay, 1.0)
                    except Exception:
                        delay = max(2.0 * delay, 1.0)
                    time.sleep(delay)
                    continue

                if r.ok:
                    try:
                        if r.json().get("ok", False):
                            break  # listo este chunk
                    except Exception:
                        # si no parsea JSON pero ok HTTP, lo damos por bueno
                        break

                # HTTP != ok
                logger.warning("Notifier: Telegram %s: %s", r.status_code, r.text[:300])
                if attempt < retries:
                    time.sleep(delay)
                    delay = max(2.0 * delay, 1.0)
                    continue
                ok_all = False
            except requests.RequestException as e:
                logger.warning("Notifier: error de red (intento %d/%d): %s", attempt, retries, e)
                if attempt < retries:
                    time.sleep(delay)
                    delay = max(2.0 * delay, 1.0)
                    continue
                ok_all = False
                break

    return ok_all
Cómo integrarlo con tu endpoint.py
Ya que ahí usás config, podés delegar:

python
Copiar código
# en endpoint.py
from notifier import notify

def enviar_notificacion(mensaje):
    if not notify(mensaje):
        print("Advertencia: no se pudo enviar la notificación por Telegram.")
