"""Unified notifier utilities for Telegram alerts."""
from __future__ import annotations

import asyncio
import logging
import os
import time
import html
from typing import Iterable, Optional, Tuple

import requests
from telegram.constants import ParseMode
from bot.logger import _warn

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4096


def _load_token_chat() -> Tuple[Optional[str], Optional[str]]:
    token = None
    chat = None
    try:
        from config import get_telegram_token, get_telegram_chat_id  # type: ignore

        token = get_telegram_token() or None
        chat = get_telegram_chat_id() or None
    except Exception:
        pass

    token = token or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN")
    chat = chat or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT_ID")

    if chat is not None:
        chat = str(chat).strip()
    if token is not None:
        token = str(token).strip()
    return token, chat


_TELEGRAM_BOT_TOKEN, _TELEGRAM_CHAT_ID = _load_token_chat()


def _chunks(text: str, limit: int = TELEGRAM_MAX_LEN) -> Iterable[str]:
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
    parse_mode: Optional[str] = ParseMode.HTML,
    disable_preview: bool = True,
    timeout: Tuple[float, float] = (3.0, 7.0),
    retries: int = 3,
) -> bool:
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
                resp = requests.post(url, json=payload, timeout=timeout)
                if resp.status_code == 429:
                    try:
                        data = resp.json()
                        retry_after = (
                            (data.get("parameters") or {}).get("retry_after")
                            or data.get("retry_after")
                        )
                        delay = float(retry_after) if retry_after else max(2.0 * delay, 1.0)
                    except Exception:
                        delay = max(2.0 * delay, 1.0)
                    time.sleep(delay)
                    continue

                if resp.ok:
                    try:
                        if resp.json().get("ok", False):
                            break
                    except Exception:
                        break

                logger.warning("Notifier: Telegram %s: %s", resp.status_code, resp.text[:300])
                if attempt < retries:
                    time.sleep(delay)
                    delay = max(2.0 * delay, 1.0)
                    continue
                ok_all = False
            except requests.RequestException as exc:
                _warn(
                    "NOTIFIER",
                    f"Notifier: error de red (intento {attempt}/{retries})",
                    exc=exc,
                )
                if attempt < retries:
                    time.sleep(delay)
                    delay = max(2.0 * delay, 1.0)
                    continue
                ok_all = False
                break

    return ok_all


class Notifier:
    """Async notifier that reuses the Telegram application when available."""

    def __init__(self, application, cfg):
        self.app = application
        self.config = cfg or {}
        self.chat_id = self._resolve_chat_id()

    def _resolve_chat_id(self) -> Optional[str]:
        cfg = self.config if isinstance(self.config, dict) else {}
        chat = cfg.get("telegram_chat_id")
        if not chat:
            telegram_cfg = cfg.get("telegram") if isinstance(cfg, dict) else {}
            if isinstance(telegram_cfg, dict):
                chat = telegram_cfg.get("chat_id") or telegram_cfg.get("chat")
        if not chat:
            chat = _TELEGRAM_CHAT_ID
        return str(chat) if chat else None

    @staticmethod
    def _sanitize_html(text: str) -> str:
        safe = html.escape(str(text), quote=False)
        for tag in ("b", "i", "u", "s", "code", "pre"):
            safe = safe.replace(f"&lt;{tag}&gt;", f"<{tag}>")
            safe = safe.replace(f"&lt;/{tag}&gt;", f"</{tag}>")
        return safe

    async def send(self, message, *, disable_preview: bool = True):
        text = str(message)
        safe = self._sanitize_html(text)
        if self.app and self.chat_id:
            try:
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=safe,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=disable_preview,
                )
                return
            except Exception as exc:
                _warn("NOTIFIER", "Notifier: Telegram error", exc=exc)
        notify(safe, parse_mode=ParseMode.HTML, disable_preview=disable_preview)
