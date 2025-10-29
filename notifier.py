import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def notify(msg: str) -> None:
    """Envía un mensaje a Telegram. Idempotente y silencioso ante fallos."""

    token = TELEGRAM_BOT_TOKEN
    chat = TELEGRAM_CHAT_ID
    if not token or not chat:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat, "text": msg}, timeout=5)
    except Exception:
        pass
