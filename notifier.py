import os
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")  # export TG_BOT_TOKEN="123456:ABC..."
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")  # export TG_CHAT_ID="123456789"


def notify(msg: str) -> None:
    """Envía un mensaje a Telegram. Idempotente y silencioso ante fallos."""
    token = TELEGRAM_BOT_TOKEN
    chat = TELEGRAM_CHAT_ID
    if not token or not chat:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat, "text": msg})
    except Exception:
        pass
