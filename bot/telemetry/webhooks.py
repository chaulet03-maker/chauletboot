import requests
from bot.settings import WEBHOOK_URL, WEBHOOK_TOKEN

def post_event(event: dict):
    if not WEBHOOK_URL: return
    try:
        headers = {"Authorization": f"Bearer {WEBHOOK_TOKEN}"} if WEBHOOK_TOKEN else {}
        requests.post(WEBHOOK_URL, json=event, headers=headers, timeout=5)
    except Exception:
        pass
