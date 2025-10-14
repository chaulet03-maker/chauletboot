import os
import time
import requests

from notifier import notify

TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")


def listen_reanudar(on_reanudar_callback):
    """
    Loopea mensajes nuevos del chat. Si encuentra 'reanudar' (case-insensitive),
    llama al callback para limpiar la pausa.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return  # sin credenciales, no escuchamos

    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    last_update_id = None

    while True:
        try:
            url = f"{base}/getUpdates"
            params = {"timeout": 20}
            if last_update_id:
                params["offset"] = last_update_id + 1
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()

            for upd in data.get("result", []):
                last_update_id = upd["update_id"]
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat_id = str(msg["chat"]["id"])
                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue
                text = (msg.get("text") or "").strip().lower()
                if text == "reanudar" or text == "/reanudar":
                    on_reanudar_callback()
                    notify(
                        "✅ Trading reanudado manualmente. Se eliminó la pausa vigente."
                    )
        except Exception:
            # silencioso; si falla reintenta
            pass
        time.sleep(2)
