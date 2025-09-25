import os, logging
from telegram import Bot

log = logging.getLogger("notifier")

class Notifier:
  def __init__(self):
    # accept both TELEGRAM_TOKEN and TELEGRAM_BOT_TOKEN
    self.token = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
    self.bot = Bot(self.token) if self.token else None

  async def send(self, text: str):
    if not self.bot or not self.chat_id:
      return
    try:
      await self.bot.send_message(self.chat_id, text)
    except Exception as e:
      log.warning("notify failed: %s", e)
