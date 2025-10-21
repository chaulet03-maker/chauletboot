import asyncio
import datetime as dt
import logging
import os

import pytz
from telegram import Bot

from bot.identity import get_bot_id
from bot.ledger import pnl_summary as ledger_pnl_summary

logger = logging.getLogger("reporting")

class ReportingScheduler:
    def __init__(self, app, config):
        self.app = app
        self.cfg = config
        self.local_tz = pytz.timezone(os.environ.get("TZ","America/Argentina/Buenos_Aires"))
        self.daily_hour = int(config.get("reporting",{}).get("daily_hour_local",9))
        self.weekly_weekday = int(config.get("reporting",{}).get("weekly_weekday_local",0))
        self.weekly_hour = int(config.get("reporting",{}).get("weekly_hour_local",9))
        self.weekly_minute = int(config.get("reporting",{}).get("weekly_minute_local",5))
        self.bot = None

    def _bot(self):
        if self.bot: return self.bot
        token = os.getenv("TELEGRAM_TOKEN")
        if not token: return None
        self.bot = Bot(token=token)
        return self.bot

    async def run(self):
        while True:
            try:
                await self.maybe_send_daily()
                await self.maybe_send_weekly()
            except Exception as e:
                logger.exception("reporting error: %s", e)
            await asyncio.sleep(55)

    async def _send(self, text: str):
        bot = self._bot()
        if not bot: return
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not chat_id: return
        try:
            await bot.send_message(chat_id, text)
        except Exception as e:
            logger.warning("telegram send failed: %s", e)

    async def maybe_send_daily(self):
        now_local = dt.datetime.now(self.local_tz)
        if now_local.minute != 0 or now_local.hour != self.daily_hour:
            return
        txt = self.build_report(days=1, title="📣 Reporte diario")
        if txt:
            await self._send(txt)

    async def maybe_send_weekly(self):
        now_local = dt.datetime.now(self.local_tz)
        if now_local.weekday() != self.weekly_weekday or now_local.hour != self.weekly_hour or now_local.minute != self.weekly_minute:
            return
        txt = self.build_report(days=7, title="📣 Reporte semanal")
        if txt:
            await self._send(txt)

    def build_report(self, days: int, title: str):
        try:
            mode = "live" if str(self.app.is_live).lower() in {"true", "1"} else "paper"
            bid = get_bot_id()
            pnl = ledger_pnl_summary(mode, bid)  # usa DB, no CSV
            daily = pnl["daily"] if days == 1 else pnl["weekly"]
            equity_now = float(self.app.trader.equity() or 0.0)
            realized = float(daily.get("realized", 0.0))
            unreal = float(daily.get("unrealized", 0.0))
            total = float(daily.get("total", 0.0))
        except Exception:
            return None

        return (
            f"{title}\n"
            f"Equity: ${equity_now:.2f}\n"
            f"PnL ({'24h' if days==1 else '7d'}): ${total:.2f} "
            f"(R={realized:+.2f} | U={unreal:+.2f})"
        )
