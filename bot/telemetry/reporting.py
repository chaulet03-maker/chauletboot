import asyncio, logging, pandas as pd, datetime as dt, pytz, os
from telegram import Bot

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
        csv_dir = self.cfg.get("storage",{}).get("csv_dir","data")
        try:
            eq = pd.read_csv(f"{csv_dir}/equity.csv", parse_dates=["ts"])
            tr = pd.read_csv(f"{csv_dir}/trades.csv", parse_dates=["ts"])
        except Exception:
            return None
        now = pd.Timestamp.utcnow()
        eq['ts'] = pd.to_datetime(eq['ts'], utc=True)
        tr['ts'] = pd.to_datetime(tr['ts'], utc=True)
        eqp = eq[eq['ts'] >= (now - pd.Timedelta(days=days))]
        trp = tr[tr['ts'] >= (now - pd.Timedelta(days=days))]

        pnl = float(eqp['pnl'].sum()) if not eqp.empty else 0.0
        n = len(trp)
        wins = int(((trp['pnl'] > 0).sum()) if 'pnl' in trp else 0)
        winrate = (wins/n*100.0) if n>0 else 0.0
        dd = 0.0
        if not eqp.empty:
            peak = eqp['equity'].cummax()
            dd = float(((eqp['equity'] - peak)/peak).min() * 100.0)

        return f"""{title}
Equity: ${self.app.trader.equity():.2f}
PnL ({days}d): ${pnl:.2f}
Trades: {n} | Win-rate: {winrate:.1f}%
Max Drawdown: {dd:.2f}%
Posiciones abiertas: {sum(len(v) for v in self.app.trader.state.positions.values())}"""
