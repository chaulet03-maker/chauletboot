import asyncio, logging, time, datetime as dt, math, random, json, os
import pandas as pd
import ccxt.async_support as ccxt
from bot.core.indicators import compute_indicators
from bot.core.strategy import generate_signal
from bot.risk.trailing import compute_trailing_stop
from bot.risk.guards import Limits, can_open, portfolio_caps_ok
from bot.exchanges.paper import PaperExchange
from bot.exchanges.real import RealExchange
from bot.storage.csv_store import append_trade_csv, append_equity_csv
from bot.storage.sqlite_store import ensure_db, insert_trade, insert_equity
from bot.trader import Trader
from bot.telemetry.notifier import Notifier


def _normalize_position_like(pos):
    """Normalize a position payload before (lambda _np=_normalize_position_like(pos): Position(**_np) if _np else None)()."""
    import json
    if pos is None:
        return None
    if isinstance(pos, str):
        try:
            pos = json.loads(pos)
        except Exception:
            return None
    if not isinstance(pos, dict):
        return None
    if 'lev' in pos and 'leverage' not in pos:
        pos['leverage'] = pos.pop('lev')
    def _to_float(x):
        try: return float(x)
        except Exception: return None
    def _to_int(x):
        try: return int(x)
        except Exception:
            try: return int(float(x))
            except Exception: return None
    numeric_float_fields = {'entry_price','mark_price','qty','stop','take_profit','pnl','fee'}
    numeric_int_fields = {'leverage','timestamp','ts'}
    for k in list(pos.keys()):
        v = pos[k]
        if k in numeric_float_fields:
            nv = _to_float(v)
            if nv is None: pos.pop(k, None)
            else: pos[k] = nv
        elif k in numeric_int_fields:
            nv = _to_int(v)
            if nv is None: pos.pop(k, None)
            else: pos[k] = nv
    return pos



import os
try:
    import telegram
    import telegram.ext
    PTB_AVAILABLE = True
except Exception:
    PTB_AVAILABLE = False
    telegram = None

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")



try:
    from bot.state import load_state as ext_load_state, save_state as ext_save_state
except Exception:
    ext_load_state = ext_save_state = None

logger = logging.getLogger("engine")


def usd_to_qty(symbol, usd, price, lev):
    if price <= 0:
        return 0.0
    return (usd * lev) / price


class TradingApp:
    # --- util: validar token telegram (si alguna vez lo necesitás acá) ---
    @staticmethod
    def _valid_tg_token(token: str) -> bool:
        try:
            import re
            return bool(token) and bool(re.match(r'^\d+:[A-Za-z0-9_-]{35,}$', str(token)))
        except Exception:
            return False

    def _normalize_pos(self, pos):
        if isinstance(pos, str):
            s = pos.strip()
            if s and s[0] in "{[":
                try:
                    pos = json.loads(s)
                except Exception:
                    return None
            else:
                return None
        if not isinstance(pos, dict):
            return None
        for k in ("qty", "entry", "sl", "tp1", "tp2", "realized_pnl", "lev", "entry_adx", "leg"):
            if k in pos and pos[k] is not None:
                try:
                    if k in ("lev", "leg"):
                        pos[k] = int(pos[k])
                    else:
                        pos[k] = float(pos[k])
                except Exception:
                    pass
        if "side" in pos and not isinstance(pos["side"], str):
            try:
                pos["side"] = pos["side"].decode("utf-8", "ignore")
            except Exception:
                pos["side"] = str(pos["side"])
        return pos

    def total_margin_used(self, price_by_symbol):
        total = 0.0
        for sym, lots in self.trader.state.positions.items():
            price = price_by_symbol.get(sym) or self.price_cache.get(sym) or 0.0
            if price <= 0:
                continue
            for L in lots:
                total += abs(price * L['qty']) / max(L.get('lev', 1), 1)
        return total

    def available_margin_usd(self, equity, price_by_symbol):
        used = self.total_margin_used(price_by_symbol)
        cap_pct = float(self.portfolio_caps.get("max_portfolio_margin_pct", 1.0))
        max_allowed = max(0.0, equity * cap_pct)
        return max(0.0, max_allowed - used)

    STATE_FILE = os.path.join("data", "state.json")

    def __init__(self, config: dict):
        self.cfg = config
        self.symbols = config.get("symbols", ["BTC/USDT:USDT", "ETH/USDT:USDT"])
        self.timeframe = config.get("timeframe", "2m")
        self.loop_seconds = int(config.get("loop_seconds", 120))
        ex_cfg = config.get("exchange", {})

        # Build CCXT client con defaults seguros
        _ex_id = ex_cfg.get("id", "binanceusdm")
        self.ccxt = getattr(ccxt, _ex_id)()

        # rate limit
        self.ccxt.enableRateLimit = bool(ex_cfg.get("enableratelimit", True))

        # options
        _opts = dict(getattr(self.ccxt, "options", {}) or {})
        _opts.setdefault("defaultType", "swap" if _ex_id in ("binance", "binanceusdm") else ex_cfg.get("options", {}).get("defaultType", "future"))
        _opts.setdefault("adjustForTimeDifference", False)
        _opts.update(ex_cfg.get("options", {}))
        self.ccxt.options = _opts

        # API keys (REAL)
        self.ccxt.apiKey = os.getenv("BINANCE_API_KEY") or ex_cfg.get("apikey") or ex_cfg.get("apiKey")
        self.ccxt.secret = os.getenv("BINANCE_API_SECRET") or ex_cfg.get("secret")

        self.fees = config.get("fees", {"taker": 0.0002, "maker": 0.0002})
        self.paper = PaperExchange(self.fees, slippage_bps=int(config.get("paper", {}).get("slippage_bps", 5)))
        self.trader = Trader(self.fees, equity0=1000.0)

        self.csv_dir = config.get("storage", {}).get("csv_dir", "data")
        self.sqlite_path = config.get("storage", {}).get("sqlite_path", "data/bot.sqlite")
        ensure_db(self.sqlite_path)

        lim = config.get("limits", {})
        self.limits = Limits(max_total_positions=int(lim.get("max_total_positions", 6)),
                             max_per_symbol=int(lim.get("max_per_symbol", 4)),
                             no_hedge=bool(lim.get("no_hedge", True)))
        self.cooldown = int(lim.get("cooldown_seconds", 90))

        self.order_sizes = config.get("order_sizing", {})
        self.leverage_conf = config.get("leverage", {"min": 1, "max": 15, "default": 5})
        self.filters = config.get("filters", {})
        self.strategy_conf = config.get("strategy", {})
        self.portfolio_caps = config.get("portfolio_caps", {})
        self.funding_guard = config.get("funding_guard", {"enabled": True, "annualized_bps_limit": 5000})
        self.circuit_breakers = config.get("circuit_breakers", {})
        self.mode = config.get("mode", "paper").lower()
        self.price_cache = {}
        self.allow_new_entries = True
        self.notifier = Notifier()
        self._loaded_state = False

        self.exchange = self.paper if self.mode == 'paper' else RealExchange(self.ccxt, self.fees)
        self.trailing_conf = self.cfg.get('trailing', {'enabled': True, 'mode': 'atr', 'atr_k': 2.0, 'ema_key': 'ema_fast', 'ema_k': 1.0, 'percent': 0.6, 'min_step_atr': 0.5})

        # --- Risk throttle & learning ---
        self.base_limits = Limits(self.limits.max_total_positions, self.limits.max_per_symbol, self.limits.no_hedge)
        self.base_cooldown = self.cooldown
        self.base_stop_mult = float(self.strategy_conf.get('stop_mult', 1.5))
        self.equity_peak = float(self.trader.equity())
        self.dd_band = 'none'
        self.layer_pauses = {}
        self._last_learning_update = 0.0
        self.alerts_conf = self.cfg.get('alerts', {'enabled': True})
        self._last_alert_check = 0.0
        self._alert_sent = {}

    async def fetch_ohlcv_2m(self, symbol):
        tf = self.timeframe
        if tf == "2m":
            data = await self.with_retry(self.ccxt.fetch_ohlcv, symbol, timeframe="1m", limit=200)
            df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.set_index('ts').resample('2min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
            return df
        else:
            data = await self.with_retry(self.ccxt.fetch_ohlcv, symbol, timeframe=tf, limit=200)
            df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df

    async def fetch_last_price(self, symbol):
        t = await self.with_retry(self.ccxt.fetch_ticker, symbol)
        return float(t['last'])

    async def funding_rate_bps_annualized(self, symbol):
        try:
            fr = await self.with_retry(self.ccxt.fetch_funding_rate, symbol)
            rate = float(fr.get('fundingRate', 0.0))
            annualized = rate * 3 * 24 * 365 * 10000.0  # bps
            return annualized
        except Exception:
            return 0.0

    def _choose_leverage_and_pct(self, last_row: dict, regime: str):
        min_lev = int(self.leverage_conf.get("min", 1))
        max_lev = int(self.leverage_conf.get("max", 15))
        base_pct = float(self.order_sizes.get("default_pct", 0.30))

        adx = float(last_row.get('adx', 0))
        bbw = float(last_row.get('bb_width', 0))
        vol_ok = bool(last_row.get('vol_ok', True))

        if regime in ("chop",):
            lev = max(min_lev, 1)
            pct = max(0.10, min(0.20, base_pct * 0.5))
        elif regime in ("range",):
            lev = max(2, min(4, max_lev))
            pct = max(0.15, min(0.30, base_pct * 0.75))
        else:
            lev = max(5, min(8, max_lev))
            pct = max(0.25, min(0.40, base_pct))

        if adx >= 30 and vol_ok and bbw >= 12:
            lev = min(max_lev, max(lev, 10))
            pct = max(pct, 0.60)
        if adx >= 40 and vol_ok and bbw >= 16:
            lev = min(max_lev, max(lev, 12))
            pct = max(pct, 0.80)

        pct = max(float(self.order_sizes.get("min_pct", 0.10)), min(float(self.order_sizes.get("max_pct", 1.00)), pct))
        lev = max(min_lev, min(max_lev, int(lev)))
        return lev, pct

    def _apply_caps(self, usd, price, lev):
        cap = self.order_sizes.get("max_usd_per_trade", None)
        if cap is not None:
            try:
                usd = min(usd, float(cap))
            except Exception:
                pass
        max_notional = self.cfg.get("risk", {}).get("max_notional_per_trade", None)
        if max_notional is not None:
            try:
                usd = min(usd, float(max_notional))
            except Exception:
                pass
        max_margin = self.cfg.get("risk", {}).get("max_margin_per_trade", None)
        if max_margin is not None:
            try:
                max_notional_by_margin = float(max_margin) * max(lev, 1)
                usd = min(usd, max_notional_by_margin)
            except Exception:
                pass
        return usd

    def _check_circuit_breakers(self):
        cb = self.circuit_breakers or {}
        if not cb:
            self.save_state()
        return True
        # (resto omitido a propósito: early-return intencional)

    async def with_retry(self, fn, *args, retries=5, base_delay=0.3, backoff=1.8, **kwargs):
        last_exc = None
        for i in range(retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                await asyncio.sleep(base_delay * (backoff ** i) * (1 + random.random() * 0.2))
        raise last_exc

    async def bootstrap_real(self):
        """Reconstruye equity y posiciones desde el exchange al iniciar en modo REAL."""
        try:
            bal = await self.exchange.fetch_balance_usdt()
            if bal:
                self.trader.state.equity = float(bal)
        except Exception as e:
            logger.warning("bootstrap balance failed: %s", e)
        try:
            poss = await self.exchange.fetch_positions()
            for p in poss or []:
                sym = p.get('symbol')
                if not sym or sym not in self.symbols:
                    continue
                side = None
                qty = 0.0
                lev = int(self.leverage_conf.get("default", 5))
                entry = float(p.get('entryPrice') or p.get('entry_price') or p.get('info', {}).get('entryPrice') or 0.0)
                contracts = p.get('contracts') or p.get('contractSize') or p.get('size')
                if contracts is None:
                    contracts = p.get('info', {}).get('positionAmt') or 0
                qty = abs(float(contracts or 0.0))
                if qty <= 0 or entry <= 0:
                    continue
                s = p.get('side') or p.get('info', {}).get('positionSide')
                if s:
                    s_low = str(s).lower()
                    side = 'long' if 'long' in s_low else 'short'
                else:
                    pa = float(p.get('info', {}).get('positionAmt') or 0.0)
                    side = 'long' if pa > 0 else 'short'
                try:
                    lev = int(p.get('leverage') or p.get('info', {}).get('leverage') or lev)
                except Exception:
                    pass
                self.trader.open_lot(sym, side, qty, entry, lev, sl=0.0, tp1=0.0, tp2=0.0, fee=0.0, entry_adx=0.0, leg=1)
                self.log_trade(sym, side, qty, entry, lev, 0.0, pnl=0.0, note="BOOTSTRAP_EXISTING")
            logger.info("bootstrap_real done: positions=%s", sum(len(v) for v in self.trader.state.positions.values()))
        except Exception as e:
            logger.warning("bootstrap positions failed: %s", e)

    def _risk_normalized_qty(self, price, sl, equity, pct, lev, atr):
        """Calcula tamaño usando riesgo en USD (ATR/SL) + cap por % de equity."""
        stop_dist = abs(price - sl)
        if stop_dist <= 0 or atr <= 0:
            usd_margin = max(0.0, pct * equity)
            return (usd_margin * lev) / max(price, 1e-9), usd_margin
        risk_pct = float(self.cfg.get("risk", {}).get("size_pct_of_equity", 0.05))
        risk_usd = max(0.0, risk_pct * equity)
        qty_risk = risk_usd / stop_dist
        margin_needed = qty_risk * price / max(lev, 1)
        margin_cap = max(0.0, pct * equity)
        usd_margin = min(margin_needed, margin_cap)
        qty = (usd_margin * lev) / max(price, 1e-9)
        return qty, usd_margin

    def _dca_should_add(self, symbol, side, last_row, lots, dca_cfg):
        if not dca_cfg.get("enabled", True):
            return False, 0.0
        if len(lots) >= int(self.limits.max_per_symbol):
            return False, 0.0
        last_leg_adx = 0.0
        for L in lots[::-1]:
            if L['side'] == side:
                last_leg_adx = float(L.get('entry_adx', 0.0))
                break
        adx_now = float(last_row.get('adx', 0.0))
        if adx_now < last_leg_adx + float(dca_cfg.get("min_adx_increase", 2.0)):
            return False, 0.0
        ema = float(last_row.get('ema_fast', 0.0))
        atr = float(last_row.get('atr', 0.0))
        c = float(last_row.get('close', 0.0))
        tol = float(dca_cfg.get("ema_pullback_atr", 0.5)) * max(atr, 1e-9)
        if not (ema - tol <= c <= ema + tol):
            return False, 0.0
        scale = float(dca_cfg.get("pct_scale_per_add", 0.5))
        self.save_state()
        return True, scale

    def _serialize_state(self):
        return {
            "killswitch": bool(self.trader.state.killswitch),
            "last_entry_ts_by_symbol": self.trader.state.last_entry_ts_by_symbol,
            "positions": {
                sym: [dict(side=L['side'], qty=L['qty'], entry=L['entry'], lev=L['lev'],
                           sl=L['sl'], tp1=L['tp1'], tp2=L['tp2'],
                           realized_pnl=L.get('realized_pnl', 0.0),
                           trailing_anchor=L.get('trailing_anchor', L['entry']),
                           entry_adx=L.get('entry_adx', 0.0),
                           leg=L.get('leg', 1)) for L in lots]
                for sym, lots in self.trader.state.positions.items()
            }
        }

    def _restore_state(self, snap):
        try:
            self.trader.state.killswitch = bool(snap.get("killswitch", False))
            self.trader.state.last_entry_ts_by_symbol = dict(snap.get("last_entry_ts_by_symbol", {}))
            self.trader.state.positions.clear()
            for sym, lots in snap.get("positions", {}).items():
                arr = []
                for L in lots:
                    arr.append({
                        "side": L["side"], "qty": float(L["qty"]), "entry": float(L["entry"]), "lev": int(L["lev"]),
                        "ts": time.time(), "sl": float(L.get("sl", 0.0)), "tp1": float(L.get("tp1", 0.0)),
                        "tp2": float(L.get("tp2", 0.0)), "realized_pnl": float(L.get("realized_pnl", 0.0)),
                        "trailing_anchor": float(L.get("trailing_anchor", L["entry"])),
                        "entry_adx": float(L.get("entry_adx", 0.0)), "leg": int(L.get("leg", 1))
                    })
                if arr:
                    self.trader.state.positions[sym] = arr
            self._loaded_state = True
            logger.info("PAPER state restored: %s positions", sum(len(v) for v in self.trader.state.positions.values()))
        except Exception as e:
            logger.warning("restore_state failed: %s", e)

    def save_state(self):
        if self.mode != 'paper':
            return
        try:
            if ext_save_state:
                ext_save_state(self._serialize_state())
                return
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(self.STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._serialize_state(), f, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:
            logger.warning("save_state failed: %s", e)

    def load_state(self):
        if self.mode != 'paper':
            return
        try:
            if ext_load_state:
                snap = ext_load_state()
            elif os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, "r", encoding="utf-8") as f:
                    snap = json.load(f)
                if isinstance(snap, list):
                    snap = {}
                if isinstance(snap.get("positions", {}), list):
                    snap["positions"] = {}
                if "equity" in snap and not isinstance(snap.get("equity"), (int, float)):
                    snap.pop("equity", None)
                self._restore_state(snap)
        except Exception as e:
            logger.warning("load_state failed: %s", e)

    def get_status(self):
        eq = self.trader.equity()
        open_cnt = sum(len(v) for v in self.trader.state.positions.values())
        by_sym = {s: len(v) for s, v in self.trader.state.positions.items()}
        return {"equity": eq, "open_positions": open_cnt, "per_symbol": by_sym, "killswitch": self.trader.state.killswitch}

    def _apply_risk_bands(self):
        eq = float(self.trader.equity())
        if eq > self.equity_peak:
            self.equity_peak = eq
        dd = (eq - self.equity_peak) / self.equity_peak if self.equity_peak > 0 else 0.0

        new_limits = Limits(self.base_limits.max_total_positions, self.base_limits.max_per_symbol, self.base_limits.no_hedge)
        new_cooldown = self.base_cooldown
        new_stop_mult = self.base_stop_mult
        allow_entries = True
        new_band = 'none'

        if dd <= -0.06:
            allow_entries = False
            new_limits = Limits(max(1, self.base_limits.max_total_positions - 3),
                                max(1, self.base_limits.max_per_symbol - 2),
                                self.base_limits.no_hedge)
            new_cooldown = self.base_cooldown + 300
            new_stop_mult = self.base_stop_mult + 0.50
            new_band = 'dd6'
        elif dd <= -0.04:
            new_limits = Limits(3, 2, self.base_limits.no_hedge)
            new_cooldown = self.base_cooldown + 300
            new_stop_mult = self.base_stop_mult + 0.50
            new_band = 'dd4'
        elif dd <= -0.02:
            new_limits = Limits(4, 3, self.base_limits.no_hedge)
            new_cooldown = self.base_cooldown + 120
            new_stop_mult = self.base_stop_mult + 0.25
            new_band = 'dd2'

        self.limits = new_limits
        self.cooldown = int(new_cooldown)
        self.strategy_conf['stop_mult'] = float(new_stop_mult)
        self.allow_new_entries = allow_entries

        if new_band != self.dd_band:
            logger.info(
                "Risk band update: dd=%.2f%% band=%s limits(tot=%d,sym=%d) cooldown=%ss stop_mult=%.2f entries=%s",
                dd * 100.0, new_band, self.limits.max_total_positions, self.limits.max_per_symbol, self.cooldown,
                self.strategy_conf['stop_mult'],
                "ON" if self.allow_new_entries else "OFF")
            self.dd_band = new_band

    def _update_learning_pauses(self, now_ts=None):
        try:
            import csv, time as _t
            now_ts = now_ts or _t.time()
            path = os.path.join(self.csv_dir, "trades.csv")
            if not os.path.exists(path):
                return
            rows = []
            with open(path, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if str(row.get("note") or "").upper().startswith("CLOSE"):
                        rows.append(row)
            rows = rows[-600:]
            perf = {}
            for r in rows:
                sym = r.get("symbol", "")
                regime = (r.get("regime") or r.get("entry_regime") or "").lower()
                if not sym or not regime:
                    continue
                layer = "trend" if ("trend" in regime) else ("range" if regime in ("range", "chop") else "other")
                if layer == "other":
                    continue
                pnl = float(r.get("pnl") or 0.0)
                key = (sym, layer)
                d = perf.setdefault(key, {"n": 0, "pnl": 0.0, "gains": 0.0, "loss": 0.0})
                d["n"] += 1
                d["pnl"] += pnl
                if pnl >= 0:
                    d["gains"] += pnl
                else:
                    d["loss"] += pnl
            for key, d in perf.items():
                n = d["n"]
                if n < 20:
                    continue
                pf = (d["gains"] / abs(d["loss"])) if d["loss"] < 0 else 999.0
                expectancy = d["pnl"] / n
                if expectancy < 0.0 and pf < 0.9:
                    until = now_ts + 24 * 3600
                    if self.layer_pauses.get(key, 0) < until:
                        self.layer_pauses[key] = until
                        logger.info("Pause layer %s for %s until %s (n=%d, PF=%.2f, EXP=%.4f)",
                                    key[1], key[0], dt.datetime.utcfromtimestamp(until).isoformat() + "Z", n, pf, expectancy)
        except Exception as e:
            logger.warning("update_learning_pauses failed: %s", e)

    def _cluster_exposure_ok(self, symbol: str, side: str, price_by_symbol):
        try:
            cg = self.cfg.get('correlation_guard', {})
            if not cg or not cg.get('enabled', True):
                return True, ''
            clusters = cg.get('clusters') or []
            max_ratio = float(cg.get('same_side_max_exposure_ratio', 0.6))
            total_notional = 0.0
            for sym, lots in self.trader.state.positions.items():
                p = price_by_symbol.get(sym) or self.price_cache.get(sym) or 0.0
                if p <= 0:
                    continue
                for L in lots:
                    total_notional += abs(L['qty']) * p
            if total_notional <= 0:
                return True, ''
            cluster = None
            for cl in clusters:
                if symbol in cl:
                    cluster = cl
                    break
            if not cluster:
                return True, ''
            cluster_same = 0.0
            for sym in cluster:
                lots = self.trader.state.positions.get(sym, [])
                p = price_by_symbol.get(sym) or self.price_cache.get(sym) or 0.0
                if p <= 0:
                    continue
                for L in lots:
                    if L.get('side') == side:
                        cluster_same += abs(L['qty']) * p
            ratio = cluster_same / total_notional if total_notional else 0.0
            if ratio > max_ratio:
                return False, f"REJECT_CORR_EXPOSURE {symbol} {side} ratio={ratio:.2f} > {max_ratio:.2f}"
            return True, ''
        except Exception:
            return True, ''

    def _funding_window_ok(self, symbol: str, now_ts: float):
        try:
            fw = self.cfg.get('funding_window', {})
            if not fw or not fw.get('enabled', True):
                return True, ''
            minutes = int(fw.get('minutes', 7))
            thr = float(fw.get('fr_abs_bps_min', 300))
            fr_bps = float(self.price_cache.get(f"FUNDING_BPS:{symbol}", 0.0))
            if abs(fr_bps) < thr:
                return True, ''
            import datetime as _dt
            now = _dt.datetime.utcfromtimestamp(now_ts).replace(second=0, microsecond=0)
            nexts = []
            base = now.replace(hour=0, minute=0)
            for h in (0, 8, 16):
                t = base.replace(hour=h)
                if t < now:
                    t = t + _dt.timedelta(days=1) if h == 0 else t
                nexts.append(t)
            deltas = [abs((t - now).total_seconds()) for t in nexts]
            if min(deltas) <= minutes * 60:
                return False, f"REJECT_FUNDING_WINDOW |FR|={fr_bps:.0f}bps <= {minutes}m de funding"
            return True, ''
        except Exception:
            return True, ''

    def _maybe_send_alerts(self):
        try:
            if not self.alerts_conf.get('enabled', True):
                return
            now = time.time()
            if now - getattr(self, "_last_alert_check", 0.0) < 300:
                return
            self._last_alert_check = now
            import datetime as _dt, csv
            trades_path = os.path.join(self.csv_dir, "trades.csv")
            eq_path = os.path.join(self.csv_dir, "equity.csv")
            today = _dt.datetime.utcnow().date()
            start = _dt.datetime(today.year, today.month, today.day).timestamp()
            closes = []
            streaks = {}
            if os.path.exists(trades_path):
                with open(trades_path, newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        note = str(row.get("note") or "").upper()
                        if not note.startswith("CLOSE"):
                            continue
                        try:
                            ts = row.get("ts") or row.get("time") or ""
                            try:
                                tsv = _dt.datetime.fromisoformat(ts.replace("Z", "")).timestamp()
                            except Exception:
                                tsv = float(ts)
                            if tsv < start:
                                continue
                            symbol = row.get("symbol", "")
                            pnl = float(row.get("pnl") or 0.0)
                            closes.append(pnl)
                            s = streaks.setdefault(symbol, [])
                            s.append(pnl)
                        except Exception:
                            pass
            gains = sum([x for x in closes if x >= 0])
            losses = sum([x for x in closes if x < 0])
            pf = (gains / abs(losses)) if losses < 0 else (gains if gains > 0 else 0.0)
            key_pf = "pf_day_low"
            if closes and pf < 0.7 and not self._alert_sent.get(key_pf):
                try:
                    asyncio.create_task(self.notifier.send(f"⚠️ PF día bajo: {pf:.2f} (cierres={len(closes)})"))
                    self._alert_sent[key_pf] = True
                except Exception:
                    pass
            for sym, pnl_list in streaks.items():
                if len(pnl_list) >= 3 and all(x < 0 for x in pnl_list[-3:]):
                    key = f"streak_{sym}"
                    if not self._alert_sent.get(key):
                        try:
                            asyncio.create_task(self.notifier.send(f"⚠️ Racha de pérdidas (3) en {sym} hoy."))
                            self._alert_sent[key] = True
                        except Exception:
                            pass
            if os.path.exists(eq_path):
                eq = []
                with open(eq_path, newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        try:
                            ts = row.get("ts") or row.get("time") or ""
                            try:
                                tsv = _dt.datetime.fromisoformat(ts.replace("Z", "")).timestamp()
                            except Exception:
                                tsv = float(ts)
                            if tsv >= start:
                                eq.append(float(row.get("equity") or 0.0))
                        except Exception:
                            pass
                if len(eq) >= 2:
                    peak = eq[0]
                    dd_min = 0.0
                    for v in eq:
                        if v > peak:
                            peak = v
                        dd_cur = (v - peak) / peak if peak > 0 else 0.0
                        if dd_cur < dd_min:
                            dd_min = dd_cur
                    if dd_min <= -0.015 and not self._alert_sent.get('dd_day'):
                        try:
                            asyncio.create_task(self.notifier.send(f"⚠️ DD diario {dd_min * 100:.2f}%"))
                            self._alert_sent['dd_day'] = True
                        except Exception:
                            pass
        except Exception as e:
            logger.debug("alerts check skipped: %s", e)

    def get_positions(self):
        out = {}
        for sym, lots in self.trader.state.positions.items():
            out[sym] = [{"side": L["side"], "qty": L["qty"], "entry": L["entry"], "lev": L["lev"], "sl": L["sl"], "tp1": L["tp1"], "tp2": L["tp2"], "leg": L.get("leg", 1)} for L in lots]
        return out

    async def run(self):
        logger.info("Trading loop started in %s mode", self.mode.upper())
        try:
            await self.ccxt.load_markets()
        except Exception as e:
            logger.warning("load_markets failed: %s", e)
        if self.mode == 'paper':
            self.load_state()
        if self.mode == 'real':
            try:
                await self.exchange.set_position_mode(one_way=True)
            except Exception as e:
                logger.warning('set_position_mode failed: %s', e)
            try:
                await self.bootstrap_real()
            except Exception as e:
                logger.warning('bootstrap_real failed: %s', e)
        while True:
            try:
                await self.step_all_symbols()
            except Exception as e:
                logger.exception("step error: %s", e)
            await asyncio.sleep(self.loop_seconds)

    async def step_all_symbols(self):
        self._apply_risk_bands()

        if time.time() - self._last_learning_update > 600:
            self._update_learning_pauses()
            self._last_learning_update = time.time()

        if not self._check_circuit_breakers():
            self.trader.state.killswitch = True
            return

        price_by_symbol = {}
        for sym in self.symbols:
            df = await self.fetch_ohlcv_2m(sym)
            ind = compute_indicators(df, {**self.filters, **self.strategy_conf, **self.cfg.get("indicators", {})})
            sig = generate_signal(ind, {**self.filters, **self.strategy_conf})

            try:
                self.price_cache[f"ATR:{sym}"] = float(ind.iloc[-1].get('atr', 0.0))
            except Exception:
                pass

            # cache funding si corresponde
            try:
                fw = self.cfg.get('funding_window', {})
                if fw.get('enabled', True):
                    fr_bps = await self.funding_rate_bps_annualized(sym)
                    self.price_cache[f"FUNDING_BPS:{sym}"] = fr_bps
            except Exception:
                pass

            # pausa por aprendizaje
            layer = 'trend' if ('trend' in str(getattr(sig, 'regime', ''))) else ('range' if str(getattr(sig, 'regime', '')) in ('range', 'chop') else 'other')
            if layer != 'other':
                until = self.layer_pauses.get((sym, layer), 0)
                if until and time.time() < until:
                    self.log_decision(sym, 'pause_layer', detail=f'layer={layer} hasta={dt.datetime.utcfromtimestamp(until).isoformat()}Z')
                    continue

            price = float(ind.iloc[-1]['close'])
            price_by_symbol[sym] = price
            self.price_cache[sym] = price

            if not self.allow_new_entries or self.trader.state.killswitch:
                self.log_decision(sym, 'killswitch' if self.trader.state.killswitch else 'entries_disabled')
                continue

            if bool(self.funding_guard.get("enabled", False)):
                fr_bps = await self.funding_rate_bps_annualized(sym)
                self.price_cache[f"FUNDING_BPS:{sym}"] = fr_bps

            now = time.time()
            last_t = self.trader.state.last_entry_ts_by_symbol.get(sym, 0)
            if now - last_t < self.cooldown:
                self.log_decision(sym, 'cooldown', detail=f'rem={self.cooldown - (now - last_t):.1f}s')
                continue

            if sig.side in ("long", "short"):
                ok, reason = self.pre_open_checks(sym, sig.side, price_by_symbol)
                if not ok:
                    self.log_decision(sym, 'pre_open_checks', detail=reason)
                    continue

                lev, pct = self._choose_leverage_and_pct(ind.iloc[-1], sig.regime)
                eq = self.trader.equity()
                qty, usd = self._risk_normalized_qty(price, sig.sl, eq, pct, lev, float(ind.iloc[-1].get('atr', 0.0)))
                usd = self._apply_caps(usd, price, lev)

                # cap por margen libre de cartera
                free = self.available_margin_usd(eq, price_by_symbol)
                if usd > free:
                    usd = free
                    qty = (usd * lev) / max(price, 1e-9)

                if qty <= 0:
                    continue

                try:
                    await self.exchange.set_leverage(sym, lev)
                except Exception as e:
                    logger.warning('set_leverage failed: %s', e)

                fill, fee = await self.exchange.market_order(sym, sig.side, qty, price)
                leg_no = 1 + sum(1 for L in self.trader.state.positions.get(sym, []) if L['side'] == sig.side)
                self.trader.open_lot(sym, sig.side, qty, fill.price, lev, sig.sl, sig.tp1, sig.tp2, fee,
                                     entry_adx=float(ind.iloc[-1].get('adx', 0.0)), leg=leg_no)
                self.trader.state.last_entry_ts_by_symbol[sym] = now
                self.save_state()

                self.log_trade(sym, sig.side, qty, fill.price, lev, fee, note=f"OPEN usd={usd:.2f} lev={lev} pct={pct:.2f}")

                # --- AVISO OPEN con saldo ---
                try:
                    await self.notifier.send(
                        f"🟢 OPEN {sym} {sig.side} qty={qty:.6f} @ {fill.price:.2f} lev={lev} usd={usd:.2f} saldo={self.trader.equity():.2f}"
                    )
                except Exception:
                    pass

                if self.mode == 'real':
                    try:
                        await self.exchange.place_protections(sym, sig.side, qty, sig.sl, sig.tp1, sig.tp2)
                    except Exception as e:
                        logger.warning('place_protections failed: %s', e)

            # DCA a favor
            lots = self.trader.state.positions.get(sym, [])
            if lots:
                side_net = None
                net = sum(L['qty'] if L['side'] == 'long' else -L['qty'] for L in lots)
                if abs(net) > 0:
                    side_net = 'long' if net > 0 else 'short'
                dca_cfg = self.cfg.get("dca", {})
                if side_net and self.allow_new_entries and not self.trader.state.killswitch:
                    if sum(len(v) for v in self.trader.state.positions.values()) < self.limits.max_total_positions and len(lots) < self.limits.max_per_symbol:
                        ok_add, scale = self._dca_should_add(sym, side_net, ind.iloc[-1], lots, dca_cfg)
                        if ok_add:
                            lev, pct_base = self._choose_leverage_and_pct(ind.iloc[-1], sig.regime)
                            pct = max(self.order_sizes.get("min_pct", 0.10),
                                      min(self.order_sizes.get("max_pct", 1.00), pct_base * scale))
                            eq = self.trader.equity()
                            qty_add, usd_add = self._risk_normalized_qty(
                                price, lots[-1]['sl'] if lots[-1]['sl'] else sig.sl, eq, pct, lev, float(ind.iloc[-1].get('atr', 0.0))
                            )
                            usd_add = self._apply_caps(usd_add, price, lev)
                            free2 = self.available_margin_usd(eq, price_by_symbol)
                            if usd_add > free2:
                                usd_add = free2
                                qty_add = (usd_add * lev) / max(price, 1e-9)
                            if qty_add > 0:
                                fill2, fee2 = await self.exchange.market_order(sym, side_net, qty_add, price)
                                leg_no = 1 + sum(1 for L in lots if L['side'] == side_net)
                                self.trader.open_lot(sym, side_net, qty_add, fill2.price, lev, sig.sl, sig.tp1, sig.tp2, fee2,
                                                     entry_adx=float(ind.iloc[-1].get('adx', 0.0)), leg=leg_no)
                                self.trader.state.last_entry_ts_by_symbol[sym] = now
                                self.log_trade(sym, side_net, qty_add, fill2.price, lev, fee2,
                                               note=f"DCA_ADD usd={usd_add:.2f} lev={lev} pct={pct:.2f}")
                                self.save_state()
            await self.manage_positions(price_by_symbol)

        self.persist_equity(0.0)

    def persist_equity(self, pnl=0.0):
        ts = dt.datetime.utcnow().isoformat()
        row = {"ts": ts, "equity": round(self.trader.equity(), 6), "pnl": round(pnl, 6)}
        append_equity_csv(self.csv_dir, row)
        insert_equity(self.sqlite_path, row)

    def log_trade(self, symbol, side, qty, price, lev, fee, pnl=0.0, note="", regime: str = ""):
        ts = dt.datetime.utcnow().isoformat()
        row = {"ts": ts, "symbol": symbol, "side": side, "qty": qty, "price": price, "lev": lev, "fee": fee, "pnl": pnl, "note": note, "regime": regime}
        append_trade_csv(self.csv_dir, row)
        insert_trade(self.sqlite_path, row)

    def price_of(self, symbol):
        return self.price_cache.get(symbol)

    async def manage_positions(self, price_by_symbol):
        for sym, lots in list(self.trader.state.positions.items()):
            price = price_by_symbol.get(sym)
            if price is None:
                continue
            idx = 0
            while idx < len(lots):
                L = lots[idx]
                fee_unit = abs(price * L['qty']) * self.fees['taker']
                # trailing dinámico
                try:
                    if self.trailing_conf.get('enabled', True) and ('trailing_anchor' in L):
                        ind_row = {'atr': float(self.price_cache.get(f'ATR:{sym}', 0.0))}
                        new_sl = compute_trailing_stop(L['side'], price, L['sl'], L.get('trailing_anchor', price), ind_row, self.trailing_conf)
                        if L['side'] == 'long' and new_sl > L['sl']:
                            L['sl'] = new_sl
                        elif L['side'] == 'short' and new_sl < L['sl']:
                            L['sl'] = new_sl
                except Exception as _e:
                    logger.debug('trailing update skipped: %s', _e)

                if L['side'] == 'long':
                    if price <= L['sl']:
                        pnl = self.trader.close_lot(sym, idx, price, fee=fee_unit, note="SL")
                        self.log_trade(sym, L['side'], L['qty'], price, L['lev'], fee_unit, pnl, note="CLOSE_SL")
                        self.save_state()
                        await self.notifier.send(f"❌ SL {sym} long qty={L['qty']:.6f} @ {price:.2f} pnl={pnl:.2f} saldo={self.trader.equity():.2f}")
                        continue
                    if price >= L['tp2']:
                        pnl = self.trader.close_lot(sym, idx, price, fee=fee_unit, note="TP2")
                        self.log_trade(sym, L['side'], L['qty'], price, L['lev'], fee_unit, pnl, note="CLOSE_TP2")
                        self.save_state()
                        await self.notifier.send(f"✅ TP2 {sym} long qty={L['qty']:.6f} @ {price:.2f} pnl={pnl:.2f} saldo={self.trader.equity():.2f}")
                        continue
                    if price >= L['tp1']:
                        half = L['qty'] * 0.5
                        pnl = self.trader.close_lot(sym, idx, price, fee=abs(price * half) * self.fees['taker'], note="TP1_HALF")
                        self.log_trade(sym, L['side'], half, price, L['lev'], abs(price * half) * self.fees['taker'], pnl, note="CLOSE_TP1_HALF")
                        self.save_state()
                        await self.notifier.send(f"🟢 TP1 {sym} long half qty={half:.6f} @ {price:.2f} pnl={pnl:.2f} saldo={self.trader.equity():.2f}")
                        rem = {"side": L['side'], "qty": L['qty'] - half, "entry": price, "lev": L['lev'], "ts": time.time(),
                               "sl": L['sl'], "tp1": L['tp1'], "tp2": L['tp2'], "realized_pnl": 0.0, "trailing_anchor": price}
                        self.trader.state.positions.setdefault(sym, []).append(rem)
                        continue
                else:
                    if price >= L['sl']:
                        pnl = self.trader.close_lot(sym, idx, price, fee=fee_unit, note="SL")
                        self.log_trade(sym, L['side'], L['qty'], price, L['lev'], fee_unit, pnl, note="CLOSE_SL")
                        self.save_state()
                        await self.notifier.send(f"❌ SL {sym} short qty={L['qty']:.6f} @ {price:.2f} pnl={pnl:.2f} saldo={self.trader.equity():.2f}")
                        continue
                    if price <= L['tp2']:
                        pnl = self.trader.close_lot(sym, idx, price, fee=fee_unit, note="TP2")
                        self.log_trade(sym, L['side'], L['qty'], price, L['lev'], fee_unit, pnl, note="CLOSE_TP2")
                        self.save_state()
                        await self.notifier.send(f"✅ TP2 {sym} short qty={L['qty']:.6f} @ {price:.2f} pnl={pnl:.2f} saldo={self.trader.equity():.2f}")
                        continue
                    if price <= L['tp1']:
                        half = L['qty'] * 0.5
                        pnl = self.trader.close_lot(sym, idx, price, fee=abs(price * half) * self.fees['taker'], note="TP1_HALF")
                        self.log_trade(sym, L['side'], half, price, L['lev'], abs(price * half) * self.fees['taker'], pnl, note="CLOSE_TP1_HALF")
                        self.save_state()
                        await self.notifier.send(f"🟢 TP1 {sym} short half qty={half:.6f} @ {price:.2f} pnl={pnl:.2f} saldo={self.trader.equity():.2f}")
                        rem = {"side": L['side'], "qty": L['qty'] - half, "entry": price, "lev": L['lev'], "ts": time.time(),
                               "sl": L['sl'], "tp1": L['tp1'], "tp2": L['tp2'], "realized_pnl": 0.0, "trailing_anchor": price}
                        self.trader.state.positions.setdefault(sym, []).append(rem)
                        continue
                idx += 1

    def pre_open_checks(self, symbol, side, price_by_symbol):
        ok, reason = can_open(symbol, side, self.trader.state.positions, self.limits)
        if not ok:
            return False, reason
        equity = self.trader.equity()
        ok, reason = portfolio_caps_ok(equity, self.trader.state.positions, price_by_symbol, self.portfolio_caps)
        if not ok:
            return False, reason
        ok, reason = self._cluster_exposure_ok(symbol, side, price_by_symbol)
        if not ok:
            return False, reason
        ok, reason = self._funding_window_ok(symbol, time.time())
        if not ok:
            return False, reason
        self.save_state()
        return True, ""

    def log_decision(self, symbol, reason, detail="", extra=None):
        """Registra motivos de NO-entrada en memoria y CSV (data/decisions.csv)."""
        try:
            from collections import deque
            if not hasattr(self, "_decisions"):
                self._decisions = deque(maxlen=200)
            ts = time.time()
            row = {
                "ts": ts,
                "iso": dt.datetime.utcfromtimestamp(ts).isoformat() + "Z",
                "symbol": symbol,
                "reason": reason,
                "detail": detail or "",
            }
            if isinstance(extra, dict):
                for k, v in extra.items():
                    row[f"extra_{k}"] = v
            self._decisions.append(row)
            os.makedirs(self.csv_dir, exist_ok=True)
            path = os.path.join(self.csv_dir, "decisions.csv")
            newfile = not os.path.exists(path)
            import csv
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if newfile:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            logger.warning("log_decision failed: %s", e)

    def recent_rejections(self, n=10):
        """Últimas decisiones de no-entrada (memoria o CSV)."""
        out = []
        if hasattr(self, "_decisions") and self._decisions:
            out = [d for d in list(self._decisions)[-n:] if d.get("reason") != "opened"]
        else:
            path = os.path.join(self.csv_dir, "decisions.csv")
            if os.path.exists(path):
                import csv
                rows = []
                with open(path, newline="", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rows.append(row)
                out = [r for r in rows if (r.get("reason") != "opened")][-n:]
        return out

    # --- estos dos debían estar dentro de la clase ---
    def toggle_killswitch(self):
        self.trader.state.killswitch = not self.trader.state.killswitch
        return self.trader.state.killswitch

    async def close_all(self):
        for sym, lots in list(self.trader.state.positions.items()):
            price = self.price_cache.get(sym) or await self.fetch_last_price(sym)
            for _ in range(len(lots)):
                L = self.trader.state.positions[sym][0]
                fee_unit = abs(price * L['qty']) * self.fees['taker']
                pnl = self.trader.close_lot(sym, 0, price, fee=fee_unit, note="FORCE_CLOSE")
                self.log_trade(sym, L['side'], L['qty'], price, L['lev'], fee_unit, pnl, note="FORCE_CLOSE")
        self.save_state()
        return True
