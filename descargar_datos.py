# descargar_datos.py
import argparse
import os
import time
import datetime as dt
import pandas as pd
import ccxt

DATA_FOLDER = "data"

# ---------------- Utilidades ---------------- #
def iso_utc(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%dT%H:%M:%SZ")

def tf_ms(exchange, timeframe: str) -> int:
    return exchange.parse_timeframe(timeframe) * 1000

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_exchange(market: str):
    market = market.lower().strip()
    if market == "usdm":
        ex = ccxt.binanceusdm({"enableRateLimit": True})
    elif market == "spot":
        ex = ccxt.binance({"enableRateLimit": True})
    else:
        raise ValueError("El mercado debe ser 'usdm' o 'spot'")
    ex.load_markets()
    return ex

def normalize_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp_ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df[["timestamp","open","high","low","close","volume"]]
    return df

def read_last_timestamp(filepath: str, timeframe_ms: int):
    if not os.path.exists(filepath):
        return None
    try:
        tail = pd.read_csv(filepath, usecols=["timestamp"]).tail(1)
        if tail.empty:
            return None
        last_iso = tail["timestamp"].iloc[0]
        last_ms = int(pd.Timestamp(last_iso).timestamp() * 1000)
        return last_ms + timeframe_ms
    except Exception:
        return None

def log_gap(prev_last_ms, new_first_ms, timeframe_ms, sym, tf):
    if prev_last_ms is None:
        return
    if new_first_ms > prev_last_ms + timeframe_ms:
        miss = (new_first_ms - prev_last_ms) // timeframe_ms
        print(f"[AVISO DE GAP] {sym} {tf}: faltan ~{int(miss)} velas entre {iso_utc(prev_last_ms)} y {iso_utc(new_first_ms)}")

# ---------------- Resolución de símbolos ---------------- #
def resolve_symbol(ex, user_symbol: str, market: str):
    user_symbol = user_symbol.upper().strip()
    if user_symbol in ex.symbols:
        base, quote = user_symbol.split("/", 1) if "/" in user_symbol else (None, None)
    else:
        base, quote = (None, None)
        if "/" in user_symbol:
            base, quote = user_symbol.split("/", 1)

    target_types = {"spot"} if market == "spot" else {"swap", "future"}
    candidates = []

    for m in ex.markets.values():
        try:
            if base and quote:
                if m.get("base") != base or m.get("quote") != quote:
                    continue
            if m.get("type") not in target_types:
                continue

            score = 0
            if market == "usdm":
                if m.get("type") == "swap": score += 50
                if m.get("linear"): score += 10
                ct = (m.get("info", {}) or {}).get("contractType") or (m.get("contractType"))
                if ct == "PERPETUAL": score += 100
                if ct in {"CURRENT_QUARTER", "NEXT_QUARTER"}:
                    score -= 100
            else:
                if m.get("spot"): score += 50

            candidates.append((score, m["symbol"]))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_symbol = candidates[0][1]

    if best_symbol != user_symbol:
        print(f"[MAPEO] {user_symbol} → {best_symbol}")
    return best_symbol

# ---------------- Descarga ---------------- #
def descargar_symbol_tf(ex, symbol, timeframe, since_ms, until_ms, out_path, max_retries=6):
    print(f"\n==> {symbol} | {timeframe} | desde {iso_utc(since_ms)} hasta {iso_utc(until_ms)}")
    ensure_dir(os.path.dirname(out_path))
    tfms = tf_ms(ex, timeframe)

    resume_ms = read_last_timestamp(out_path, tfms)
    if resume_ms and resume_ms > since_ms:
        since_ms = resume_ms
        print(f"[REANUDANDO] Continuando en {iso_utc(since_ms)}")

    all_chunks = []
    last_chunk_last_ms = None

    while since_ms < until_ms:
        time.sleep(ex.rateLimit / 1000.0)

        tries = 0
        ohlcv = None
        while tries < max_retries:
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RateLimitExceeded) as e:
                tries += 1
                wait = min(60, 2 ** tries)
                print(f"[REINTENTO] {e} | Esperando {wait}s... ({tries}/{max_retries})")
                time.sleep(wait)
            except Exception as e:
                print(f"[ERROR] {e}")
                tries = max_retries
                break

        if not ohlcv:
            print("No hay más datos o se alcanzó el límite de reintentos.")
            break

        first_ms_raw = ohlcv[0][0]
        last_ms_raw  = ohlcv[-1][0]

        df_chunk = normalize_df(ohlcv)
        df_chunk = df_chunk.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        prev_last = last_chunk_last_ms if last_chunk_last_ms is not None else read_last_timestamp(out_path, tfms)
        log_gap(prev_last, first_ms_raw, tfms, symbol, timeframe)
        last_chunk_last_ms = last_ms_raw

        all_chunks.append(df_chunk)
        print(f"  [+] Chunk {len(df_chunk):4d} velas | hasta {iso_utc(last_ms_raw)}")

        since_ms = last_ms_raw + tfms

    if all_chunks:
        df = pd.concat(all_chunks).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df.to_csv(out_path, index=False)
        print(f"[OK] {symbol} {timeframe}: {len(df)} velas guardadas en -> {out_path}")
    else:
        print(f"[AVISO] No se guardó nada para {symbol} {timeframe}")

# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser(description="Descarga OHLCV desde Binance (spot/usdm), reanudable y multi-activos.")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,BNB/USDT,XRP/USDT",
                        help="Símbolos separados por coma (ej: BTC/USDT,ETH/USDT)")
    parser.add_argument("--timeframes", type=str, default="5m,1h",
                        help="Timeframes separados por coma (ej: 1h,15m)")
    parser.add_argument("--since", type=str, default="2023-01-01T00:00:00Z",
                        help="Fecha inicio ISO (UTC)")
    parser.add_argument("--until", type=str, default="now",
                        help="Fecha fin ISO (UTC) o 'now'")
    parser.add_argument("--market", type=str, default="usdm", choices=["usdm","spot"],
                        help="Mercado: usdm (futuros perpetuos) o spot")
    parser.add_argument("--outdir", type=str, default=DATA_FOLDER,
                        help="Carpeta destino de CSVs")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs     = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    ex = build_exchange(args.market)
    since_ms = ex.parse8601(args.since)
    until_ms = ex.milliseconds() if args.until.lower() == "now" else ex.parse8601(args.until)

    ensure_dir(args.outdir)

    for user_sym in symbols:
        resolved = resolve_symbol(ex, user_sym, args.market)
        if not resolved:
            print(f"[AVISO] Símbolo no válido en {args.market}: {user_sym} (omitido)")
            continue
        for tf in tfs:
            if tf not in ex.timeframes:
                print(f"[AVISO] Timeframe no soportado por exchange: {tf} (omitido)")
                continue
            filename = f"{resolved.replace('/','')}_{tf}_{args.market}.csv"
            out_path = os.path.join(args.outdir, filename)
            descargar_symbol_tf(ex, resolved, tf, since_ms, until_ms, out_path)

if __name__ == "__main__":
    main()

