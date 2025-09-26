# descargar_datos.py — versión robusta (USDM/Spot), sin CSV de 0 KB, reanudable
import argparse
import os
import time
import datetime as dt
import pandas as pd
import ccxt

DATA_FOLDER = "data"  # podés cambiar con --outdir

# ---------- Utilidades ----------
def iso_utc(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_any_date(s: str, ex: ccxt.Exchange) -> int:
    if str(s).lower() == "now":
        return ex.milliseconds()
    try:
        ms = ex.parse8601(s)
        if ms is not None:
            return ms
    except Exception:
        pass
    return int(pd.to_datetime(s, utc=True).timestamp() * 1000)

def timeframe_ms(ex: ccxt.Exchange, timeframe: str) -> int:
    return ex.parse_timeframe(timeframe) * 1000

def normalize_df(ohlcv):
    # ccxt: [ms, open, high, low, close, volume]
    df = pd.DataFrame(ohlcv, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]

def read_last_timestamp(filepath: str) -> int | None:
    """Devuelve el último timestamp en ms del CSV (o None si no aplica)."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None
    try:
        tail = pd.read_csv(filepath, usecols=["timestamp"]).tail(1)
        if tail.empty:
            return None
        last_iso = str(tail["timestamp"].iloc[0])
        return int(pd.Timestamp(last_iso, tz="UTC").timestamp() * 1000)
    except Exception:
        return None

def log_gap(prev_last_ms, new_first_ms, tf_ms, sym, tf):
    if prev_last_ms is None:
        return
    if new_first_ms > prev_last_ms + tf_ms:
        miss = (new_first_ms - prev_last_ms) // tf_ms
        print(f"[GAP] {sym} {tf}: faltan ~{int(miss)} velas entre {iso_utc(prev_last_ms)} y {iso_utc(new_first_ms)}")

# ---------- Exchange ----------
def build_exchange(market: str) -> ccxt.Exchange:
    cfg = {"enableRateLimit": True, "timeout": 30000}
    market = market.lower().strip()
    if market == "usdm":
        ex = ccxt.binanceusdm(cfg)
    elif market == "spot":
        ex = ccxt.binance(cfg)
    else:
        raise ValueError("El mercado debe ser 'usdm' o 'spot'")
    print("Cargando mercados, puede tardar un momento...")
    ex.load_markets()
    print("Mercados cargados.")
    return ex

# ---------- Símbolos ----------
def normalize_user_symbol_for_market(user_sym: str, market: str) -> str:
    """
    USDM: si el usuario pasa 'BTC/USDT', agregamos ':USDT' -> 'BTC/USDT:USDT'.
    Spot: lo dejamos tal cual.
    """
    s = user_sym.upper().strip()
    if market == "usdm" and ":USDT" not in s:
        s = s + ":USDT"
    return s

def resolve_symbol(ex: ccxt.Exchange, user_symbol: str, market: str) -> str | None:
    """
    Devuelve el símbolo exacto reconocido por ccxt.
    En USDM prioriza swaps PERPETUAL lineales en USDT.
    """
    s = normalize_user_symbol_for_market(user_symbol, market)
    if s in ex.symbols:
        return s

    base, quote = (None, None)
    if "/" in s:
        base, quote = s.split("/", 1)
    quote_clean = "USDT"
    candidates = []
    for m in ex.markets.values():
        try:
            if base and m.get("base") != base:
                continue
            if quote and m.get("quote") != quote_clean:
                continue
            if market == "usdm" and m.get("type") != "swap":
                continue
            if market == "spot" and m.get("type") != "spot":
                continue

            score = 0
            if market == "usdm":
                if m.get("linear"): score += 10
                ct = (m.get("info", {}) or {}).get("contractType") or m.get("contractType")
                if ct == "PERPETUAL": score += 100
                if ct in {"CURRENT_QUARTER", "NEXT_QUARTER"}: score -= 50
            else:
                if m.get("spot"): score += 10
            candidates.append((score, m["symbol"]))
        except Exception:
            continue

    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0][1]
    if best != s:
        print(f"[MAPEO] {user_symbol} → {best}")
    return best

# ---------- Descarga ----------
def fetch_all_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int,
                    max_retries: int = 6) -> pd.DataFrame:
    tfms = timeframe_ms(ex, timeframe)
    cursor = since_ms
    all_chunks = []
    last_chunk_last_ms = None

    while cursor < until_ms:
        time.sleep(ex.rateLimit / 1000.0)
        tries = 0
        ohlcv = None
        while tries < max_retries:
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
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
            print("No hay más datos o límite de reintentos alcanzado.")
            break

        first_ms_raw = ohlcv[0][0]
        last_ms_raw  = ohlcv[-1][0]

        df_chunk = normalize_df(ohlcv)
        df_chunk = df_chunk.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        log_gap(last_chunk_last_ms, first_ms_raw, tfms, symbol, timeframe)
        last_chunk_last_ms = last_ms_raw

        all_chunks.append(df_chunk)
        print(f"  [+] {symbol} {timeframe}: chunk {len(df_chunk):4d} velas | hasta {iso_utc(last_ms_raw)}")

        next_cursor = last_ms_raw + tfms
        if next_cursor <= cursor:  # seguridad ante velas idénticas
            next_cursor = cursor + tfms
        cursor = next_cursor

    if all_chunks:
        df = pd.concat(all_chunks, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return df
    else:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

def descargar_symbol_tf(ex: ccxt.Exchange, symbol: str, timeframe: str,
                        since_ms: int, until_ms: int, out_path: str):
    print(f"\n==> {symbol} | {timeframe} | desde {iso_utc(since_ms)} hasta {iso_utc(until_ms)}")
    ensure_dir(os.path.dirname(out_path))

    # reanudación si el archivo ya existe
    last_ms = read_last_timestamp(out_path)
    if last_ms is not None:
        tfms = timeframe_ms(ex, timeframe)
        since_ms = max(since_ms, last_ms + tfms)
        print(f"[REANUDANDO] Continuando en {iso_utc(since_ms)}")

    df_new = fetch_all_ohlcv(ex, symbol, timeframe, since_ms, until_ms)
    if df_new.empty:
        print(f"[AVISO] No se guardó nada para {symbol} {timeframe}")
        return

    # merge con histórico existente para evitar duplicados y reescribir ordenado
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            prev = pd.read_csv(out_path)
            df_new = pd.concat([prev, df_new], ignore_index=True)
            df_new = df_new.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        except Exception:
            pass

    df_new.to_csv(out_path, index=False)
    print(f"[OK] {symbol} {timeframe}: {len(df_new)} velas guardadas en -> {out_path}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Descarga OHLCV desde Binance (spot/usdm), reanudable y multi-activos.")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,BNB/USDT,XRP/USDT",
                        help="Símbolos separados por coma (ej: BTC/USDT,ETH/USDT)")
    parser.add_argument("--timeframes", type=str, default="5m,1h",
                        help="Timeframes separados por coma (ej: 5m,1h)")
    parser.add_argument("--since", type=str, default="2023-01-01T00:00:00Z",
                        help="Fecha inicio ISO (UTC) o legible por pandas")
    parser.add_argument("--until", type=str, default="now",
                        help="Fecha fin ISO (UTC) o 'now'")
    parser.add_argument("--market", type=str, default="usdm", choices=["usdm", "spot"],
                        help="Mercado: usdm (perpetuos USDT) o spot")
    parser.add_argument("--outdir", type=str, default=os.path.join(DATA_FOLDER, "hist"),
                        help="Carpeta destino de CSVs")
    args = parser.parse_args()

    ex = build_exchange(args.market)

    symbols_in = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()]
    timeframes = [t.strip() for t in (args.timeframes or "").split(",") if t.strip()]

    since_ms = parse_any_date(args.since, ex)
    until_ms = parse_any_date(args.until, ex)

    ensure_dir(args.outdir)

    for user_sym in symbols_in:
        resolved = resolve_symbol(ex, user_sym, args.market)
        if not resolved:
            print(f"[AVISO] Símbolo no válido en {args.market}: {user_sym} (omitido)")
            continue

        for tf in timeframes:
            if tf not in ex.timeframes:
                print(f"[AVISO] Timeframe no soportado: {tf} (omitido)")
                continue

            filename = f"{resolved.replace('/','').replace(':','')}_{tf}_{args.market}.csv"
            out_path = os.path.join(args.outdir, filename)
            descargar_symbol_tf(ex, resolved, tf, since_ms, until_ms, out_path)

if __name__ == "__main__":
    main()
