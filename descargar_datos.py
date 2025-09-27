# descargar_datos.py — robusto (USDM/Spot), sin CSV 0 KB, reanudable
import argparse
import os
import time
import datetime as dt
import pandas as pd
import ccxt

DATA_FOLDER = "data"

# ---------- Utilidades ----------
def iso_utc(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def parse_any_date(s: str, ex) -> int:
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
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 32:
        return None
    try:
        # leer última línea rápidamente
        with open(filepath, "rb") as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode("utf-8", errors="ignore").strip()
        last_iso = last_line.split(",")[0]
        return int(pd.Timestamp(last_iso, tz="UTC").timestamp() * 1000)
    except Exception:
        return None

# ---------- Exchange ----------
def build_exchange(market: str):
    market = (market or "usdm").lower().strip()
    cfg = {"enableRateLimit": True, "timeout": 30000}
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
    Normaliza entrada del usuario.
    - usdm: preferir strings tipo 'BTC/USDT:USDT' (perp lineal USDT).
    - spot: 'BTC/USDT'.
    """
    s = user_sym.strip().upper()
    if market == "usdm" and ":USDT" not in s:
        # permitir 'BTC/USDT' y mapear a perp USDT
        if "/" in s and s.endswith("/USDT"):
            s = s + ":USDT"
    return s

def resolve_symbol(ex: ccxt.Exchange, user_sym: str, market: str) -> str | None:
    """
    Busca el símbolo correcto en el exchange.
    - usdm: prioriza swaps PERPETUAL, lineales en USDT.
    - spot: mercados spot.
    """
    s = normalize_user_symbol_for_market(user_sym, market)
    base = quote = None
    if "/" in s:
        base, quote_part = s.split("/", 1)
        quote = quote_part.split(":")[0]

    target_types = {"spot"} if market == "spot" else {"swap", "future"}
    best = None
    best_score = -1

    for m in ex.markets.values():
        try:
            if base and m.get("base") != base:
                continue
            if quote and m.get("quote") != quote:
                continue
            if m.get("type") not in target_types:
                continue

            score = 0
            if market == "usdm":
                if m.get("type") == "swap":
                    score += 50
                if m.get("linear"):
                    score += 10
                ct = (m.get("info", {}) or {}).get("contractType") or m.get("contractType")
                if ct == "PERPETUAL":
                    score += 100
                if ct in {"CURRENT_QUARTER", "NEXT_QUARTER"}:
                    score -= 100
            else:
                if m.get("spot"):
                    score += 50

            if score > best_score:
                best_score = score
                best = m.get("symbol")
        except Exception:
            continue

    if best and best != s:
        print(f"[MAPEO] {s} → {best}")
    return best

# ---------- Descarga ----------
def descargar_symbol_tf(ex: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, until_ms: int, out_path: str):
    print(f"\n==> {symbol} | {timeframe} | desde {iso_utc(since_ms)} hasta {iso_utc(until_ms)}")
    ensure_dir(os.path.dirname(out_path))
    tfms = timeframe_ms(ex, timeframe)

    resume_from = read_last_timestamp(out_path)
    if resume_from and resume_from > since_ms:
        since_ms = resume_from + tfms
        print(f"[REANUDANDO] Continuando en {iso_utc(since_ms)}")

    header_written = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    total_rows = 0

    import csv
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not header_written:
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            header_written = True

        while since_ms < until_ms:
            # rate limit
            time.sleep(ex.rateLimit / 1000.0)
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
            except Exception as e:
                print(f"[ERROR] {e} | reintento en 20s")
                time.sleep(20)
                continue

            if not ohlcv:
                print("No hay más datos.")
                break

            first_ms = ohlcv[0][0]
            last_ms = ohlcv[-1][0]
            df_chunk = normalize_df(ohlcv)
            df_chunk = df_chunk.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

            for _, row in df_chunk.iterrows():
                writer.writerow(row.tolist())

            total_rows += len(df_chunk)
            print(f"  [+] Guardadas {len(df_chunk):4d} velas | Total {total_rows:7d} | hasta {iso_utc(last_ms)}")

            since_ms = last_ms + tfms

    print(f"[OK] {symbol} {timeframe}: {total_rows} velas -> {out_path}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Descarga OHLCV desde Binance (usdm/spot), reanudable.")
    ap.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,BNB/USDT,XRP/USDT")
    ap.add_argument("--timeframes", type=str, default="1h,4h")
    ap.add_argument("--since", type=str, default="2023-01-01T00:00:00Z")
    ap.add_argument("--until", type=str, default="now")
    ap.add_argument("--market", type=str, default="usdm", choices=["usdm", "spot"])
    ap.add_argument("--outdir", type=str, default=DATA_FOLDER)
    args = ap.parse_args()

    ex = build_exchange(args.market)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    since_ms = parse_any_date(args.since, ex)
    until_ms = parse_any_date(args.until, ex)

    ensure_dir(args.outdir)

    for user_sym in symbols:
        resolved = resolve_symbol(ex, user_sym, args.market)
        if not resolved:
            print(f"[AVISO] Símbolo no válido en {args.market}: {user_sym} (omitido)")
            continue
        for tf in tfs:
            if tf not in ex.timeframes:
                print(f"[AVISO] Timeframe no soportado: {tf} (omitido)")
                continue
            fname = f"{resolved.replace('/','').replace(':','_')}_{tf}_{args.market}.csv"
            out_path = os.path.join(args.outdir, fname)
            descargar_symbol_tf(ex, resolved, tf, since_ms, until_ms, out_path)

if __name__ == "__main__":
    main()

