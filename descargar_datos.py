# descargar_datos.py (Versión Corregida y Final)
import argparse
import os
import time
import datetime as dt
import pandas as pd
import ccxt

DATA_FOLDER = "data"

# --- Utilidades ---
def iso_utc(ms):
    return dt.datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%dT%H:%M:%SZ")

def tf_ms(exchange, timeframe: str) -> int:
    return exchange.parse_timeframe(timeframe) * 1000

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def build_exchange(market: str):
    market = market.lower().strip()
    config = {
        "enableRateLimit": True,
        "timeout": 30000,  # Timeout de 30 segundos
    }
    if market == "usdm":
        ex = ccxt.binanceusdm(config)
    elif market == "spot":
        ex = ccxt.binance(config)
    else:
        raise ValueError("El mercado debe ser 'usdm' o 'spot'")
    
    print("Cargando mercados, puede tardar un momento...")
    ex.load_markets()
    print("Mercados cargados.")
    return ex

def symbol_ok(ex, sym):
    # LA FORMA CORRECTA Y SIMPLE DE VERIFICAR
    return sym in ex.symbols

def normalize_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["timestamp_ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df[["timestamp","open","high","low","close","volume"]]
    return df

def read_last_timestamp(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 50: # Si es muy chico, probablemente solo tiene header
        return None
    try:
        with open(filepath, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        
        last_iso = last_line.split(',')[0]
        last_ms = int(pd.Timestamp(last_iso).tz_localize("UTC").timestamp() * 1000)
        return last_ms
    except Exception:
        return None

# --- Lógica de Descarga ---
def descargar_symbol_tf(ex, symbol, timeframe, since_ms, until_ms, out_path, max_retries=6):
    print(f"\n==> {symbol} | {timeframe} | desde {iso_utc(since_ms)} hasta {iso_utc(until_ms)}")
    ensure_dir(os.path.dirname(out_path))
    tfms = tf_ms(ex, timeframe)

    last_ts_ms = read_last_timestamp(out_path)
    if last_ts_ms and last_ts_ms > since_ms:
        since_ms = last_ts_ms + tfms
        print(f"[REANUDANDO] Continuando descarga en {iso_utc(since_ms)}")

    header_needed = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    
    while since_ms < until_ms:
        try:
            time.sleep(ex.rateLimit / 1000.0) # Respetar rate limit
            
            ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
            if not ohlcv:
                print("No hay más datos.")
                break

            df_chunk = normalize_df(ohlcv)
            
            df_chunk.to_csv(out_path, mode='a', index=False, header=header_needed)
            header_needed = False # Solo escribimos el encabezado una vez
            
            total_velas = len(df_chunk)
            last_ms_in_chunk = ohlcv[-1][0]
            print(f"  [+] Guardadas {total_velas} velas | hasta {iso_utc(last_ms_in_chunk)}")
            
            since_ms = last_ms_in_chunk + tfms

        except Exception as e:
            print(f"[ERROR] Ocurrió un error: {e}. Reintentando en 20 segundos...")
            time.sleep(20)

    print(f"[OK] Descarga finalizada para {symbol} {timeframe}")

# --- Ejecución Principal ---
def main():
    parser = argparse.ArgumentParser(description="Descarga OHLCV desde Binance (spot/usdm), reanudable y multi-activos.")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,XRP/USDT,BNB/USDT", help="Símbolos separados por coma")
    parser.add_argument("--timeframes", type=str, default="1h,4h,5m", help="Timeframes separados por coma")
    parser.add_argument("--since", type=str, default="2023-01-01T00:00:00Z", help="Fecha inicio ISO (UTC)")
    parser.add_argument("--market", type=str, default="usdm", choices=["usdm","spot"], help="Mercado: usdm o spot")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    ex = build_exchange(args.market)
    since_ms = ex.parse8601(args.since)
    until_ms = ex.milliseconds()

    ensure_dir(DATA_FOLDER)

    for sym in symbols:
        if not symbol_ok(ex, sym):
            print(f"[AVISO] Símbolo no válido en {args.market}: {sym} (será omitido)")
            continue
        for tf in tfs:
            if tf not in ex.timeframes:
                print(f"[AVISO] Timeframe no soportado por exchange: {tf} (será omitido)")
                continue
            
            filename = f"{sym.replace('/','')}_{tf}_{args.market}.csv"
            out_path = os.path.join(DATA_FOLDER, filename)
            descargar_symbol_tf(ex, sym, tf, since_ms, until_ms, out_path)

if __name__ == "__main__":
    main()
