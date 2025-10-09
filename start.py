import os
import asyncio
import copy
import yaml

# >>> MEJORA: Cargar .env lo antes posible (con fallback manual)
try:
    from dotenv import load_dotenv, find_dotenv  # requiere python-dotenv
    _envp = find_dotenv(usecwd=True)
    load_dotenv(_envp or ".env")
    print(f".env loaded: {(_envp or '.env')}")
except Exception as e:
    # Fallback manual si la lib no está disponible o falló la import/carga
    try:
        import pathlib
        p = pathlib.Path(".env")
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            print("Loaded .env via manual fallback")
        else:
            print("No .env file found")
    except Exception:
        # Si incluso el fallback falla, seguimos sin romper el arranque
        pass

from bot.engine import TradingApp

# ──────────────────────────────────────────────────────────────────────────────
# Normalizador 1:1 con el simulador_final
# - symbols -> lista directa y BTC only
# - strategy.entry_mode (legacy | rsi_cross | pullback_grid)
# - mapea limits.* a los nombres que usa el engine
# - conserva indicadores 4h y trailing
# - fija timeframe=2m y loop_seconds=120 como en el sim
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg or {})

    # 1) Symbols → lista directa. Si venía en {whitelist: [...]}, lo aplanamos.
    syms = cfg.get("symbols") or []
    if isinstance(syms, dict) and "whitelist" in syms:
        syms = syms["whitelist"]
    if not isinstance(syms, (list, tuple)):
        syms = [str(syms)] if syms else []

    # Fuerza BTC only (como el simulador)
    cfg["symbols"] = ["BTC/USDT:USDT"]

    # 2) Estrategia / modo de entrada del simulador
    strat = cfg.get("strategy") or {}
    # Si no está, por defecto usamos rsi_cross (cambialo en tu YAML si querés legacy o pullback_grid)
    strat.setdefault("entry_mode", "rsi_cross")
    # Mantiene stop/tp y filtros del simulador si ya estaban en el YAML
    # (rsi4h_gate, ema200_1h_confirm, atrp_gate_min/max, ban_hours, grid_*, etc.)
    cfg["strategy"] = strat

    # 3) Limits: renombrar a los que el engine espera
    limits = cfg.get("limits") or {}
    limits["max_total_positions"] = int(limits.get("max_total_positions", limits.get("max_open_total", 6)))
    limits["max_per_symbol"]      = int(limits.get("max_per_symbol", limits.get("max_open_per_symbol", 4)))
    limits["no_hedge"]            = bool(limits.get("no_hedge", True))
    # cooldown_seconds se respeta si está
    if "cooldown_seconds" in limits:
        limits["cooldown_seconds"] = int(limits["cooldown_seconds"])
    cfg["limits"] = limits

    # 4) Caps de cartera (engine usa dataclass Caps)
    pc = cfg.get("portfolio_caps") or {}
    pc.setdefault("max_portfolio_leverage", 8.0)
    pc.setdefault("max_portfolio_margin_pct", 1.0)
    cfg["portfolio_caps"] = pc

    # 5) Indicadores: asegurar claves de 4h que usa compute_indicators
    inds = cfg.get("indicators") or {}
    inds.setdefault("ema200_4h_window", 200)
    inds.setdefault("rsi4h_len", 14)
    cfg["indicators"] = inds

    # 6) Trailing: el engine acepta strategy.trailing o top-level trailing
    # No forzamos nada si ya lo definiste en el YAML.

    # 7) Loop & timeframe como en el simulador
    cfg.setdefault("timeframe", "2m")
    cfg.setdefault("loop_seconds", 120)

    # 8) Modo de ejecución
    cfg["mode"] = str(cfg.get("mode", "paper")).lower()

    # 9) BTC only también en guards de correlación si estuvieran definidos
    cg = cfg.get("correlation_guard") or {}
    if cg:
        cg["enabled"] = bool(cg.get("enabled", True))
        if "clusters" in cg:
            cg["clusters"] = [["BTC/USDT:USDT"]]  # una sola canasta con BTC
        cfg["correlation_guard"] = cg

    return cfg


async def main():
    # Podés cambiar el nombre de archivo vía env CONFIG_FILE
    path = os.environ.get("CONFIG_FILE", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = _normalize_config(raw)

    # Log rápido para confirmar arranque (opcional)
    print("=== START ===")
    print("Symbols:", cfg.get("symbols"))
    print("Entry mode:", (cfg.get("strategy") or {}).get("entry_mode"))
    print("Timeframe:", cfg.get("timeframe"), "| Loop(s):", cfg.get("loop_seconds"))
    print("Mode:", cfg.get("mode"))

    app = TradingApp(cfg)
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())
