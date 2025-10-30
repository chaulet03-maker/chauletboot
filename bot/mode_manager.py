from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Tuple

import yaml

from bot.runtime_state import get_mode as runtime_get_mode, set_mode as runtime_set_mode

log = logging.getLogger(__name__)

Mode = Literal["real", "simulado"]

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


@dataclass
class ModeResult:
    ok: bool
    msg: str
    mode: Mode | None = None


def _read_cfg(path: str = CONFIG_PATH) -> dict:
    if not path:
        return {}
    cfg_path = os.path.expanduser(path)
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        log.warning("Config en %s no es un objeto dict.", cfg_path)
        return {}
    return data


def _write_cfg(cfg: dict, path: str = CONFIG_PATH) -> None:
    cfg_path = os.path.expanduser(path)
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.abspath(cfg_path)
    # asegurar carpeta destino
    os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
    # write-then-rename (atómico)
    tmp = cfg_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    os.replace(tmp, cfg_path)


def get_mode() -> Mode:
    # Runtime state tiene prioridad para comandos/manuales.
    runtime_mode = (runtime_get_mode() or "").strip().lower()
    if runtime_mode in {"live", "real"}:
        return "real"
    if runtime_mode in {"paper", "sim", "simulado"}:
        return "simulado"

    cfg = _read_cfg()
    mode = str(cfg.get("trading_mode", "simulado")).lower()
    return "real" if mode == "real" else "simulado"


def set_mode_in_yaml(mode: Mode) -> None:
    cfg = _read_cfg()
    cfg["trading_mode"] = "real" if mode == "real" else "simulado"
    _write_cfg(cfg)
    log.info("Config actualizada: trading_mode=%s", cfg["trading_mode"])
    runtime_set_mode("live" if mode == "real" else "paper")


def _env_key_candidates() -> Tuple[Tuple[str, str], ...]:
    return (
        ("BINANCE_KEY", "BINANCE_SECRET"),
        ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
        ("BINANCE_API_KEY_REAL", "BINANCE_API_SECRET_REAL"),
        ("BINANCE_API_KEY_TEST", "BINANCE_API_SECRET_TEST"),
    )


def check_keys_present(env=os.environ) -> Tuple[bool, str]:
    for key_name, secret_name in _env_key_candidates():
        key = env.get(key_name)
        secret = env.get(secret_name)
        if key and secret:
            if len(str(key)) < 10 or len(str(secret)) < 10:
                return False, f"Credenciales {key_name}/{secret_name} parecen inválidas (muy cortas)."
            return True, f"Usando {key_name}/{secret_name}"
    return False, (
        "Faltan credenciales Binance. Admitidos: "
        "BINANCE_API_KEY/_SECRET, ..._REAL, ..._TEST, BINANCE_KEY/_SECRET."
    )


def safe_switch(new_mode: Mode, services) -> ModeResult:
    """
    services debe exponer:
      - position_status(): dict con {"side": "FLAT|LONG|SHORT", ...} del modo ACTUAL
      - rebuild(mode: Mode): reconstruye BROKER, POSITION_SERVICE, clientes ccxt
    """
    current = get_mode()
    if new_mode == current:
        runtime_set_mode("live" if new_mode == "real" else "paper")
        return ModeResult(True, f"Ya estabas en modo {new_mode}.", new_mode)

    try:
        status = services.position_status()
    except Exception as exc:  # pragma: no cover - defensivo
        log.debug("No se pudo obtener position_status antes de cambiar modo: %s", exc)
        status = None

    warn_msg = ""
    if status:
        side = str(status.get("side", "FLAT")).upper()
        try:
            qty_val = float(status.get("qty") or status.get("pos_qty") or 0.0)
        except Exception:
            qty_val = 0.0
        has_open = side != "FLAT" and abs(qty_val) > 0.0
        if has_open:
            # En ambos sentidos permitimos el cambio de modo y notificamos la situación.
            if new_mode == "real":
                warn_msg = (
                    "⚠️ Se detectó una posición ABIERTA del modo anterior. "
                    "Activé REAL; la posición previa seguirá siendo gestionada en segundo plano "
                    "y no se abrirán nuevas en ese modo."
                )
            else:
                warn_msg = (
                    "⚠️ Se detectó una posición ABIERTA del modo anterior. "
                    "Activé SIMULADO; la posición previa seguirá siendo gestionada en segundo plano "
                    "y no se abrirán nuevas en ese modo."
                )

    if new_mode == "real":
        ok, msg = check_keys_present(env=os.environ)
        if not ok:
            return ModeResult(False, f"No pude cambiar a REAL: {msg}", None)

    try:
        set_mode_in_yaml(new_mode)
        services.rebuild(new_mode)
        try:
            import trading

            trading.force_refresh_clients()
        except Exception:
            pass
        message = f"Modo cambiado a {new_mode.upper()} correctamente."
        if warn_msg:
            message = f"{message}\n{warn_msg}"
        return ModeResult(True, message, new_mode)
    except Exception as exc:  # pragma: no cover - defensivo
        # rollback config/runtime si falló el rebuild
        try:
            set_mode_in_yaml(current)
        except Exception:
            pass
        log.exception("Error al cambiar de modo: %s", exc)
        return ModeResult(False, f"Error al cambiar de modo: {exc}", None)
