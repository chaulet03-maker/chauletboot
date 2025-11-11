# start.py (reemplazar completo)
import argparse
import asyncio
import os
import sys
import logging
import atexit
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Asegurar import relativo al proyecto
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logging_setup import setup_logging

# Inicializamos logging de manera centralizada (rota archivos, filtra ruido, etc.)
setup_logging()

from config import load_raw_config, get_telegram_token, get_telegram_chat_id
from bot.engine import TradingApp
from bot.mode_manager import ensure_startup_mode
from paths import get_data_dir
from bot.async_state_store import AsyncRedisStateStore
from state_store import configure_async_state_store


LOCK_PATH = Path("/tmp/chauletbot.lock")
_STATE_STORE_INSTANCE: AsyncRedisStateStore | None = None


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        with lock_path.open("r", encoding="utf-8") as lock_file:
            content = lock_file.read().strip()
    except OSError:
        return None

    if not content:
        return None

    try:
        return int(content)
    except ValueError:
        return None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but belongs to another user.
        return True
    except OSError:
        return False
    else:
        return True


def _acquire_single_instance_lock() -> None:
    current_pid = os.getpid()

    if LOCK_PATH.exists():
        existing_pid = _read_lock_pid(LOCK_PATH)

        if existing_pid and existing_pid != current_pid and _pid_is_running(existing_pid):
            print(
                f"[LOCK] Ya hay un bot ejecutándose (PID={existing_pid}). Cerralo antes de iniciar otro.",
                file=sys.stderr,
            )
            sys.exit(1)

        # El lock es inválido o huérfano, intentamos eliminarlo.
        try:
            LOCK_PATH.unlink()
            logging.warning("Lock huérfano eliminado: %s", LOCK_PATH)
        except FileNotFoundError:
            pass
        except PermissionError as exc:
            print(
                f"[LOCK] No se pudo limpiar el lock obsoleto ({LOCK_PATH}): {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        except OSError as exc:
            logging.exception("No se pudo limpiar el lock huérfano")
            print(
                f"[LOCK] Error eliminando lock obsoleto ({LOCK_PATH}): {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        LOCK_PATH.write_text(str(current_pid), encoding="utf-8")
    except OSError as exc:
        logging.exception("No se pudo crear el archivo de lock")
        print(
            f"[LOCK] No se pudo crear el archivo de lock ({LOCK_PATH}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    @atexit.register
    def _cleanup_lock() -> None:
        try:
            if LOCK_PATH.exists():
                existing_pid = _read_lock_pid(LOCK_PATH)
                if existing_pid == current_pid:
                    LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logging.exception("No se pudo eliminar el archivo de lock al finalizar")


def _setup_state_store(cfg: dict) -> None:
    global _STATE_STORE_INSTANCE

    cfg_state_store = cfg.get("state_store") if isinstance(cfg, dict) else {}
    cfg_state_store = cfg_state_store if isinstance(cfg_state_store, dict) else {}

    redis_url = os.getenv("REDIS_URL") or str(cfg_state_store.get("redis_url") or "").strip()
    redis_key = os.getenv("BOT_STATE_REDIS_KEY") or cfg_state_store.get("redis_key") or None

    if not redis_url and not os.getenv("REDIS_HOST"):
        configure_async_state_store(None)
        return

    try:
        if redis_url:
            store = AsyncRedisStateStore(url=redis_url)
        else:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            password = os.getenv("REDIS_PASSWORD") or None
            store = AsyncRedisStateStore(
                host=host,
                port=port,
                db=db,
                password=password,
            )
        store.connect_sync(timeout=5.0)
    except Exception:
        logging.exception(
            "No se pudo inicializar la persistencia Redis; se usará almacenamiento local."
        )
        configure_async_state_store(None)
        return

    _STATE_STORE_INSTANCE = store
    if redis_url:
        redis_target = redis_url
    else:
        redis_target = f"{host}:{port}/{db}"
    configure_async_state_store(store, redis_key=redis_key)
    logging.info("Persistencia de estado habilitada con Redis (%s)", redis_target)

    def _close_store() -> None:
        try:
            store.close_sync()
        except Exception:
            logging.debug("No se pudo cerrar la conexión Redis limpiamente.", exc_info=True)

    atexit.register(_close_store)


def main(argv: list[str] | None = None):
    # --- SINGLE INSTANCE LOCK ---
    _acquire_single_instance_lock()

    parser = argparse.ArgumentParser(description="Inicia el bot de trading")
    parser.add_argument(
        "--mode",
        dest="mode",
        help="Forzar modo de ejecución (real/live/paper/simulado)",
    )
    args = parser.parse_args(argv)

    cfg = load_raw_config()

    env_mode = os.getenv("TRADING_MODE") or os.getenv("MODE")
    effective_mode, mode_source, persisted = ensure_startup_mode(
        cfg,
        cli_mode=args.mode,
        env_mode=env_mode,
        persist=True,
    )

    # Normalizamos telegram acá (una sola fuente de verdad)
    cfg["telegram_token"] = get_telegram_token(cfg.get("telegram_token"))
    cfg["telegram_chat_id"] = get_telegram_chat_id(cfg.get("telegram_chat_id"))

    logging.info(
        "Iniciando bot en modo: %s (fuente=%s, persistido=%s)",
        "REAL" if effective_mode == "real" else "SIMULADO",
        mode_source,
        persisted,
    )
    logging.info("DATA_DIR=%s", get_data_dir())

    _setup_state_store(cfg)

    app = TradingApp(cfg, mode_source=mode_source)

    try:
        app.run()
    except Exception:
        logging.exception("Error ejecutando el bot")
        raise
    finally:
        try:
            asyncio.run(app.shutdown())
        except Exception:
            logging.exception("Error durante el shutdown ordenado", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Bot detenido manualmente.")
    except Exception:
        logging.exception("Error fatal en el bot")
        sys.exit(1)
