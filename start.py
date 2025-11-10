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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.default_msec_format = "%s,%03d"

# Asegurar import relativo al proyecto
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logging_setup import setup_logging
from config import load_raw_config, get_telegram_token, get_telegram_chat_id
from bot.engine import TradingApp
from bot.mode_manager import ensure_startup_mode
from paths import get_data_dir


LOCK_PATH = Path("/tmp/chauletbot.lock")


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

    setup_logging()
    logging.info(
        "Iniciando bot en modo: %s (fuente=%s, persistido=%s)",
        "REAL" if effective_mode == "real" else "SIMULADO",
        mode_source,
        persisted,
    )
    logging.info("DATA_DIR=%s", get_data_dir())

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
