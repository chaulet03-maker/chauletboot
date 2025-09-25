
import os, re, sys, time, shutil, textwrap, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(p): return Path(p).read_text(encoding="utf-8", errors="ignore")
def write(p,s): Path(p).write_text(s, encoding="utf-8")

def backup(p):
    src = Path(p)
    if not src.exists(): return
    ts = time.strftime("%Y%m%d-%H%M%S")
    dst = src.with_suffix(src.suffix + f".bak-{ts}")
    shutil.copy2(src, dst)
    print(f"[backup] {src} -> {dst.name}")

def fix_engine(bot_dir: Path):
    p = bot_dir/"engine.py"
    print(f"[engine] Patching {p}")
    backup(p)
    s = read(p)

    # Remove broken leftover line
    s = re.sub(r"^\s*task_bot\s*=\s*#.*$", "# removed leftover invalid line", s, flags=re.MULTILINE)

    # Remove create_task(run_polling(...))
    s = re.sub(r"asyncio\.create_task\s*\(\s*app\.run_polling[^\)]*\)\s*", "# removed create_task(run_polling) (PTB v20 fix)", s)

    # Comment any app.run_polling( occurrences
    if "app.run_polling(" in s:
        s = s.replace("app.run_polling(", "# PTB v20 fix: replaced by async start/stop\n# app.run_polling(")

    # Ensure imports
    s = re.sub(r"^\s*import asyncio, json, traceback, math\b",
               "import asyncio, json, traceback, math, os",
               s, flags=re.M)

    # Ensure notify is imported when build_app is imported
    s = s.replace(
        "from bot.telemetry.telegram_bot import build_app",
        "from bot.telemetry.telegram_bot import build_app, notify"
    )

    # Insert a PTB v20 async start/stop block after app = build_app()
    if "PTB v20 START BLOCK" not in s:
        s = s.replace(
            "app = build_app()",
            textwrap.dedent("""
            app = build_app()
            # === PTB v20 START BLOCK ===
            try:
                if app:
                    await app.initialize()
                    await app.start()
                    if getattr(app, "updater", None):
                        await app.updater.start_polling()
            except Exception as e:
                try:
                    log_exception("telegram_startup_error", str(e))
                except Exception:
                    pass
                app = None
            # === PTB v20 START BLOCK END ===
            """ ).strip()
        )

    # Add a safe stop in finally or after loop
    if "PTB v20 STOP BLOCK" not in s:
        if "finally:" in s and "updater.stop()" not in s:
            s = s.replace(
                "finally:",
                textwrap.dedent("""
                finally:
                    # === PTB v20 STOP BLOCK ===
                    try:
                        if app and getattr(app, "updater", None):
                            await app.updater.stop()
                        if app:
                            await app.stop()
                            await app.shutdown()
                    except Exception as e:
                        try:
                            log_exception("telegram_shutdown_error", str(e))
                        except Exception:
                            pass

                """ ).rstrip()
            )
        elif "await loop(" in s and "updater.stop()" not in s:
            s = s.replace(
                "await loop(app, trader, exchange)",
                textwrap.dedent("""
                await loop(app, trader, exchange)
                # === PTB v20 STOP BLOCK ===
                try:
                    if app and getattr(app, "updater", None):
                        await app.updater.stop()
                    if app:
                        await app.stop()
                        await app.shutdown()
                except Exception as e:
                    try:
                        log_exception("telegram_shutdown_error", str(e))
                    except Exception:
                        pass
                """ ).rstrip()
            )

    write(p, s)

def fix_telegram(bot_dir: Path):
    p = bot_dir/"telemetry"/"telegram_bot.py"
    if not p.exists():
        print("[telegram] No telemetry/telegram_bot.py found, skipping")
        return
    print(f"[telegram] Patching {p}")
    backup(p)
    s = read(p)

    # Ensure CommandHandler import
    s = s.replace(
        "from telegram.ext import Application, MessageHandler, filters, ContextTypes",
        "from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes"
    )

    # Chunk helper
    if "def _chunk_and_send(" not in s:
        s = s.replace("from bot.state import enqueue_cmd",
                      "from bot.state import enqueue_cmd\n\nMAX_TG=3900\nasync def _chunk_and_send(bot, chat_id, text):\n    if not text:\n        return\n    for i in range(0, len(text), MAX_TG):\n        await bot.send_message(chat_id=int(chat_id), text=text[i:i+MAX_TG])\n")

    # reply() uses chunking (cover common patterns)
    s = re.sub(
        r"async def reply\([^\)]*\):[\s\S]*?reply_text\(t\)[\s\S]*?except Exception as e:[\s\S]*?log_exception\([^\)]*\)\n",
        "async def reply(t):\n            try:\n                if chat_id:\n                    await _chunk_and_send(context.bot, chat_id, t)\n            except Exception as e:\n                log_exception('reply_error', str(e))\n",
        s
    )
    s = s.replace("await context.bot.send_message(chat_id=int(chat_id), text=msg)",
                  "await _chunk_and_send(context.bot, chat_id, msg)")

    # notify() helper if missing
    if "def notify(" not in s:
        s += "\nasync def notify(app, text: str):\n    try:\n        if TELEGRAM_CHAT_ID:\n            await _chunk_and_send(app.bot, TELEGRAM_CHAT_ID, text)\n    except Exception as e:\n        log_exception('notify_error', str(e))\n"

    # Accept commands + text
    s = re.sub(
        r"app\.add_handler\(MessageHandler\([^)]*\)\)",
        "app.add_handler(CommandHandler(['start','help','status','precio','pausa','pause','on','off','bot','saldo','riesgo'], on_text))\n    app.add_handler(MessageHandler(filters.TEXT | filters.COMMAND, on_text))",
        s
    )

    # Token missing -> explicit log
    s = s.replace("if not TELEGRAM_BOT_TOKEN:\n        return None",
                  "if not TELEGRAM_BOT_TOKEN:\n        BOT_LOGGER.error('TELEGRAM_BOT_TOKEN vacío; Telegram deshabilitado.'); return None")

    write(p, s)

def fix_trader(bot_dir: Path):
    p = bot_dir/"execution"/"trader.py"
    if not p.exists():
        print("[trader] No execution/trader.py found, skipping")
        return
    print(f"[trader] Patching {p}")
    backup(p)
    s = read(p)

    # Ensure BOT_LOGGER import present
    if "BOT_LOGGER" not in s and "from bot.logger import" in s:
        s = s.replace("from bot.logger import decision_event, log_exception",
                      "from bot.logger import decision_event, log_exception, BOT_LOGGER")

    # Replace equity() body
    s = re.sub(
        r"def equity\(self\)[\s\S]*?return [^\n]*\n\s*\n",
        """def equity(self) -> float:
        \"\"\"Equity USDT compatible con PaperExchange y ccxt (binanceusdm/spot).\"\"\"
        try:
            if hasattr(self.ex, "fetch_balance_usdt"):
                BOT_LOGGER.info("equity(): using wrapper fetch_balance_usdt")
                bal = self.ex.fetch_balance_usdt()
                try: return float(bal)
                except Exception:
                    try: return float(bal.get("total", 0))
                    except Exception: pass
        except Exception as e:
            log_exception("equity_wrapper_error", str(e))
        try:
            BOT_LOGGER.info("equity(): using ccxt fetch_balance")
            bal = self.ex.fetch_balance()
            if isinstance(bal, dict):
                if "USDT" in bal and isinstance(bal["USDT"], dict):
                    usdt = bal["USDT"]
                    if "total" in usdt: return float(usdt["total"])
                    if "free" in usdt:  return float(usdt["free"])
                if "total" in bal and isinstance(bal["total"], dict) and "USDT" in bal["total"]:
                    return float(bal["total"]["USDT"])
                if "free" in bal and isinstance(bal["free"], dict) and "USDT" in bal["free"]:
                    return float(bal["free"])
                info = bal.get("info") or {}
                for k in ("totalWalletBalance","walletBalance","availableBalance","crossWalletBalance"):
                    if k in info:
                        try: return float(k)
                        except Exception: pass
            return float(bal)
        except Exception as e:
            log_exception("equity_ccxt_error", str(e))
            return 0.0

\n""",
        s, count=1
    )

    write(p, s)

def syntax_check(root):
    errs = []
    for r, d, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(r, f)
                try:
                    src = Path(p).read_text(encoding="utf-8", errors="ignore")
                    compile(src, p, "exec")
                except SyntaxError as se:
                    errs.append((p, se.lineno, se.offset, se.msg))
                except Exception:
                    pass
    return errs

def main():
    # Must run from project root (directory that contains 'bot')
    if not (ROOT/"bot").exists():
        print("ERROR: corré este script en la RAÍZ del proyecto (donde está la carpeta 'bot').")
        sys.exit(1)

    bot_dir = ROOT/"bot"
    fix_engine(bot_dir)
    fix_telegram(bot_dir)
    fix_trader(bot_dir)

    errs = syntax_check(ROOT)
    if errs:
        print("\n[SYNTAX] Errores encontrados:")
        for f, ln, col, msg in errs:
            print(f" - {f}:{ln}:{col} -> {msg}")
        sys.exit(2)
    print("\n[SYNTAX] OK: no hay errores de sintaxis en .py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        sys.exit(3)
