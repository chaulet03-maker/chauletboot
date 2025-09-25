import asyncio, logging, os, re, datetime as dt, csv

async def _cmd_help(reply):
    texto = (
        "📖 *Ayuda*\n"
        "• *ayuda*: muestra esta ayuda.\n"
        "• *estado*: saldo, operaciones abiertas y últimos saldos.\n"
        "• *saldo* / *saldo=NUM*: consulta / fija saldo inicial (paper).\n"
        "• *posicion* / *posiciones*: detalle de posiciones abiertas.\n"
        "• *precio [SIMBOLO]*: precio actual.\n"
        "• *bot on* / *bot off*: habilita/deshabilita nuevas entradas.\n"
        "• *kill* / *killswitch*: cierra todas y bloquea nuevas entradas.\n"
        "• *cerrar todo*: cierra todas las posiciones.\n"
        "• *recientes* / *motivos*: últimos motivos de NO-entrada.\n"
        "• *stats* / *stats semana*: PF, winrate, expectancy por símbolo/capa.\n"
        "• *diag on* / *diag off*: activa/desactiva diagnóstico.\n"
    )
    return await reply(texto)

async def _cmd_positions_detail(engine, reply):
    st = getattr(engine.trader, "state", None)
    positions = getattr(st, "positions", {}) if st else {}
    if not positions:
        return await reply("No hay posiciones abiertas.")
    price_cache = getattr(engine, "price_cache", {}) or {}
    lines = []
    for sym, lots in positions.items():
        px_now = price_cache.get(sym)
        try:
            px_now = float(px_now) if px_now is not None else None
        except Exception:
            px_now = None
        for L in lots:
            side = L.get("side","long")
            s_side = "long" if side == "long" else "short"
            lev = int(L.get("lev", 1) or 1)
            entry = float(L.get("entry", 0.0) or 0.0)
            qty = float(L.get("qty", 0.0) or 0.0)
            sl = float(L.get("sl", 0.0) or 0.0)
            tp = float(L.get("tp2", L.get("tp1", 0.0)) or 0.0)
            pnl_abs = 0.0
            pnl_pct = 0.0
            if px_now and entry:
                if side == "long":
                    pnl_abs = (px_now - entry) * qty * max(1, lev)
                    pnl_pct = (px_now / entry - 1.0) * 100.0 * max(1, lev)
                else:
                    pnl_abs = (entry - px_now) * qty * max(1, lev)
                    pnl_pct = (entry / px_now - 1.0) * 100.0 * max(1, lev)
            lines.append(f"{sym} {s_side} x{lev}\nentrada: {entry:.2f}\npnl: {pnl_abs:+.2f} ({pnl_pct:+.2f}%)\nsl: {sl:.2f}\ntp: {tp:.2f}")
            lines.append("")
    return await reply("\n".join(lines).strip())
from telegram.ext import Application, MessageHandler, filters
import unicodedata

log = logging.getLogger("tg")

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    # quitar markdown simple y signos
    s = re.sub(r"[*_`~]+", "", s)
    # colapsar espacios y signos
    s = re.sub(r"[,.;:!?()\[\]{}<>\\|/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # quitar acentos
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def _fmt_money(x):
    try:
        return f"${float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def _status_text(engine):
    # Construir estado en español
    try:
        eq = float(engine.trader.equity())
    except Exception:
        eq = 0.0
    # posiciones abiertas desde el estado vivo
    per_symbol = {s: len(v) for s, v in getattr(engine.trader.state, "positions", {}).items()} if getattr(engine, "trader", None) else {}
    open_cnt = sum(per_symbol.values()) if per_symbol else 0

    # fallback: si no hay nada en memoria, mirar CSV de trades
    try:
        if open_cnt == 0:
            csv_dir = getattr(engine, "csv_dir", "data")
            path = os.path.join(csv_dir, "trades.csv")
            if os.path.exists(path):
                abiertos = {}
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        sym = (r.get("symbol") or "").upper()
                        note = (r.get("note") or "").upper()
                        if not sym:
                            continue
                        if note.startswith("OPEN"):
                            abiertos[sym] = abiertos.get(sym, 0) + 1
                        elif note.startswith("CLOSE") and abiertos.get(sym, 0) > 0:
                            abiertos[sym] -= 1
                            if abiertos[sym] <= 0:
                                abiertos.pop(sym, None)
                per_symbol = {k: v for k, v in abiertos.items() if v > 0}
                open_cnt = sum(per_symbol.values())
    except Exception as e:
        log.warning("No pude leer trades.csv para estado: %s", e)

    # últimos saldos (equity.csv)
    recientes_txt = ""
    try:
        csv_dir = getattr(engine, "csv_dir", "data")
        eq_path = os.path.join(csv_dir, "equity.csv")
        if os.path.exists(eq_path):
            rows = []
            with open(eq_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            ult = rows[-3:] if len(rows) >= 3 else rows[-len(rows):]
            if ult:
                saldos = " → ".join(_fmt_money(r.get("equity", "0")) for r in ult)
                recientes_txt = f"\nSaldos recientes: {saldos}"
    except Exception as e:
        pass

    partes = [f"📊 Estado",
              f"Saldo: {_fmt_money(eq)}",
              f"Operaciones abiertas: {open_cnt}"]
    if per_symbol:
        partes.append("Por símbolo: " + ", ".join(f"{k}: {v}" for k, v in per_symbol.items()))
    partes.append(f"Killswitch: {'ACTIVADO' if getattr(engine.trader.state, 'killswitch', False) else 'desactivado'}")
    if recientes_txt:
        partes.append(recientes_txt)

    return "\n".join(partes)

class CommandBot:
    def __init__(self, app_engine):
        self.engine = app_engine
        # acepta ambos nombres de variable de entorno
        self.token = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")  # opcional

    async def _handle_text(self, update, context):
        msg_raw = (update.message.text or "")
        msg = msg_raw.strip().lower()
        norm_all = _normalize_text(msg_raw)

        def reply(text):
            return update.message.reply_text(text)

        # Normalizar para comandos tipo "Bot, on" / "Bot off"
        norm = re.sub(r"[,.;:!?\s]+", " ", msg).strip()

        # --- BOT ON / OFF (alias killswitch inverso) ---
        if norm in ("bot on", "prender bot", "activar bot", "bot prender"):
            try:
                ks = getattr(self.engine.trader.state, "killswitch", False)
            except Exception:
                ks = False
            desired_ks = False  # ON => permitir nuevas operaciones
            if ks != desired_ks:
                self.engine.toggle_killswitch()
            return await reply("✅ Bot ON: habilitadas nuevas operaciones (killswitch desactivado).")

        if norm in ("bot off", "apagar bot", "desactivar bot", "bot apagar"):
            try:
                ks = getattr(self.engine.trader.state, "killswitch", False)
            except Exception:
                ks = False
            desired_ks = True  # OFF => bloquear nuevas operaciones
            if ks != desired_ks:
                self.engine.toggle_killswitch()
            return await reply("⛔ Bot OFF: bloqueadas nuevas operaciones (killswitch ACTIVADO).")

        # --- POSICION DETALLE ---
        if norm_all in ('posicion','posiciones','position','positions'):
            return await _cmd_positions_detail(self.engine, reply)

        # --- AYUDA ---
        if norm_all in ('ayuda','menu','comandos','help'):
            return await _cmd_help(reply)

        # --- ESTADO ---
        if msg in ("estado", "status"):
            return await reply(_status_text(self.engine))

        # --- SALDO ---
        if msg in ("equity", "saldo"):
            try:
                eq = self.engine.trader.equity()
            except Exception:
                eq = 0.0
            return await reply(f"Saldo: {_fmt_money(eq)}")

        # --- POSICIONES ---
        if msg in ("posicion", "posición", "posiciones"):
            st = getattr(self.engine.trader.state, "positions", {}) if getattr(self.engine, "trader", None) else {}
            if not st:
                # fallback CSV
                csv_dir = getattr(self.engine, "csv_dir", "data")
                path = os.path.join(csv_dir, "trades.csv")
                abiertos = {}
                if os.path.exists(path):
                    try:
                        with open(path, newline="", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for r in reader:
                                sym = (r.get("symbol") or "").upper()
                                note = (r.get("note") or "").upper()
                                if not sym:
                                    continue
                                if note.startswith("OPEN"):
                                    abiertos[sym] = abiertos.get(sym, 0) + 1
                                elif note.startswith("CLOSE") and abiertos.get(sym, 0) > 0:
                                    abiertos[sym] -= 1
                                    if abiertos[sym] <= 0:
                                        abiertos.pop(sym, None)
                        if abiertos:
                            listado = "\n".join(f"• {k}: {v}" for k, v in abiertos.items())
                            return await reply(f"Posiciones abiertas (por símbolo):\n{listado}")
                    except Exception as e:
                        pass
                return await reply("No hay posiciones abiertas.")

            # Hay estado en memoria
            lineas = []
            for sym, lots in st.items():
                lineas.append(f"• {sym}: {len(lots)}")
            return await reply("Posiciones abiertas (por símbolo):\n" + "\n".join(lineas))

        # --- PRECIO ---
        if msg.startswith("precio"):
            parts = msg.split()
            if len(parts) >= 2:
                sym = parts[1].upper()
                p = self.engine.price_cache.get(sym)
                if p is not None:
                    return await reply(f"{sym}: {_fmt_money(p)}")
                return await reply(f"No tengo precio de {sym} todavía.")
            # Sin símbolo: listar todos los que conocemos
            cache = getattr(self.engine, "price_cache", {}) or {}
            if not cache:
                return await reply("Todavía no tengo precios en caché.")
            listado = "\n".join(f"• {k}: {_fmt_money(v)}" for k, v in cache.items())
            return await reply("Precios:\n" + listado)

        # --- DIAGNOSTICO ---
        if norm_all in ("diag on", "diagnostico on", "diagnostico activar"):
            setattr(self.engine, "diag", True)
            return await reply("Modo diagnóstico activado: registrando motivos de no-entrada.")
        if norm_all in ("diag off", "diagnostico off", "diagnostico desactivar"):
            setattr(self.engine, "diag", False)
            return await reply("Modo diagnóstico desactivado.")

        # --- KILL / KILLSWITCH ---
        if msg in ("kill", "killswitch"):
            ks = self.engine.toggle_killswitch()
            return await reply(f"Killswitch: {'ACTIVADO' if ks else 'desactivado'}")

        # --- STATS ---
        if norm_all in ('stats', 'estadisticas', 'estadísticas'):
            csv_dir = getattr(self.engine, 'csv_dir', 'data')
            closes, by_sym, by_layer, pf_total, wr, exp, dur_avg_min, dd = _compute_stats(1, csv_dir)
            partes = [
                f'Estadísticas (24h):',
                f'PF total: {pf_total:.2f}',
                f'Winrate: {wr*100:.1f}%',
                f'Expectancy: {exp:.2f}',
                ('Tiempo promedio en trade: ' + (f'{dur_avg_min:.1f} min' if dur_avg_min is not None else 'N/A')),
                f'DD máx período: {dd*100:.2f}%',
            ]
            if by_layer:
                for lay, d in by_layer.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {lay}: PF {pf:.2f} (n={d["n"]})')
            if by_sym:
                for sym, d in by_sym.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {sym}: PF {pf:.2f} (n={d["n"]})')
            return await reply('\n'.join(partes))

        if norm_all in ('stats semana', 'estadisticas semana', 'estadísticas semana'):
            csv_dir = getattr(self.engine, 'csv_dir', 'data')
            closes, by_sym, by_layer, pf_total, wr, exp, dur_avg_min, dd = _compute_stats(7, csv_dir)
            partes = [
                f'Estadísticas (7 días):',
                f'PF total: {pf_total:.2f}',
                f'Winrate: {wr*100:.1f}%',
                f'Expectancy: {exp:.2f}',
                ('Tiempo promedio en trade: ' + (f'{dur_avg_min:.1f} min' if dur_avg_min is not None else 'N/A')),
                f'DD máx período: {dd*100:.2f}%',
            ]
            if by_layer:
                for lay, d in by_layer.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {lay}: PF {pf:.2f} (n={d["n"]})')
            if by_sym:
                for sym, d in by_sym.items():
                    g = d['g']; l = d['l']; pf = (g/abs(l)) if l<0 else (g if g>0 else 0.0)
                    partes.append(f'• {sym}: PF {pf:.2f} (n={d["n"]})')
            return await reply('\n'.join(partes))

        # --- CERRAR TODO ---
        if msg in ("cerrar todo", "close all", "close_all", "cerrar"):
            ok = await self.engine.close_all()
            if ok:
                return await reply("Cerré todas las posiciones.")
            return await reply("No pude cerrar todo.")


        # --- RECIENTES / MOTIVOS ---
        if norm_all in ("recientes", "motivos"):
            try:
                rej = self.engine.recent_rejections(n=10)
            except Exception as e:
                rej = []
            if not rej:
                return await reply(
                    "No tengo motivos registrados aún.\n"
                    "Tip: activá diagnóstico con 'diag on' para registrar próximas oportunidades."
                )
            lines = []
            for r in rej:
                iso = r.get("iso") or ""
                sym = r.get("symbol") or ""
                reason = r.get("reason") or ""
                det = r.get("detail") or ""
                # Traducciones simples de razones
                if reason == "killswitch":
                    reason_es = "Bot OFF (killswitch activado)"
                elif reason == "entries_disabled":
                    reason_es = "Entradas deshabilitadas por configuración"
                elif reason == "funding_guard":
                    reason_es = "Funding anualizado por encima del límite"
                elif reason == "cooldown":
                    reason_es = "En cooldown entre entradas"
                elif reason == "pre_open_checks":
                    reason_es = f"Guardas de riesgo: {det}"
                    det = ""
                else:
                    reason_es = reason
                extra = []
                for k, v in r.items():
                    if k.startswith("extra_"):
                        extra.append(f"{k[6:]}={v}")
                extra_txt = (" [" + ", ".join(extra) + "]") if extra else ""
                line = f"• {iso} — {sym}: {reason_es}" + (f" ({det})" if det else "") + extra_txt
                lines.append(line)
            return await reply("🕒 Motivos recientes (últimas 10 oportunidades NO abiertas):\n" + "\n".join(lines))
        # Si no entendí, mostrar ayuda breve
        return await reply("No entendí. Escribí *ayuda* para ver comandos.")

    async def run(self):
        if not self.token:
            log.warning("TELEGRAM_TOKEN no configurado; comandos desactivados")
            return
        app = Application.builder().token(self.token).build()
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))
        # PTB v20+ necesita iniciar explícitamente
        await app.initialize()
        # Desactivar webhook previo y empezar polling
        try:
            await app.bot.delete_webhook(drop_pending_updates=True)
        except Exception:
            pass
        await app.start()
        try:
            await app.updater.start_polling()
        except Exception:
            pass
        log.info("Telegram command listener started")
        await asyncio.Event().wait()
