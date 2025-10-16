#!/usr/bin/env python3
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Candidatos típicos: raíz actual, subcarpeta "chauletboot-main/chauletboot-main",
# y la carpeta padre por si lo corrés un nivel arriba.
candidates = [
    HERE,
    HERE / "chauletboot-main" / "chauletboot-main",
    HERE / "chauletboot-main",
    HERE.parent,
]

project_root = None
for c in candidates:
    if (c / "bot").is_dir() and (c / "core").is_dir():
        project_root = c
        break

if not project_root:
    sys.stderr.write(
        "[ERROR] No encontré carpetas 'bot/' y 'core/'. "
        "Parate en la raíz del proyecto o ajustá 'candidates'.\n"
    )
    sys.exit(1)

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Opcional: asegurá que sean paquetes
for pkg in ("bot", "core"):
    initp = project_root / pkg / "__init__.py"
    if not initp.exists():
        initp.write_text("")  # crea __init__.py si faltara

# --- arrancar app ---
from bot.engine import TradingApp  # ahora sí resuelve core.strategy
if __name__ == "__main__":
    TradingApp().run()
