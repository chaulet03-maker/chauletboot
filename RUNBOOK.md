
# RUNBOOK (PowerShell)

```powershell
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
# Si tenés requirements.txt, instálalo. Para usar parity con ccxt, instala ccxt:
# py -m pip install ccxt pandas numpy python-telegram-bot

# Opcional: Telegram para /status
# $env:TELEGRAM_BOT_TOKEN='123:AA...'
# $env:TELEGRAM_CHAT_ID='-100xxxxxxxx'

# PAPER + COTS + PARITY (autopatch ccxt) ya activos por defecto:
py -X dev -u start.py
```
