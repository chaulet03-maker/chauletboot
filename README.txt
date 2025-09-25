PASOS (Windows/PowerShell):
1) Crear venv e instalar deps:
   py -m venv .venv
   . .\.venv\Scripts\Activate.ps1
   py -m pip install --upgrade pip
   py -m pip install -r requirements.txt

2) Crear .env (copiar desde .env.example y completar TOKEN/CHAT_ID):
   copy .env.example .env
   # Editar .env y poner TELEGRAM_TOKEN y TELEGRAM_CHAT_ID

3) Ejecutar:
   py start.py

Comandos en Telegram:
/ayuda, /estado, /saldo, /posiciones, /bot on, /bot off, /cerrar todo
También podés mandar:  open BTC/USDT:USDT long
