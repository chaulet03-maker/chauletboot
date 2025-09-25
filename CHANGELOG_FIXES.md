
# Cambios y Fixes aplicados

- **CCXT / Binance**: ya no se sobrescribe `client.options`; ahora se fusiona y se define por defecto `adjustForTimeDifference=False` y `defaultType='swap'` (evita `KeyError: 'adjustForTimeDifference'`).
- **Carga de .env / Telegram**: ahora se aceptan tanto `TELEGRAM_TOKEN` como `TELEGRAM_BOT_TOKEN`. Se corrige el listener para iniciar *polling* con `Application` (PTB v21).
- **Import paths**: `start.py` ahora importa desde el paquete `bot.*` y elimina duplicaciones.
- **Persistencia (state.json)**: `load_state()` ahora tolera formatos viejos (listas) y no crashea; migra/ignora estructuras inválidas.
- **Rutas de Data**: se homogeniza el uso de `data/` (minúsculas) y se corrige `STATE_FILE`.
- **requirements.txt**: deduplicado y sin conflictos (una línea por paquete).
