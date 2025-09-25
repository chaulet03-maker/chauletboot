import telebot
import config  # Asumo que tu config.py tiene funciones para obtener los tokens

# --- Configuración del Bot ---
# Obtenemos el token de nuestro archivo de configuración
try:
    TELEGRAM_BOT_TOKEN = config.get_telegram_token()
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("El token de Telegram no puede estar vacío.")
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    print("Bot de Telegram inicializado correctamente.")
except Exception as e:
    print(f"Error Crítico: No se pudo inicializar el bot de Telegram. Causa: {e}")
    # Si no podemos iniciar el bot, no tiene sentido continuar.
    exit()

# --- Definición de Comandos (Handlers) ---

# Este "decorador" le dice al bot: "Cuando alguien te envíe el comando /start o /help,
# ejecutá la función que está justo debajo."
@bot.message_handler(commands=['start', 'help'])
def enviar_mensaje_bienvenida(message):
    """
    Responde a los comandos /start y /help con un mensaje de bienvenida.
    """
    texto_bienvenida = (
        "¡Hola! Soy tu Bot de Trading.\n\n"
        "Estos son los comandos que entiendo:\n"
        "/start o /help - Muestra este mensaje de ayuda.\n"
        "/status - Muestra el estado actual de las operaciones.\n"
    )
    bot.reply_to(message, texto_bienvenida)

@bot.message_handler(commands=['status'])
def enviar_estado_bot(message):
    """
    Responde al comando /status. Aquí deberíamos conectar la lógica
    para obtener el estado real del bot de trading.
    """
    # --- TAREA PENDIENTE ---
    # Acá necesitaremos una función que vaya al "engine" del bot
    # y pregunte por el estado actual (ej: si está en una operación,
    # cuál fue la última decisión, etc.)
    # Por ahora, enviamos una respuesta temporal.
    
    estado_actual = "Función de status en desarrollo. ¡Pronto estará disponible!"
    
    bot.reply_to(message, estado_actual)

# --- Función para enviar notificaciones desde otras partes del código ---
def enviar_notificacion(mensaje):
    """
    Esta función puede ser llamada desde tu 'engine' de trading
    para enviar alertas o notificaciones a un chat específico.
    """
    try:
        CHAT_ID = config.get_telegram_chat_id() # Necesitas el ID de tu chat en la config
        if not CHAT_ID:
            print("Advertencia: No se configuró un CHAT_ID para enviar notificaciones.")
            return
        bot.send_message(CHAT_ID, mensaje)
    except Exception as e:
        print(f"Error al enviar notificación por Telegram: {e}")


# --- Inicio del Bot ---
def iniciar_escucha():
    """
    Esta es la función que pone al bot a escuchar mensajes nuevos.
    Es un ciclo infinito, por lo que debe ser lo último que se ejecute.
    """
    print("El bot de Telegram está ahora escuchando comandos...")
    # infinity_polling() mantiene al bot corriendo y esperando mensajes.
    bot.infinity_polling()

# Esta construcción permite que el archivo se pueda importar en otros
# sin que se ejecute automáticamente el bot. El bot solo se inicia
# si ejecutamos este archivo directamente (ej: python endpoint.py)
if __name__ == '__main__':
    iniciar_escucha()
