import logging


class Notifier:
    def __init__(self, application, cfg):
        """
        Inicializa el notificador.
        :param application: La instancia de la aplicación de Telegram.
        :param cfg: El diccionario de configuración.
        """
        self.app = application
        self.config = cfg
        self.chat_id = self.config.get('telegram_chat_id')
        logging.info("Notifier inicializado correctamente.")

    def send(self, message):
        """ Envía un mensaje a través del bot de Telegram. """
        if not self.app or not self.chat_id:
            logging.warning("Notifier no configurado correctamente. No se puede enviar mensaje.")
            print("Notifier no configurado correctamente. No se puede enviar mensaje.")
            return

        try:
            # Usamos 'app.bot.send_message' para enviar el mensaje de forma asíncrona
            # Esta es una forma más robusta de enviar mensajes desde fuera de un handler
            self.app.job_queue.run_once(self._send_message_callback, 0, data={'message': message})
        except Exception as e:
            logging.error(f"Error al planificar el envío de notificación por Telegram: {e}")

    def _send_message_callback(self, context):
        """Callback ejecutado por el job_queue para enviar el mensaje."""
        data = getattr(context.job, 'data', {}) or {}
        message = data.get('message')
        if message is None:
            logging.warning("No se recibió mensaje para enviar en el job de notificación.")
            return

        try:
            context.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='MarkdownV2'
            )
        except Exception as e:
            logging.error(f"Error final al enviar la notificación por Telegram: {e}")
