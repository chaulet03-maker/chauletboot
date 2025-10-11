import logging
import asyncio


class Notifier:
    def __init__(self, application, cfg):
        self.app = application
        self.config = cfg
        self.chat_id = self.config.get('telegram_chat_id')
        logging.info("Notifier inicializado correctamente.")

    async def send(self, message):
        """ Envía un mensaje de forma asíncrona. """
        if not self.app or not self.chat_id:
            logging.warning("Notifier no configurado. No se puede enviar mensaje.")
            return
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logging.error(f"Error al enviar notificación por Telegram: {e}")
