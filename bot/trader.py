import logging

class Trader:
    def __init__(self, cfg):
        self.config = cfg
        self.balance = self.config.get('balance', 1000)
        self.open_position = None # Aquí guardaremos la información de la posición abierta

    def get_balance(self):
        """ Devuelve el balance actual de la cuenta. """
        # En el futuro, esto podría pedir el balance a la API del exchange
        return self.balance

    def check_open_position(self):
        """ Revisa si hay una posición abierta. """
        # En una versión real, esto consultaría al exchange
        # Por ahora, usamos una variable interna
        return self.open_position

    def set_position(self, position_data):
        """ Actualiza el estado de la posición. """
        self.open_position = position_data
        logging.info(f"Nuevo estado de posición: {position_data}")
