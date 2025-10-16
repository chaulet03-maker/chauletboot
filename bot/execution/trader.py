# Adaptador hacia bot.trader.Trader para compatibilidad de nombres
from dataclasses import dataclass
from typing import Dict, List
from bot.trader import Trader as _Trader

Trader = _Trader  # alias de compatibilidad

# opcionalmente podríamos definir Lot = dict, pero el Trader interno ya maneja dicts de lotes
