"""Módulo de paquetes de exchanges.

No realices imports aquí que provoquen efectos secundarios. De esta forma
evitamos ciclos de importación cuando otros módulos acceden a
``bot.exchanges`` o a cualquiera de sus submódulos.
"""

# Se puede exponer explícitamente aquello que deba ser público sin forzar
# la importación de los submódulos pesados.
__all__ = [
    "binance_filters",
]
