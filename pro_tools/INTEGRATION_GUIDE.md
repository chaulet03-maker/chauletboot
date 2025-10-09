
# Parity Integration (PAPER≡LIVE)

## Usar con ccxt (automático en PAPER)
Si tu código usa `ccxt.Exchange.create_order(...)`, **ya está parcheado** en PAPER para rutear por `pro_tools.parity`.

## Usar a mano (recomendado y claro)
```python
from pro_tools.autopatch import autopatch_if_enabled, parity
autopatch_if_enabled()  # crea `parity` global si no existe

# colocar orden (PAPER por defecto)
res = parity.place_order_sync(symbol="BTC/USDT", side="buy", type="market", qty=None, price=60000.0, params={})
print(res)
```
