# chauletboot

bot de prueba para curso de programacion

## Tips para la terminal

Para evitar que el autocompletado de la shell pida confirmación cuando hay muchas opciones podés ejecutar los siguientes comandos en la sesión actual:

```bash
bind 'set completion-query-items 2000'
bind 'set show-all-if-ambiguous on'
```

Para que la configuración sea permanente agregá estas líneas a `~/.inputrc`:

```
set completion-query-items 2000
set show-all-if-ambiguous on
set bell-style none
```

Luego abrí una nueva shell o ejecutá `bind -f ~/.inputrc` para recargar la configuración.
