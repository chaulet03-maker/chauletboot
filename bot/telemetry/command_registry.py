"""Central registry for Telegram commands with normalization and aliases."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Tuple


def normalize(text: str) -> str:
    """Normalize a piece of text by removing accents and collapsing spaces."""

    if text is None:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    lowered = stripped.lower().strip()
    return re.sub(r"\s+", " ", lowered)


class CommandRegistry:
    """Keeps track of command handlers, aliases and help metadata."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable] = {}
        self._aliases: Dict[str, List[str]] = {}
        self._aliases_display: Dict[str, List[str]] = {}
        self._help: Dict[str, str] = {}
        self._show_in_help: Dict[str, bool] = {}

    def register(
        self,
        canonical: str,
        fn: Callable,
        *,
        aliases: List[str],
        help_text: str,
        show_in_help: bool = True,
    ) -> None:
        """Register a canonical command with aliases and metadata."""

        display_aliases = [canonical] + list(aliases)
        self._handlers[canonical] = fn
        self._aliases_display[canonical] = display_aliases
        self._aliases[canonical] = [normalize(alias) for alias in display_aliases]
        self._help[canonical] = help_text
        self._show_in_help[canonical] = show_in_help

    def resolve(self, text: str) -> Optional[str]:
        """Return the canonical command name that matches *text*, if any."""

        if not text:
            return None
        norm = normalize(text)
        candidates = [norm]
        if " " in norm:
            candidates.append(norm.split(" ", 1)[0])
        if norm.startswith("/"):
            trimmed = norm[1:]
            candidates.append(trimmed)
            if " " in trimmed:
                candidates.append(trimmed.split(" ", 1)[0])

        for candidate in candidates:
            for canonical, alias_list in self._aliases.items():
                if candidate in alias_list:
                    return canonical
        return None

    def handler_for(self, canonical: str) -> Optional[Callable]:
        return self._handlers.get(canonical)

    def help_lines(self) -> List[Tuple[str, str]]:
        """Return (command, description) pairs for help output."""

        lines: List[Tuple[str, str]] = []
        for command, show in self._show_in_help.items():
            if not show:
                continue
            aliases_display = self._aliases_display.get(command, [])
            common: List[str] = []
            for alias in aliases_display:
                if normalize(alias) == normalize(command):
                    continue
                if alias not in common:
                    common.append(alias)
                if len(common) == 2:
                    break
            alias_hint = f" (alias: {', '.join(common)})" if common else ""
            lines.append((command, self._help.get(command, "") + alias_hint))
        return lines

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._handlers)

