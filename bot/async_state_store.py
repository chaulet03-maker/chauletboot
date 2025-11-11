"""Asynchronous Redis-backed state store for the trading bot."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Coroutine, TypeVar

try:  # Prefer the standalone aioredis package when available
    import aioredis  # type: ignore
except Exception:  # pragma: no cover - fallback to redis asyncio client
    aioredis = None  # type: ignore

try:  # Redis >=4 exposes an asyncio interface
    from redis import asyncio as redis_async  # type: ignore
except Exception:  # pragma: no cover - redis is optional at runtime
    redis_async = None  # type: ignore


logger = logging.getLogger(__name__)


def _build_url(host: str, port: int, db: int, password: str | None = None) -> str:
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


@dataclass(slots=True)
class RedisConfig:
    """Connection configuration for Redis."""

    url: str


T = TypeVar("T")


class AsyncRedisStateStore:
    """Asynchronous state persistence using Redis."""

    def __init__(
        self,
        *,
        url: str | None = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        decode_responses: bool = True,
    ) -> None:
        if url:
            self._config = RedisConfig(url=url)
        else:
            self._config = RedisConfig(url=_build_url(host, port, db, password))
        self._decode_responses = decode_responses
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="redis-state-store",
            daemon=True,
        )
        self._loop_thread.start()
        self.redis = None
        self._connected = asyncio.Event()

    # ------------------------------------------------------------------
    # Internal helpers
    async def _connect_impl(self) -> None:
        if self.redis is not None:
            return
        client = None
        try:
            if aioredis is not None:
                client = aioredis.from_url(
                    self._config.url,
                    decode_responses=self._decode_responses,
                )
            elif redis_async is not None:
                client = redis_async.from_url(
                    self._config.url,
                    decode_responses=self._decode_responses,
                )
            else:
                raise RuntimeError(
                    "Neither 'aioredis' nor 'redis' asyncio client is available."
                )

            await client.ping()
        except Exception as exc:  # pragma: no cover - guarded connection
            logger.error("Fallo al conectar a Redis: %s", exc, exc_info=True)
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
            self.redis = None
            self._connected.clear()
            raise
        else:
            self.redis = client
            self._connected.set()
            logger.info("Conexión a Redis establecida (%s)", self._config.url)

    async def _load_state_impl(self, key: str) -> Dict[str, Any]:
        if self.redis is None:
            return {}
        try:
            raw = await self.redis.get(key)
        except Exception as exc:  # pragma: no cover - network error path
            logger.error("Error al cargar estado desde Redis (%s): %s", key, exc)
            return {}
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Estado inválido en Redis para %s: %s", key, raw)
            return {}

    async def _save_state_impl(self, key: str, state: Dict[str, Any]) -> None:
        if self.redis is None:
            return
        try:
            payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
            await self.redis.set(key, payload)
        except Exception as exc:  # pragma: no cover - network error path
            logger.error("Error al guardar estado en Redis (%s): %s", key, exc)

    async def _close_impl(self) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.close()
        finally:
            self.redis = None
            self._connected.clear()

    def _run_sync(
        self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None
    ) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    async def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)

    # ------------------------------------------------------------------
    # Public API
    async def connect(self) -> None:
        await self._run_async(self._connect_impl())

    def connect_sync(self, timeout: Optional[float] = None) -> None:
        self._run_sync(self._connect_impl(), timeout)

    async def load_state(self, key: str) -> Dict[str, Any]:
        if not self.is_connected:
            return {}
        return await self._run_async(self._load_state_impl(key))

    def load_state_sync(self, key: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        if not self.is_connected:
            return {}
        return self._run_sync(self._load_state_impl(key), timeout)

    async def save_state(self, key: str, state: Dict[str, Any]) -> None:
        if not self.is_connected:
            return
        await self._run_async(self._save_state_impl(key, state))

    def save_state_sync(
        self, key: str, state: Dict[str, Any], timeout: Optional[float] = None
    ) -> None:
        if not self.is_connected:
            return
        self._run_sync(self._save_state_impl(key, state), timeout)

    async def close(self) -> None:
        await self._run_async(self._close_impl())
        await asyncio.sleep(0)  # yield to ensure closure propagates

    def close_sync(self, timeout: Optional[float] = None) -> None:
        try:
            self._run_sync(self._close_impl(), timeout)
        finally:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and self.redis is not None


__all__ = ["AsyncRedisStateStore", "RedisConfig"]

