import asyncio
from types import SimpleNamespace

import pytest

import config
from paper_store import PaperStore


def _reset_paths(monkeypatch):
    import paths

    paths.get_data_dir.cache_clear()
    paths.get_paper_store_path.cache_clear()
    paths.get_live_store_path.cache_clear()
    paths.get_runtime_dir.cache_clear()
    for var in ("LIVE_STORE_PATH", "PAPER_STORE_PATH"):
        try:
            monkeypatch.delenv(var)
        except KeyError:
            pass


def _reset_trading_state():
    import trading

    trading.BROKER = None
    trading.POSITION_SERVICE = None
    trading.PUBLIC_CCXT_CLIENT = None
    trading.ACTIVE_MODE = "simulado"
    trading.ACTIVE_STORE_PATH = None
    trading.ACTIVE_DATA_DIR = None
    trading.LAST_MODE_CHANGE_SOURCE = "startup"
    trading._INITIALIZED = False


def test_startup_defaults_sim_mode(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("trading_mode: simulado\n")
    data_dir = tmp_path / "data"

    monkeypatch.setenv("CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    _reset_paths(monkeypatch)

    from bot import mode_manager

    mode_manager.CONFIG_PATH = str(cfg_path)
    cfg = config.load_raw_config(str(cfg_path))

    mode, source, changed = mode_manager.ensure_startup_mode(cfg, persist=False)

    assert mode == "simulado"
    assert source in {"config", "config_mode", "default"}
    assert changed is False
    assert cfg["mode"] == "paper"
    assert cfg["trading_mode"] == "simulado"

    store = PaperStore()
    assert store.path.name == "paper_state.json"
    assert store.path.parent == data_dir


def test_startup_real_env_persists_and_live_store(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("trading_mode: simulado\n")
    data_dir = tmp_path / "data"

    monkeypatch.setenv("CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("TRADING_MODE", "real")
    monkeypatch.setenv("BINANCE_API_KEY", "apikey-1234567890")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret-1234567890")
    _reset_paths(monkeypatch)

    from bot import mode_manager

    mode_manager.CONFIG_PATH = str(cfg_path)
    cfg = config.load_raw_config(str(cfg_path))

    mode, source, changed = mode_manager.ensure_startup_mode(
        cfg,
        env_mode="real",
        persist=True,
    )

    assert mode == "real"
    assert source == "env"
    assert changed is True

    cfg_disk = config.load_raw_config(str(cfg_path))
    assert str(cfg_disk.get("trading_mode")).lower() == "real"

    import trading
    from paths import get_live_store_path

    _reset_trading_state()

    from config import S

    def _fake_build_public_ccxt():
        return SimpleNamespace(
            apiKey=getattr(S, "binance_api_key", None),
            secret=getattr(S, "binance_api_secret", None),
        )

    monkeypatch.setattr(trading, "_build_public_ccxt", _fake_build_public_ccxt)

    def _fake_build_broker(settings, client_factory):
        return SimpleNamespace(store=PaperStore(path=get_live_store_path(), start_equity=settings.start_equity))

    monkeypatch.setattr(trading, "build_broker", _fake_build_broker)

    trading.ensure_initialized(mode="real")

    assert trading.ACTIVE_MODE == "real"
    assert trading.ACTIVE_STORE_PATH == get_live_store_path()
    assert trading.PUBLIC_CCXT_CLIENT.apiKey == "apikey-1234567890"


def test_runtime_flip_preserves_live_store(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("trading_mode: simulado\n")
    data_dir = tmp_path / "data"

    monkeypatch.setenv("CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("TRADING_MODE", "real")
    monkeypatch.setenv("BINANCE_API_KEY", "flipkey-1234567890")
    monkeypatch.setenv("BINANCE_API_SECRET", "flipsecret-1234567890")
    _reset_paths(monkeypatch)

    from bot import mode_manager

    mode_manager.CONFIG_PATH = str(cfg_path)
    cfg = config.load_raw_config(str(cfg_path))
    mode_manager.ensure_startup_mode(cfg, env_mode="real", persist=True)

    import trading
    from paths import get_live_store_path, get_paper_store_path

    _reset_trading_state()

    live_store_path = get_live_store_path()

    from config import S

    def _fake_build_public_ccxt():
        return SimpleNamespace(
            apiKey=getattr(S, "binance_api_key", None),
            secret=getattr(S, "binance_api_secret", None),
        )

    monkeypatch.setattr(trading, "_build_public_ccxt", _fake_build_public_ccxt)

    def _fake_build_broker(settings, client_factory):
        target_path = get_live_store_path() if settings.trading_mode == "real" else get_paper_store_path()
        return SimpleNamespace(store=PaperStore(path=target_path, start_equity=settings.start_equity))

    monkeypatch.setattr(trading, "build_broker", _fake_build_broker)

    trading.ensure_initialized(mode="real")
    live_snapshot = live_store_path.read_text()

    result = trading.set_trading_mode("simulado", source="test")
    assert result.ok
    assert trading.ACTIVE_MODE == "simulado"
    assert trading.ACTIVE_STORE_PATH == get_paper_store_path()
    assert live_store_path.read_text() == live_snapshot


def test_exchange_respects_paper_mode(tmp_path, monkeypatch):
    from bot.exchange import Exchange
    import trading

    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    _reset_paths(monkeypatch)
    _reset_trading_state()

    class DummyStore:
        def load(self):
            return {
                "equity": 1234.56,
                "pos_qty": 0.0,
                "avg_price": 0.0,
                "fees": 0.0,
                "realized_pnl": 10.0,
            }

    class DummyPosSvc:
        def __init__(self):
            self.store = DummyStore()

        def get_status(self):
            return {"side": "FLAT", "qty": 0.0, "equity": 1244.56, "mark": 0.0}

    trading.POSITION_SERVICE = DummyPosSvc()

    ex = Exchange({"symbol": "BTC/USDT"})

    loop = asyncio.get_event_loop()
    bal = loop.run_until_complete(ex.fetch_balance_usdt())
    assert bal == pytest.approx(1244.56)

    pos = loop.run_until_complete(ex.get_open_position("BTC/USDT"))
    assert pos is None

    lst = loop.run_until_complete(ex.list_open_positions())
    assert lst == []


def test_telegram_mode_commands_use_source(monkeypatch):
    from bot.telemetry.telegram_bot import _cmd_modo_real, _cmd_modo_simulado

    calls = []

    class DummyExchange:
        is_authenticated = True
        client = SimpleNamespace(apiKey="dummy")

        async def upgrade_to_real_if_needed(self):
            return None

        async def fetch_balance_usdt(self):
            return 0.0

    class DummyEngine:
        def __init__(self):
            self.mode = "paper"
            self.is_live = False
            self.exchange = DummyExchange()
            self.config = {}

        def set_mode(self, mode, *, source):
            calls.append((mode, source))
            self.mode = mode
            self.is_live = mode == "live"
            return mode

    async def _reply(text, **_kwargs):
        return text

    async def _run():
        engine = DummyEngine()
        await _cmd_modo_real(engine, _reply)
        await _cmd_modo_simulado(engine, _reply)

    asyncio.run(_run())

    assert ("live", "telegram") in calls
    assert ("paper", "telegram") in calls
