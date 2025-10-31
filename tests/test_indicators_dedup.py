import pandas as pd

from bot.core.indicators import _ensure_ohlc_schema, _normalize_ts


def test_normalize_ts_removes_all_duplicate_timestamps():
    df = pd.DataFrame(
        {
            "ts": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T02:00:00Z",
            ],
            "value": [1, 2, 3, 4],
        }
    )

    normalized = _normalize_ts(df, ts_col="ts")

    assert normalized["ts"].tolist() == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T02:00:00Z"),
    ]
    assert normalized["value"].tolist() == [1, 4]


def test_ensure_ohlc_schema_strips_duplicate_rows():
    df = pd.DataFrame(
        {
            "ts": [
                "2024-01-01T00:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T02:00:00Z",
            ],
            "open": [1, 2, 3, 4],
            "high": [1, 2, 3, 4],
            "low": [1, 2, 3, 4],
            "close": [1, 2, 3, 4],
            "volume": [10, 20, 30, 40],
        }
    )

    clean = _ensure_ohlc_schema(df)

    assert clean["ts"].tolist() == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T02:00:00Z"),
    ]
    assert clean[["open", "high", "low", "close", "volume"]].values.tolist() == [
        [1.0, 1.0, 1.0, 1.0, 10.0],
        [4.0, 4.0, 4.0, 4.0, 40.0],
    ]
