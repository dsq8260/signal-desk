"""账户桥读取测试（离线伪造 CSV）。"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from gold.account import account, by_magic, deals, positions, status

ACC = ("time,login,server,company,currency,balance,equity,margin,"
       "margin_free,margin_level,profit,leverage")
POS = "ticket,symbol,type,volume,price_open,price_current,profit,swap,magic,comment"
DEAL = ("time,ticket,symbol,type,entry,volume,price,profit,swap,"
        "commission,magic,comment")


def _acc(d: pathlib.Path, **kw) -> None:
    row = {"time": "2026.10.29 11:30", "login": 16959846,
           "server": "XMGlobal-MT5 2", "company": "XM Global Limited",
           "currency": "USD", "balance": 10673.16, "equity": 11192.52,
           "margin": 318.23, "margin_free": 10874.29, "margin_level": 3517.12,
           "profit": 519.36, "leverage": 100}
    row.update(kw)
    (d / "mt5_account.csv").write_text(
        ACC + "\n" + ",".join(str(row[k]) for k in ACC.split(",")) + "\n",
        encoding="ascii")


def _pos(d: pathlib.Path, rows: list[tuple]) -> None:
    lines = [POS] + [",".join(str(x) for x in r) for r in rows]
    (d / "mt5_positions.csv").write_text("\n".join(lines) + "\n", encoding="ascii")


def _deal(d: pathlib.Path, rows: list[tuple]) -> None:
    lines = [DEAL] + [",".join(str(x) for x in r) for r in rows]
    (d / "mt5_deals.csv").write_text("\n".join(lines) + "\n", encoding="ascii")


def test_account_parses(tmp_path):
    _acc(tmp_path)
    a = account(tmp_path)
    assert a is not None
    assert a["equity"] == pytest.approx(11192.52)
    assert a["leverage"] == 100
    assert a["currency"] == "USD"


def test_account_missing(tmp_path):
    assert account(tmp_path) is None


def test_positions_side_mapping(tmp_path):
    _pos(tmp_path, [
        (123, "GOLD#", 0, 0.08, 3977.82, 4042.74, 519.36, 0.0, 20261001, ""),
        (124, "EURUSD", 1, 0.10, 1.1000, 1.0950, -50.0, -1.2, 20261002, ""),
    ])
    df = positions(tmp_path)
    assert len(df) == 2
    assert list(df["side"]) == ["BUY", "SELL"]
    assert df["profit"].iloc[0] == pytest.approx(519.36)


def test_positions_empty(tmp_path):
    assert len(positions(tmp_path)) == 0


def test_deals_sorted_and_typed(tmp_path):
    _deal(tmp_path, [
        ("2026.10.28 10:00", 1, "GOLD#", 0, 1, 0.08, 4000, 0, 0, -0.5, 20261001, ""),
        ("2026.10.27 10:00", 2, "GOLD#", 1, 0, 0.08, 4010, 80.0, 0, -0.5, 20261001, ""),
    ])
    df = deals(tmp_path)
    assert len(df) == 2
    assert df["time"].is_monotonic_increasing
    assert df["profit"].sum() == pytest.approx(80.0)


def test_by_magic_groups(tmp_path):
    _deal(tmp_path, [
        ("2026.10.27 10:00", 1, "GOLD#", 1, 0, 0.08, 4010, 80.0, 0, -0.5, 20261001, ""),
        ("2026.10.27 11:00", 2, "GOLD#", 1, 0, 0.08, 3990, -30.0, 0, -0.5, 20261002, ""),
        ("2026.10.27 12:00", 3, "GOLD#", 1, 0, 0.08, 4000, 10.0, 0, -0.5, 20261001, ""),
    ])
    g = by_magic(tmp_path)
    assert len(g) == 2
    assert list(g["magic"]) == [20261001, 20261002]
    assert g.iloc[0]["净收益"] == pytest.approx(90.0)
    assert g.iloc[0]["笔数"] == 2


def test_by_magic_empty(tmp_path):
    g = by_magic(tmp_path)
    assert len(g) == 0
    assert "magic" in g.columns


def test_status_flags(tmp_path):
    _acc(tmp_path)
    st = status(tmp_path)
    assert st["account"] is True
    assert st["positions"] is False
    assert st["deals"] is False
