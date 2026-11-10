# -*- coding: utf-8 -*-
"""影子账本：记录、M5 结算、统计。全部用假 K 线，不依赖真实行情文件。"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "signal_desk"))

import shadow  # noqa: E402


def _bar(t, o, h, l, c):
    return [t, o, h, l, c, 100, 0.3]


class ShadowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="shadow_test_"))
        self._orig_book = shadow.BOOK
        self._orig_bars = shadow._bars
        shadow.BOOK = self._tmp / "book.json"
        shadow._bars = lambda sym, tf: [
            _bar("2026-11-09 12:00:00", 4100, 4100, 4100, 4100)]
        self.cfg = {"tf": "M30"}
        self.ctx = {"price": 4100.0, "spread": 0.3}

    def tearDown(self):
        shadow.BOOK = self._orig_book
        shadow._bars = self._orig_bars
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _res(self, direction="buy", plan=None, verdict="allow", ai_dir=None):
        plan = plan or {"entry_zone": [4095.0, 4095.0], "stop": 4085.0,
                        "targets": [{"price": 4125.0, "close_pct": 30}]}
        a1 = {"direction": direction, "confidence": 60, "plan":
              (plan if direction in ("buy", "sell") else None)}
        if ai_dir:
            a1["ai_direction"] = ai_dir
            a1["ai_plan"] = plan
        return {"a1": a1, "a2": {"verdict": verdict, "local_lots":
                                 {"lots": 0.05, "risk_usd": 53.5}},
                "facts": {"zone": {"zone": "lower"},
                          "sentiment": {"label": "偏多"},
                          "kickoff": {"tf": "M30"}}}

    def _item(self):
        d = shadow._read()
        return (d.get("items") or [])[-1] if d.get("items") else None

    def test_record_trade_sample(self):
        it = shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        self.assertIsNotNone(it)
        self.assertEqual(it["kind"], "trade")
        self.assertEqual(it["direction"], "buy")
        self.assertEqual(it["entry"], 4095.0)
        self.assertEqual(it["rr"], 3.0)          # (4125-4095)/(4095-4085)
        self.assertEqual(it["status"], "open")
        self.assertEqual(it["bar_start"], "2026-11-09 12:00:00")

    def test_blocked_sample_kept_separately(self):
        it = shadow.record("GOLD", self.cfg, self._res("wait", ai_dir="sell"),
                           self.ctx)
        self.assertEqual(it["kind"], "blocked")
        self.assertEqual(it["direction"], "sell")

    def test_wait_without_plan_not_recorded(self):
        self.assertIsNone(shadow.record("GOLD", self.cfg, self._res("wait"),
                                        self.ctx))
        self.assertEqual(shadow._read()["items"], [])

    def test_settle_win_after_entry(self):
        shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        shadow._bars = lambda sym, tf: [
            _bar("2026-11-09 12:00:00", 4100, 4100, 4100, 4100),
            _bar("2026-11-09 12:05:00", 4100, 4100, 4094, 4096),   # 回踩入场
            _bar("2026-11-09 12:10:00", 4096, 4110, 4096, 4108),
            _bar("2026-11-09 12:15:00", 4108, 4130, 4108, 4126),   # 到 TP1
        ]
        shadow.settle()
        it = self._item()
        self.assertEqual(it["status"], "won")
        self.assertAlmostEqual(it["r"], 3.0, places=2)
        self.assertAlmostEqual(it["cost_r"], (0.3 + 0.05) / 10.0, places=3)
        self.assertLess(it["net_r"], it["r"])    # 扣掉点差+滑点

    def test_settle_loss(self):
        shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        shadow._bars = lambda sym, tf: [
            _bar("2026-11-09 12:00:00", 4100, 4100, 4100, 4100),
            _bar("2026-11-09 12:05:00", 4100, 4100, 4094, 4096),
            _bar("2026-11-09 12:10:00", 4096, 4096, 4084, 4086),   # 破止损
        ]
        shadow.settle()
        it = self._item()
        self.assertEqual(it["status"], "lost")
        self.assertAlmostEqual(it["r"], -1.0, places=2)

    def test_same_bar_stop_and_target_counts_as_loss(self):
        shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        shadow._bars = lambda sym, tf: [
            _bar("2026-11-09 12:00:00", 4100, 4100, 4100, 4100),
            _bar("2026-11-09 12:05:00", 4100, 4100, 4094, 4096),
            _bar("2026-11-09 12:10:00", 4096, 4130, 4080, 4100),   # 同时碰到
        ]
        shadow.settle()
        self.assertEqual(self._item()["status"], "lost")

    def test_missed_when_never_returns(self):
        shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        bars = [_bar("2026-11-09 12:00:00", 4100, 4100, 4100, 4100)]
        t = 0
        for _ in range(shadow.CFG["expire_bars"] + 2):
            t += 5
            bars.append(_bar(f"2026-11-09 13:{t:02d}:00", 4110, 4115, 4105, 4112))
        shadow._bars = lambda sym, tf: bars
        shadow.settle()
        self.assertEqual(self._item()["status"], "missed")

    def test_state_stats_and_progress(self):
        shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        shadow.record("GOLD", self.cfg,
                      self._res("wait", ai_dir="sell"), self.ctx)
        shadow._bars = lambda sym, tf: [
            _bar("2026-11-09 12:00:00", 4100, 4100, 4100, 4100)]
        st = shadow.state()
        self.assertEqual(st["trade"]["n"], 1)
        self.assertEqual(st["blocked"]["n"], 1)
        self.assertEqual(st["trade"]["target"], 100)
        self.assertIsNotNone(st["trade"]["progress"])

    def test_debounce_same_plan(self):
        shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        shadow.record("GOLD", self.cfg, self._res(), self.ctx)
        self.assertEqual(len(shadow._read()["items"]), 1)


if __name__ == "__main__":
    unittest.main()
