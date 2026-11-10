# -*- coding: utf-8 -*-
"""多品种并行：品种配置 / 价格精度 / 按品种成本 / 手机推送。

全部用假数据，不依赖真实行情文件与网络。
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "signal_desk"))

import markets   # noqa: E402
import pxfmt     # noqa: E402
import plan as plan_mod  # noqa: E402
import shadow    # noqa: E402
import notify    # noqa: E402


def _bar(t, o, h, l, c):
    return [t, o, h, l, c, 100, 0.0]


class MarketsTest(unittest.TestCase):
    def test_defaults_have_three_symbols(self):
        syms = [x["sym"] for x in markets.load()]
        self.assertIn("GOLD", syms)
        self.assertIn("EURUSD", syms)
        self.assertIn("GBPUSD", syms)

    def test_digits_and_contract_per_symbol(self):
        self.assertEqual(markets.digits("GOLD"), 2)
        self.assertEqual(markets.digits("EURUSD"), 5)
        self.assertEqual(markets.contract("GOLD"), 100)
        self.assertEqual(markets.contract("EURUSD"), 100000)
        # GOLD# 这类带后缀的写法也要能归一
        self.assertEqual(markets.digits("GOLD#"), 2)

    def test_unknown_symbol_gets_sane_fallback(self):
        o = markets.of("USDJPY")
        self.assertTrue(o["contract"] in (100, 100000))
        self.assertGreater(o["near"], 0)

    def test_save_only_touches_allowed_keys(self):
        tmp = Path(tempfile.mkdtemp(prefix="mk_test_"))
        orig = markets.FILE
        try:
            markets.FILE = tmp / "markets.json"
            markets._cache.update({"key": None})
            r = markets.save({"symbols": [
                {"sym": "GOLD", "enabled": False, "contract": 999999, "near": 2.5}]})
            self.assertTrue(r["ok"])
            g = markets.of("GOLD")
            self.assertFalse(g["enabled"])
            self.assertEqual(g["near"], 2.5)
            self.assertEqual(g["contract"], 100)     # 合约由代码定，页面改不动
        finally:
            markets.FILE = orig
            markets._cache.update({"key": None})
            shutil.rmtree(tmp, ignore_errors=True)

    def test_costs_scale_with_symbol(self):
        # 黄金的点差/滑点按「美金」，直盘按「价格单位」——直盘必须远小于黄金
        self.assertLess(markets.spread_default("EURUSD"), 0.001)
        self.assertLess(markets.slip("EURUSD"), markets.slip("GOLD") / 100)


class PxFormatTest(unittest.TestCase):
    def test_use_switches_precision(self):
        pxfmt.use("GOLD")
        self.assertEqual(pxfmt.px(), 2)
        self.assertEqual(pxfmt.pr(4113.456), 4113.46)
        self.assertEqual(pxfmt.fmt(4113.456), "4113.46")
        pxfmt.use("EURUSD")
        self.assertEqual(pxfmt.px(), 5)
        self.assertEqual(pxfmt.pr(1.11873), 1.11873)     # 不再被抹成 1.12
        self.assertEqual(pxfmt.fmt(1.11873), "1.11873")
        self.assertEqual(pxfmt.fmt(None), "—")

    def test_plan_lots_respect_contract(self):
        # 风险 20000 × 0.5% = 100 美金。
        # 黄金止损 30 美金/盎司 → 每手亏 3000 → 0.03 手；
        # 直盘止损 30 pips（0.0030）→ 每手亏 0.0030×100000 = 300 → 0.33 手。
        # 若合约写死 100，直盘会算出 33 手（放大 100 倍）——这是本测试要防的事故。
        gold = plan_mod._lots(20000, 0.5, 30.0, 1.0, contract=100)
        eur = plan_mod._lots(20000, 0.5, 0.0030, 1.0, contract=100000)
        self.assertAlmostEqual(gold, 0.03, places=2)
        self.assertAlmostEqual(eur, 0.33, places=2)
        eur_wrong = plan_mod._lots(20000, 0.5, 0.0030, None, contract=100)
        self.assertAlmostEqual(eur_wrong, 333.33, places=1)  # 反证：写死 100 的后果


class ShadowFxTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="shadow_fx_"))
        self._orig_book, self._orig_bars = shadow.BOOK, shadow._bars
        shadow.BOOK = self._tmp / "book.json"
        # EURUSD：1.0800 入场，止损 1.0770（30 pips），目标 1.0845（45 pips）
        # 记录时刻只有 12:00 这根 K 线；「未来」那根在结算前才补给 _bars
        self._past = [_bar("2026-11-09 12:00:00", 1.0800, 1.0800, 1.0800, 1.0800)]
        self._future = [_bar("2026-11-09 12:05:00", 1.0800, 1.0845, 1.0795, 1.0840)]
        shadow._bars = lambda sym, tf: list(self._past)
        self.cfg = {"tf": "M30"}
        self.ctx = {"price": 1.0799, "spread": 0.00012}

    def tearDown(self):
        shadow.BOOK = self._orig_book
        shadow._bars = self._orig_bars
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _res(self):
        return {"a1": {"direction": "buy", "confidence": 60,
                       "plan": {"entry_zone": [1.0800, 1.0800], "stop": 1.0770,
                                "targets": [{"price": 1.0845}]}},
                "a2": {"verdict": "allow", "local_lots":
                       {"lots": 0.33, "risk_usd": 100.0}},
                "facts": {"zone": {"zone": "lower"}}}

    def test_fx_sample_settles_with_tiny_cost(self):
        it = shadow.record("EURUSD", self.cfg, self._res(), self.ctx)
        self.assertIsNotNone(it)
        self.assertEqual(it["entry"], 1.08)          # 5 位精度，不再被抹成 1.08
        self.assertEqual(it["bar_start"], "2026-11-09 12:00:00")
        shadow._bars = lambda sym, tf: self._past + self._future
        self.assertEqual(shadow.settle()["settled"], 1)
        st = shadow.state(settle_now=False)
        row = next(x for x in st["recent"] if x["id"] == it["id"])
        self.assertEqual(row["status"], "won")
        self.assertGreater(row["net_r"], 1.3)        # 1.5R 扣直盘成本，必须远大于金的口径误差
        self.assertLess(row["cost_r"], 0.1)          # 直盘成本应是「点」级，不是金级 0.35R
        # 分品种统计里能看到 EURUSD
        self.assertIn("EURUSD", st["by_sym"])

    def test_gold_cost_stays_old_scale(self):
        ctx = {"price": 4100.0, "spread": 0.3}
        res = self._res()
        res["a1"]["plan"] = {"entry_zone": [4100.0, 4100.0], "stop": 4090.0,
                             "targets": [{"price": 4115.0}]}
        self._past = [_bar("2026-11-09 12:00:00", 4100, 4100, 4100, 4100)]
        shadow._bars = lambda sym, tf: list(self._past)
        it = shadow.record("GOLD", self.cfg, res, ctx)
        shadow._bars = lambda sym, tf: self._past + [
            _bar("2026-11-09 12:05:00", 4100, 4115, 4095, 4114)]
        shadow.settle()
        st = shadow.state(settle_now=False)
        row = next(x for x in st["recent"] if x["id"] == it["id"])
        self.assertEqual(row["status"], "won")
        self.assertAlmostEqual(row["cost_r"], 0.035, delta=0.02)


class NotifyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="nt_test_"))
        self._orig = (notify.CFG_FILE, notify.STATE_FILE, notify.LOG_FILE, notify._http)
        notify.CFG_FILE = self._tmp / "notify.json"
        notify.STATE_FILE = self._tmp / "state.json"
        notify.LOG_FILE = self._tmp / "log.jsonl"

    def tearDown(self):
        (notify.CFG_FILE, notify.STATE_FILE, notify.LOG_FILE, notify._http) = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_wecom_payload_and_cooldown(self):
        seen = []

        def fake_http(url, payload=None, timeout=8.0):
            seen.append((url, payload))
            return True, "ok"

        notify._http = fake_http
        notify.save({"channel": "wecom", "webhook": "https://qyapi.example/xxx"})
        r1 = notify.send("测试", "正文", kind="alert", dedupe="k1", force=False)
        r2 = notify.send("测试", "正文", kind="alert", dedupe="k1", force=False)
        self.assertTrue(r1["ok"])
        self.assertFalse(r2["ok"])                       # 冷却期内不再发
        self.assertEqual(len(seen), 1)
        url, payload = seen[0]
        self.assertIn("qyapi.example", url)
        self.assertEqual(payload["msgtype"], "markdown")
        self.assertIn("测试", payload["markdown"]["content"])

    def test_quiet_hours_cross_midnight(self):
        self.assertTrue(notify._quiet_now("00:00-23:59"))
        self.assertFalse(notify._quiet_now(""))
        # 12:00 不在 23:00-07:00 内（静默跨零点的判定不能反）
        import datetime as dt
        real = dt.datetime
        class FakeDt(dt.datetime):
            @classmethod
            def now(cls):
                return cls(2026, 11, 9, 12, 0)
        notify.dt.datetime = FakeDt
        try:
            self.assertFalse(notify._quiet_now("23:00-07:00"))
        finally:
            notify.dt.datetime = real

    def test_disabled_channel_short_circuits(self):
        notify.save({"channel": "telegram", "webhook": "", "token": ""})
        r = notify.send("t", "x", kind="order", force=True)
        self.assertFalse(r["ok"])
        self.assertIn("token", r["error"])


if __name__ == "__main__":
    unittest.main()
