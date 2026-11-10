# -*- coding: utf-8 -*-
"""ai_council 三角色决策面板：mock LLM 测编排逻辑，不打真实网络。

数据文件必须指到临时目录：run() 会写 state/audit/config，之前直接写真实文件，
把页面上的 AI-1 结论污染成了 mock 的「做多 60%」（2026-11-09 实际发生过）。
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "signal_desk"))

import ai_council  # noqa: E402


class AiCouncilTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="aicouncil_test_"))
        self._orig = (ai_council.STATE_FILE, ai_council.AUDIT_FILE,
                      ai_council.CFG_FILE)
        ai_council.STATE_FILE = self._tmp / "state.json"
        ai_council.AUDIT_FILE = self._tmp / "audit.jsonl"
        ai_council.CFG_FILE = self._tmp / "cfg.json"
        # run() 会往影子账本记样本：必须指到临时目录，否则 mock 计划会混进真实账本
        import shadow
        self._orig_book = shadow.BOOK
        shadow.BOOK = self._tmp / "shadow_book.json"

    def tearDown(self):
        ai_council.STATE_FILE, ai_council.AUDIT_FILE, ai_council.CFG_FILE = self._orig
        import shadow
        shadow.BOOK = self._orig_book
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_hard_gates_quote_stale(self):
        ctx = {"quote_age_s": 300, "event": {}, "account": {},
               "gate_reasons": []}
        gates = ai_council._hard_gates(ctx)
        self.assertTrue(any(g["gate"] == "报价新鲜度" for g in gates))

    def test_hard_gates_event_window(self):
        ctx = {"quote_age_s": 1, "event": {"name": "非农", "mins": 15},
               "account": {}, "gate_reasons": []}
        gates = ai_council._hard_gates(ctx)
        self.assertTrue(any(g["gate"] == "高影响数据窗口" for g in gates))

    def test_hard_gates_day_loss(self):
        ctx = {"quote_age_s": 1, "event": {},
               "account": {"equity": 9000, "balance": 10000},
               "gate_reasons": []}
        gates = ai_council._hard_gates(ctx)
        self.assertTrue(any(g["gate"] == "日亏损预算" for g in gates))

    def test_hard_gates_clean(self):
        ctx = {"quote_age_s": 1, "event": {},
               "account": {"equity": 10100, "balance": 10000},
               "gate_reasons": []}
        self.assertEqual(ai_council._hard_gates(ctx), [])

    def test_run_mock_llm(self):
        """mock _llm：验证三角色编排（自然语言+<result> 机器字段）、审计落盘与缓存。"""
        seq = [
            "【核心结论】观望，置信度 35%。H1 包络中部无极值。\n"
            "【信号校验拆解】- 区间位置：中部，无候选区。\n"
            "<result>" + json.dumps({"direction": "wait", "confidence": 35,
                                     "summary": "测试观望", "plan": None,
                                     "wait_rules": ["⑤"]}) + "</result>",
            json.dumps({"verdict": "warn", "risk_notes": ["测试意见"],
                        "size_advice": "0.01 手"}),
            json.dumps({"positions": [], "summary": "空仓，无需监控"}),
        ]
        calls = []

        def fake_llm(messages, cfg=None, max_tokens=None, **kw):
            calls.append(messages[0]["content"][:20])
            return seq[len(calls) - 1]

        orig = ai_council._llm
        orig_bc = ai_council.build_context
        ai_council._llm = fake_llm
        ai_council.build_context = lambda *a, **k: {
            "price": 4100.0, "quote_age_s": 0, "event": {},
            "account": {"equity": 10000, "balance": 10000},
            "positions": [], "gate_reasons": [], "levels": [], "cross": []}
        try:
            out = ai_council.run("GOLD", force=True, tf="H1")
            self.assertTrue(out["ok"], out.get("result", {}).get("errors"))
            res = out["result"]
            self.assertEqual(res["a1"]["direction"], "wait")
            self.assertIn("观望", res["a1"]["summary"])
            self.assertIn("⑤", res["a1"]["wait_rules"][0])   # 编号已展开成条款全文
            self.assertIn("【核心结论】", res["a1"]["analysis"])
            # 观望=没有计划可审：AI-2 直接 standby（不调 LLM），不再误报「否决」
            self.assertEqual(res["a2"]["verdict"], "standby")
            self.assertEqual(res["a3"]["summary"], "空仓，无需监控")
            # 审计已落盘
            tail = ai_council.audit_tail(1)
            self.assertTrue(tail and tail[0].get("a1"))
            # 观望跳过 AI-2、空仓跳过 AI-3 → 只打 1 次 LLM
            self.assertEqual(len(calls), 1)
        finally:
            ai_council._llm = orig
            ai_council.build_context = orig_bc

    # ---------------- 包络共振法：本地计算与体系闸门 ----------------

    @staticmethod
    def _flat_rows(n=150, px=4100.0, spread=10.0):
        """构造一组平盘 K 线：ATR 恒等于 spread，包络 = px ± 2.5×spread。"""
        return [{"t": i, "o": px, "h": px + spread / 2, "l": px - spread / 2,
                 "c": px, "v": 1} for i in range(n)]

    def test_envelope_zones(self):
        rows = self._flat_rows()
        rows[-1]["c"] = 4095.0                       # 距下沿 4075 只差 20 → ~40%
        e = ai_council._envelope(rows, "H1", 20, 0, 120, 5.0)
        self.assertIsNotNone(e)
        self.assertAlmostEqual(e["lower"], 4075.0, delta=1.0)   # 末根拉低 MA，允许 1 美金偏差
        self.assertAlmostEqual(e["upper"], 4125.0, delta=1.0)
        self.assertEqual(e["zone"], "mid")

        rows[-1]["c"] = 4078.0                       # 贴近下沿
        e = ai_council._envelope(rows, "H1", 20, 0, 120, 5.0)
        self.assertEqual(e["zone"], "lower")
        self.assertLessEqual(e["pos"], 25)

        rows[-1]["c"] = 4122.0                       # 贴近上沿
        e = ai_council._envelope(rows, "H1", 20, 0, 120, 5.0)
        self.assertEqual(e["zone"], "upper")
        self.assertGreaterEqual(e["pos"], 75)

    def test_kickoff_detects_start(self):
        rows = self._flat_rows(10)
        rows[-2]["h"] = 4106.0
        rows[-1] = {"t": 99, "o": 4100.0, "h": 4110.0, "l": 4099.0,
                    "c": 4109.0, "v": 2}            # 收阳 + 破前高 + 实体 9
        k = ai_council._kickoff(rows, "M30", atr=10.0)
        self.assertTrue(k["up"])
        self.assertFalse(k["down"])

        rows[-1] = {"t": 99, "o": 4100.0, "h": 4101.0, "l": 4099.0,
                    "c": 4100.5, "v": 2}            # 十字星：不算启动
        k = ai_council._kickoff(rows, "M30", atr=10.0)
        self.assertFalse(k["up"])

    @staticmethod
    def _gate_ctx(**kw):
        ctx = {
            "price": 4100.0, "atr_main": 10.0, "levels": [],
            "env": {"H1": {"zone": "lower", "pos": 12.0, "upper": 4125.0,
                           "mid": 4100.0, "lower": 4075.0}},
            "kickoff": {"M30": {"up": True, "down": False},
                        "M15": {"up": False, "down": False}},
            "news": {"ok": True, "label": "偏多"},
            "account": {"equity": 10000.0},
            "mtf": {"D1": {"dir": "bull"}, "H4": {"dir": "bull"}},
        }
        ctx.update(kw)
        return ctx

    @staticmethod
    def _buy_plan():
        return {"direction": "buy",
                "plan": {"entry_zone": [4075.0, 4080.0], "stop": 4070.0,
                         "targets": [{"price": 4100.0, "close_pct": 30},
                                     {"price": 4125.0, "close_pct": 40}]}}

    def test_plan_gates_clean_at_lower_band(self):
        g = ai_council._plan_gates(self._gate_ctx(), self._buy_plan())
        self.assertEqual([x["gate"] for x in g], [])

    def test_plan_gates_mid_zone_forces_wait(self):
        ctx = self._gate_ctx()
        ctx["env"]["H1"].update({"zone": "mid", "pos": 50.0})
        g = ai_council._plan_gates(ctx, self._buy_plan())
        self.assertTrue(any(x["gate"] == "区间位置" for x in g))

    def test_plan_gates_requires_kickoff(self):
        ctx = self._gate_ctx()
        ctx["kickoff"] = {"M30": {"up": False, "down": False},
                          "M15": {"up": False, "down": False}}
        g = ai_council._plan_gates(ctx, self._buy_plan())
        self.assertTrue(any(x["gate"] == "启动确认" for x in g))

    def test_plan_gates_requires_news(self):
        ctx = self._gate_ctx(news={"ok": False, "label": "无数据"})
        g = ai_council._plan_gates(ctx, self._buy_plan())
        self.assertTrue(any(x["gate"] == "情绪校验" for x in g))

    def test_plan_gates_counter_trend_needs_band_edge(self):
        # 上沿做空 vs D1/H4 全多：有区间极值加持 → 不算趋势违规
        ctx = self._gate_ctx()
        ctx["env"]["H1"].update({"zone": "upper", "pos": 90.0})
        ctx["kickoff"] = {"M30": {"up": False, "down": True},
                          "M15": {"up": False, "down": True}}
        ctx["news"] = {"ok": True, "label": "偏空"}
        sell = {"direction": "sell",
                "plan": {"entry_zone": [4120.0, 4125.0], "stop": 4130.0,
                         "targets": [{"price": 4100.0, "close_pct": 30},
                                     {"price": 4075.0, "close_pct": 40}]}}
        g = ai_council._plan_gates(ctx, sell)
        self.assertFalse(any(x["gate"] == "趋势一致" for x in g))

        # 同样逆势，但价格在中部 → 没有极值加持，必须拦
        ctx["env"]["H1"].update({"zone": "mid", "pos": 50.0})
        g = ai_council._plan_gates(ctx, sell)
        self.assertTrue(any(x["gate"] == "趋势一致" for x in g))

    def test_run_positions_calls_a3(self):
        """有持仓时应触发 AI-3 的第三次 LLM 调用（输出为自然语言+<result>）。"""
        seq = [
            # 带 plan 的 buy：AI-2 才有东西可审（否则本地 standby 跳过 LLM）
            "【核心结论】做多候选。\n<result>" + json.dumps(
                {"direction": "buy", "confidence": 60, "summary": "测试做多",
                 "plan": {"entry_zone": [4075.0, 4080.0], "stop": 4070.0,
                          "targets": [{"price": 4100.0, "close_pct": 30}]}}) + "</result>",
            "【硬闸门校验】盈亏比通过。\n<result>" + json.dumps(
                {"verdict": "allow", "risk_level": "mid", "size_advice": "0.01 手"}) + "</result>",
            "第一步：持有。\n<result>" + json.dumps(
                {"positions": [{"ticket": 1, "action": "hold", "new_stop": None,
                                "action_why": "通道内运行"}],
                 "summary": "1 笔持仓监控中"}) + "</result>",
        ]
        calls = []

        def fake_llm(messages, cfg=None, max_tokens=None, **kw):
            calls.append(1)
            return seq[len(calls) - 1]

        orig = ai_council._llm
        orig_bc = ai_council.build_context
        ai_council._llm = fake_llm
        ai_council.build_context = lambda *a, **k: {
            "price": 4100.0, "quote_age_s": 0, "event": {},
            "atr": 10.0, "atr_main": 10.0, "levels": [],
            "account": {"equity": 10000, "balance": 10000},
            "env": {"H1": {"zone": "lower", "pos": 12.0, "upper": 4125.0,
                           "mid": 4100.0, "lower": 4075.0, "atr": 10.0,
                           "width_atr": 5.0}},
            "kickoff": {"M30": {"up": True, "down": False},
                        "M15": {"up": True, "down": False}},
            "news": {"ok": True, "label": "偏多"},
            "mtf": {"D1": {"dir": "bull"}, "H4": {"dir": "bull"},
                    "H1": {"dir": "bull"}},
            "positions": [{"ticket": 1, "side": "buy", "volume": 0.01,
                           "price_open": 4105.0, "price_current": 4126.0,
                           "sl": 4070.0, "profit": 21.0, "hold_h": 2.0}],
            "gate_reasons": [], "cross": []}
        try:
            out = ai_council.run("GOLD", force=True, tf="H1")
            self.assertTrue(out["ok"])
            self.assertEqual(len(calls), 3)
            self.assertEqual(out["result"]["a3"]["summary"], "1 笔持仓监控中")
            # AI-3 机器字段已合并进本地持仓事实：健康度/通道位置是本地算的
            p = out["result"]["a3"]["positions"][0]
            self.assertEqual(p["action"], "hold")
            self.assertEqual(p["action_why"], "通道内运行")
            self.assertIn("health", p)
            self.assertIn("analysis", out["result"]["a3"])
        finally:
            ai_council._llm = orig
            ai_council.build_context = orig_bc

    # ---------------- 本地硬事实层：拆分器 / 观望条款 / 持仓通道 ----------------

    def test_split_result(self):
        txt, mach = ai_council._split_result(
            "【核心结论】观望。\n<result>{\"direction\":\"wait\",\"confidence\":30}</result>")
        self.assertEqual(mach["direction"], "wait")
        self.assertIn("观望", txt)
        # JSON 被代码围栏包裹也能解析
        _, mach2 = ai_council._split_result(
            "分析正文\n<result>```json\n{\"verdict\":\"veto\"}\n```</result>")
        self.assertEqual(mach2["verdict"], "veto")
        # 没有 <result> 标签：全文是分析，机器字段为空（不抛异常）
        txt3, mach3 = ai_council._split_result("只有自然语言，没有标签")
        self.assertEqual(mach3, {})
        self.assertIn("自然语言", txt3)

    def test_split_result_unclosed_tag_no_leak(self):
        """模型截断导致 </result> 缺失时，标签本身不能漏进正文。"""
        txt, mach = ai_council._split_result(
            "正文一句\n<result>{\"direction_bias\":\"观望\"")
        self.assertEqual(mach, {})
        self.assertNotIn("<result>", txt)
        self.assertEqual(txt, "正文一句")

    def test_split_result_lenient_braces(self):
        """JSON 前后有杂字、或被压成一行，也要能抓出来。"""
        _, m1 = ai_council._split_result(
            '<result>{"market_state":"震荡","confidence":30}</result>')
        self.assertEqual(m1["market_state"], "震荡")
        _, m2 = ai_council._split_result(
            '<result>说明：{"direction_bias":"做多"} 以上</result>')
        self.assertEqual(m2["direction_bias"], "做多")
        # 纯 JSON 输出（模型没写标签）
        txt3, m3 = ai_council._split_result('{"verdict":"allow"}')
        self.assertEqual(m3["verdict"], "allow")
        self.assertEqual(txt3, "")
        # 自然语言里带花括号数字，不应被误当成 JSON
        txt4, m4 = ai_council._split_result("价格在 {4075} 附近，观望。")
        self.assertEqual(m4, {})
        self.assertIn("4075", txt4)
        # 坏 JSON：只丢机器字段，正文保留
        txt4, mach4 = ai_council._split_result(
            "正文\n<result>{\"bad\": 1,,}</result>")
        self.assertEqual(mach4, {})
        self.assertIn("正文", txt4)

    def test_facts_wait_rules_local(self):
        """观望条款由本地判定，AI 只能引用编号。"""
        facts = ai_council._facts({"tf": "H1", "env": {"H1": {
            "zone": "mid", "pos": 50.0, "upper": 4125.0, "mid": 4100.0,
            "lower": 4075.0, "width_atr": 5.0, "atr": 10.0}},
            "kickoff": {}, "news": {"ok": False, "label": "无数据"},
            "mtf": {"D1": {"dir": "bull"}, "H4": {"dir": "bull"},
                    "H1": {"dir": "bull"}}})
        self.assertTrue(any(r.startswith("⑤") for r in facts["wait_rules"]))
        self.assertFalse(facts["candidates"]["long"])
        self.assertFalse(facts["candidates"]["short"])
        self.assertEqual(facts["resonance"]["aligned"], True)

        facts2 = ai_council._facts({"tf": "H1", "env": {"H1": {
            "zone": "upper", "pos": 92.0, "upper": 4125.0, "mid": 4100.0,
            "lower": 4075.0, "width_atr": 5.0, "atr": 10.0}},
            "kickoff": {"M30": {"up": False, "down": False}},
            "news": {"ok": True, "label": "偏多"}, "mtf": {}})
        self.assertTrue(any(r.startswith("②") for r in facts2["wait_rules"]))
        self.assertTrue(any(r.startswith("①") for r in facts2["wait_rules"]))
        self.assertTrue(facts2["candidates"]["short"])

    def test_pos_facts_channel(self):
        """AI-3 硬事实：多单顶到上沿 → 止盈区/减仓建议/移动止损。"""
        ctx = {"tf": "H1", "price": 4126.0, "atr": 10.0,
               "env": {"H1": {"zone": "upper", "pos": 100.0, "upper": 4125.0,
                              "mid": 4100.0, "lower": 4075.0, "atr": 10.0}},
               "positions": [{"ticket": 9, "side": "buy", "volume": 0.01,
                              "price_open": 4105.0, "price_current": 4126.0,
                              "profit": 21.0, "hold_h": 2.0}]}
        f = ai_council._pos_facts(ctx, {})[0]
        self.assertEqual(f["pos"], 100.0)
        self.assertEqual(f["zone"], "upper")
        self.assertEqual(f["action"], "reduce")       # 顶到止盈区
        self.assertEqual(f["tp_trigger"], 4125.0)     # 上沿=止盈触发
        self.assertGreaterEqual(f["health"], 60)      # 浮盈垫厚+距破位远：健康
        self.assertEqual(f["action"], "reduce")       # 但已到止盈区 → 动作是减仓
        self.assertTrue(f["trail"]["move_to_be"])
        self.assertIsNotNone(f["trail"]["new_stop"])
        # 空单镜像：砸到下沿=止盈区（env 也要跟着价格切到下沿）
        ctx2 = dict(ctx)
        ctx2["env"] = {"H1": {"zone": "lower", "pos": 0.0, "upper": 4125.0,
                              "mid": 4100.0, "lower": 4075.0, "atr": 10.0}}
        ctx2["positions"] = [{"ticket": 8, "side": "sell", "volume": 0.01,
                              "price_open": 4080.0, "price_current": 4074.0,
                              "profit": 6.0, "hold_h": 1.0}]
        f2 = ai_council._pos_facts(ctx2, {})[0]
        self.assertEqual(f2["action"], "reduce")
        self.assertEqual(f2["tp_trigger"], 4075.0)


    # ---------------- 本地裁判 / 1R 铁律 / 状态识别 ----------------

    def test_enforce_blocks_ai_buy_without_kickoff(self):
        """AI 说做多，但 M30/M15 没有启动 → 本地强制改观望并留痕。"""
        ctx = self._gate_ctx()
        ctx["kickoff"] = {"M30": {"up": False, "down": False},
                          "M15": {"up": False, "down": False}}
        facts = ai_council._facts(ctx)
        a1 = self._buy_plan()
        a1["confidence"] = 80
        out = ai_council._enforce_a1(a1, facts)
        self.assertEqual(out["direction"], "wait")
        self.assertEqual(out["ai_direction"], "buy")
        self.assertIsNone(out["plan"])
        self.assertLessEqual(out["confidence"], 39)
        self.assertTrue(any("启动" in b for b in out["enforced"]["blocked"]))

    def test_enforce_blocks_when_dual_plan_fails(self):
        """02 面板的双方向方案不达标时，AI 连提方向都不许。"""
        ctx = self._gate_ctx()
        ctx["plans"] = {"long": {"label": "做多方案", "rr": 1.05,
                                 "n_ok": 2, "qualify": False,
                                 "block_reason": "盈亏比不足"}}
        facts = ai_council._facts(ctx)
        a1 = self._buy_plan()
        out = ai_council._enforce_a1(a1, facts)
        self.assertEqual(out["direction"], "wait")
        self.assertTrue(any("未达标" in b for b in out["enforced"]["blocked"]))

    def test_enforce_passes_complete_setup(self):
        """区间下沿 + 双确认齐 + 计划完整 → 不改方向。"""
        facts = ai_council._facts(self._gate_ctx())
        out = ai_council._enforce_a1(self._buy_plan(), facts)
        self.assertEqual(out["direction"], "buy")
        self.assertNotIn("enforced", out)

    def test_sentiment_switch_on_blocks_opposite_news(self):
        """「必须消息同向」默认开启：消息与方向相反 → 本地拦截。"""
        ctx = self._gate_ctx(news={"ok": True, "label": "偏空"})
        facts = ai_council._facts(ctx, {"require_sentiment": True})
        self.assertTrue(facts["require_sentiment"])
        self.assertFalse(facts["sentiment"]["matched"])
        out = ai_council._enforce_a1(self._buy_plan(), facts)
        self.assertEqual(out["direction"], "wait")
        self.assertTrue(any("消息面" in b for b in out["enforced"]["blocked"]))

    def test_sentiment_switch_off_allows_entry(self):
        """关掉「必须消息同向」：消息不同向只作参考，不拦（启动确认仍必须）。"""
        ctx = self._gate_ctx(news={"ok": True, "label": "偏空"})
        facts = ai_council._facts(ctx, {"require_sentiment": False})
        self.assertFalse(facts["require_sentiment"])
        self.assertTrue(facts["sentiment"]["ok"])      # ok = 匹配 或 不强制
        out = ai_council._enforce_a1(self._buy_plan(), facts)
        self.assertEqual(out["direction"], "buy")
        self.assertNotIn("enforced", out)

    def test_sentiment_gate_respects_switch(self):
        """体系闸门「情绪校验」跟着开关走：开=命中，关=不出现。"""
        ctx = self._gate_ctx(news={"ok": True, "label": "偏空"})
        base = {"min_rr": 1.5, "risk_pct": 0.5, "atr_period": 14}
        on = ai_council._plan_gates(ctx, self._buy_plan(),
                                    dict(base, require_sentiment=True))
        self.assertTrue(any(x["gate"] == "情绪校验" for x in on))
        off = ai_council._plan_gates(ctx, self._buy_plan(),
                                     dict(base, require_sentiment=False))
        self.assertFalse(any(x["gate"] == "情绪校验" for x in off))

    def test_cfg_follows_assist_min_rr(self):
        """三审的盈亏比门槛跟 02 面板同源，不再各持一套值。"""
        cfg = ai_council._cfg()
        self.assertEqual(cfg["min_rr"], 1.5)
        self.assertIn("require_sentiment", cfg)
        self.assertTrue(cfg["require_sentiment"])

    def test_pos_facts_1r_rules(self):
        """1R 铁律：到 1R 止损移保本；到 -1R 直接离场（本地判，不靠 LLM）。"""
        base = {"tf": "H1", "price": 4102.0, "atr": 50.0,
                "env": {"H1": {"zone": "mid", "pos": 50.0, "upper": 4300.0,
                               "mid": 4200.0, "lower": 4100.0, "atr": 50.0}},
                "positions": [{"ticket": 1, "side": "buy", "volume": 0.01,
                               "price_open": 4090.0, "price_current": 4102.0,
                               "sl": 4080.0, "profit": 12.0, "hold_h": 1.0}]}
        f = ai_council._pos_facts(base, {})[0]
        self.assertAlmostEqual(f["r_now"], 1.2, places=2)
        self.assertTrue(f["trail"]["move_to_be"])
        self.assertEqual(f["trail"]["new_stop"], 4090.0)     # 移成本价
        # 反向 1R → 无条件平仓
        bad = dict(base)
        bad["price"] = 4078.0
        bad["positions"] = [{"ticket": 1, "side": "buy", "volume": 0.01,
                             "price_open": 4090.0, "price_current": 4078.0,
                             "sl": 4080.0, "profit": -12.0, "hold_h": 1.0}]
        f2 = ai_council._pos_facts(bad, {})[0]
        self.assertLess(f2["r_now"], -1)
        self.assertEqual(f2["action"], "close")

    def test_market_state_local(self):
        ctx = {"adx": {"value": 30, "trend_line": 25, "range_line": 20},
               "atr_main": 10.0, "day": {"high": 4130.0, "low": 4100.0}}
        self.assertEqual(ai_council._market_state(ctx)["state"], "趋势")
        ctx2 = {"adx": {"value": 15, "trend_line": 25, "range_line": 20},
                "atr_main": 10.0, "day": {"high": 4140.0, "low": 4090.0}}
        self.assertEqual(ai_council._market_state(ctx2)["state"], "极度无序")
        ctx3 = {"adx": {"value": 18, "trend_line": 25, "range_line": 20},
                "atr_main": 10.0, "day": {"high": 4105.0, "low": 4098.0}}
        self.assertEqual(ai_council._market_state(ctx3)["state"], "震荡")

    def test_plan_lots_local(self):
        """手数本地反算：净值 10000 × 0.5% = 50 美金；入场中值 4095、止损 4080
        → 每手亏 1500 美金 → 0.03 手（向下取整到 0.01）。"""
        ctx = {"price": 4100.0, "account": {"equity": 10000.0}}
        lots = ai_council._plan_lots(
            ctx, {"entry_zone": [4090.0, 4100.0], "stop": 4080.0},
            {"risk_pct": 0.5})
        self.assertEqual(lots["lots"], 0.03)
        self.assertEqual(lots["risk_usd"], 50.0)


if __name__ == "__main__":
    unittest.main()
