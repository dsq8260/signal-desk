"""MT5 测试器报告导入测试（用伪造的 HTML / XML 报告）。"""

from __future__ import annotations

import pathlib

import pytest

from gold.tester_import import (
    _to_num,
    find_reports,
    parse_report,
    rank_reports,
)

HTML_CN = """<html><body>
<table>
  <tr><td>交易品种</td><td>GOLD#,H1</td></tr>
  <tr><td>初始入金</td><td>10 000.00</td></tr>
  <tr><td>总净盈利</td><td>-306.48</td></tr>
  <tr><td>毛利</td><td>216.59</td></tr>
  <tr><td>毛损</td><td>-523.07</td></tr>
  <tr><td>盈利因子</td><td>0.41</td></tr>
  <tr><td>预期收益</td><td>-7.13</td></tr>
  <tr><td>恢复因子</td><td>-0.85</td></tr>
  <tr><td>夏普比率</td><td>-5.00</td></tr>
  <tr><td>交易总计</td><td>43</td></tr>
  <tr><td>最大结余亏损</td><td>359.36 (3.59%)</td></tr>
  <tr><td>相对结余亏损</td><td>3.80% (380.72)</td></tr>
</table></body></html>"""

XML_EN = """<?xml version="1.0"?>
<Report>
  <Symbol>EURUSD#</Symbol>
  <Period>M30</Period>
  <InitialDeposit>10000.00</InitialDeposit>
  <TotalNetProfit>1442.10</TotalNetProfit>
  <GrossProfit>3200.50</GrossProfit>
  <GrossLoss>-1758.40</GrossLoss>
  <ProfitFactor>1.82</ProfitFactor>
  <ExpectedPayoff>42.41</ExpectedPayoff>
  <RecoveryFactor>3.10</RecoveryFactor>
  <SharpeRatio>0.19</SharpeRatio>
  <TotalTrades>34</TotalTrades>
  <BalanceDrawdownMaximal>465.20</BalanceDrawdownMaximal>
  <RelativeDrawdown>4.20%</RelativeDrawdown>
</Report>"""


def _write(d: pathlib.Path, name: str, body: str) -> pathlib.Path:
    p = d / name
    p.write_text(body, encoding="utf-8")
    return p


def test_to_num_formats():
    assert _to_num("1 234.56") == pytest.approx(1234.56)
    assert _to_num("-306.48") == pytest.approx(-306.48)
    assert _to_num("(0.07%)") == pytest.approx(0.07)
    assert _to_num("abc") is None


def test_parse_html_chinese(tmp_path):
    p = _write(tmp_path, "MACD_M30.html", HTML_CN)
    r = parse_report(p)
    assert r["净盈利"] == pytest.approx(-306.48)
    assert r["毛利"] == pytest.approx(216.59)
    assert r["毛损"] == pytest.approx(-523.07)
    assert r["盈利因子"] == pytest.approx(0.41)
    assert r["交易总计"] == pytest.approx(43)
    assert r["夏普"] == pytest.approx(-5.00)
    assert r["初始入金"] == pytest.approx(10000.0)


def test_parse_xml_english(tmp_path):
    p = _write(tmp_path, "Scalper_en.xml", XML_EN)
    r = parse_report(p)
    assert r["净盈利"] == pytest.approx(1442.10)
    assert r["盈利因子"] == pytest.approx(1.82)
    assert r["交易总计"] == pytest.approx(34)
    assert r["符号"] == "EURUSD#"
    assert r["周期"] == "M30"


def test_parse_missing_file(tmp_path):
    r = parse_report(tmp_path / "nope.xml")
    assert r["净盈利"] is None
    assert r["文件"] == "nope.xml"


def test_find_reports_filters(tmp_path):
    _write(tmp_path, "a.html", HTML_CN)
    _write(tmp_path, "b.xml", XML_EN)
    _write(tmp_path, "notes.txt", "hello")
    _write(tmp_path, "other.html", "<html><body>随便写的</body></html>")
    found = find_reports(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.html", "b.xml"]


def test_find_reports_missing_dir(tmp_path):
    assert find_reports(tmp_path / "nothere") == []


def test_rank_reports_sorted_by_profit(tmp_path):
    _write(tmp_path, "loss.html", HTML_CN)
    _write(tmp_path, "win.xml", XML_EN)
    df = rank_reports(tmp_path)
    assert len(df) == 2
    assert df.iloc[0]["净盈利"] > df.iloc[1]["净盈利"]
    assert set(df.columns) >= {"文件", "净盈利", "盈利因子", "交易总计"}


def test_rank_reports_empty(tmp_path):
    df = rank_reports(tmp_path)
    assert len(df) == 0
    assert "净盈利" in df.columns
