"""本地 FinBERT 金融新闻情绪（可选增强，ProsusAI/finbert）。

设计边界：
  * 只给「这条新闻文字偏正面/负面/中性」打标签，不做涨跌预测、不给买卖建议。
  * 模型只支持英文；中文标题直接跳过，由调用方沿用原有关键词规则。
  * 模型/权重/依赖任一缺失都返回 enabled=false，调用方必须显示「关键词规则」，
    不得把降级结果包装成模型结果。
  * 结果按文本缓存，避免 10 秒轮询反复跑 CPU 推理。
"""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time
from typing import Iterable

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE / "models" / "finbert"
STATUS = HERE / "models" / "finbert_status.json"
MODEL_ID = "ProsusAI/finbert"

_lock = threading.Lock()
_cache: dict[str, dict] = {}
_tok = None
_mdl = None
_labels: list[str] = []
_ready = False
_error = ""


def status() -> dict:
    """给页面看的真实状态：不猜、不美化。"""
    st = {}
    if STATUS.is_file():
        try:
            st = json.loads(STATUS.read_text("utf-8")) or {}
        except (OSError, ValueError):
            st = {}
    has_weight = any((MODEL / f).is_file() for f in ("pytorch_model.bin", "model.safetensors"))
    ok = bool(st.get("ok")) and (MODEL / "config.json").is_file() and has_weight
    return {
        "enabled": ok,
        "model": MODEL_ID,
        "torch": st.get("torch"),
        "transformers": st.get("transformers"),
        "error": (st.get("error") or _error or "") if not ok else "",
        "note": "仅英文标题；中文新闻沿用关键词规则" if ok else "模型未就绪，逐条信号使用关键词规则",
    }


def _load() -> bool:
    global _tok, _mdl, _labels, _ready, _error
    if _ready:
        return True
    if not status()["enabled"]:
        _error = "模型未就绪"
        return False
    try:
        # 权重全在本地，强制离线：不去连 hf.co，也不写全局 HF 缓存目录
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", str(HERE / "models" / ".hf"))
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        _mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL), local_files_only=True)
        _mdl.eval()
        _labels = [_mdl.config.id2label.get(i, str(i)) for i in range(_mdl.config.num_labels)]
        _ready = True
        return True
    except Exception as e:  # noqa: BLE001
        _error = "%s: %s" % (type(e).__name__, str(e)[:200])
        return False


def _english(text: str) -> bool:
    if not text:
        return False
    letters = sum(1 for c in text if c.isascii() and c.isalpha())
    return letters >= 8 and letters / max(1, len(text)) >= 0.45


def _compute(titles: Iterable[str]) -> dict:
    """真正跑模型的地方：只由后台刷新线程调用，耗时约 15~35 秒（CPU）。"""
    st = status()
    texts = [str(t or "").strip() for t in titles]
    eng = [t for t in texts if _english(t) and t not in _cache]
    if st["enabled"] and eng:
        with _lock:                      # 加载与推理共用一把锁，避免并发重复载入 438MB 权重
            if not _load():
                eng = []
            else:
                try:
                    import torch
                    for i in range(0, len(eng), 8):
                        batch = eng[i:i + 8]
                        enc = _tok(batch, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt")
                        with torch.no_grad():
                            probs = _mdl(**enc).logits.softmax(-1).tolist()
                        for txt, row in zip(batch, probs):
                            k = max(range(len(row)), key=lambda j: row[j])
                            _cache[txt] = {"label": str(_labels[k]).lower(),
                                           "score": round(float(row[k]), 3)}
                except Exception as e:  # noqa: BLE001
                    _error = "%s: %s" % (type(e).__name__, str(e)[:200])
    scored = {t: _cache[t] for t in texts if t in _cache}
    up = sum(1 for v in scored.values() if v["label"] == "positive")
    dn = sum(1 for v in scored.values() if v["label"] == "negative")
    nu = len(scored) - up - dn
    if not scored:
        lab, summ = "—", "本批没有可打分的英文标题，逐条信号沿用关键词规则。"
    else:
        lab = "偏多" if up > dn * 1.3 else ("偏空" if dn > up * 1.3 else "中性")
        summ = ("FinBERT 对 %d 条英文标题：偏多 %d / 偏空 %d / 中性 %d → 消息情绪%s。"
                "这是对已发生文字的分类，不是涨跌预测。") % (len(scored), up, dn, nu, lab)
    out = dict(st)
    out.update({
        "n_scored": len(scored), "n_up": up, "n_down": dn, "n_neutral": nu,
        "label": lab, "summary": summ, "scores": scored,
        "source": "finbert" if scored else "keyword",
        "model_note": "仅英文标题；中文新闻沿用关键词规则，不混算。",
    })
    return out


# ---------------------------------------------------------------- 结果缓存
# 关键：推理放在独立进程（finbert_worker.py）里跑。
# 服务进程曾被 EA 工厂常驻回测占住 GIL，后台线程里 import torch /
# 载入 438MB 权重会与 Windows DLL 载入锁互相卡住，页面永远等不到结果。
# 所以：worker 算完写文件，HTTP 请求只读文件，不 import torch、不跑推理。
CACHE = HERE / "data" / "finbert_cache.json"
MAX_AGE = 900.0          # 结果超过 15 分钟视为过期，页面明确降级


def score(titles: Iterable[str]) -> dict:
    """供独立 worker 进程调用：真正跑模型，返回本次结果（不写文件）。"""
    return _compute(titles)


def write_cache(result: dict) -> None:
    """worker 调用：把本次结果原子写盘，供服务进程读取。"""
    payload = {"at": time.time(), "result": result}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    tmp.replace(CACHE)


def fallback(reason: str) -> dict:
    """公开的降级结果构造，供 worker 在模型未就绪时落盘。"""
    out = dict(status())
    out.update({
        "n_scored": 0, "n_up": 0, "n_down": 0, "n_neutral": 0,
        "label": "—", "summary": reason, "scores": {}, "source": "keyword",
        "model_note": "仅英文标题；中文新闻沿用关键词规则，不混算。",
    })
    return out


def analyze(titles: Iterable[str] | None = None) -> dict:
    """请求路径：只读 worker 落盘的结果；没有 / 过期就明确降级，不阻塞、不假装。"""
    try:
        payload = json.loads(CACHE.read_text("utf-8"))
        if time.time() - float(payload.get("at") or 0) <= MAX_AGE and payload.get("result"):
            return dict(payload["result"])
    except (OSError, ValueError, TypeError):
        pass
    return fallback("情绪模型独立进程正在计算（首次约 40 秒），本轮逐条信号沿用关键词规则。")
