"""Chạy: python -m pytest tests -q   (hoặc python tests/test_metrics.py)
Kiểm tra công thức bằng chính các con số của ma trận nhầm lẫn trong đề cương (TP=768, FN=82, TN=830, FP=45)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import metrics as M  # noqa: E402


def make(tp, fn, tn, fp):
    y = np.array([1] * (tp + fn) + [0] * (tn + fp))
    p = np.array([1] * tp + [0] * fn + [0] * tn + [1] * fp)
    return y, p


def test_confusion_formulas():
    y, p = make(768, 82, 830, 45)
    m = M.classification_metrics(y, p)
    assert (m["TP"], m["FN"], m["TN"], m["FP"]) == (768, 82, 830, 45)
    assert math.isclose(m["Recall"], 768 / 850) and math.isclose(m["ASR"], 82 / 850)
    assert math.isclose(m["ASR"], 1 - m["Recall"])
    assert math.isclose(m["FPR"], 45 / 875)
    assert math.isclose(m["Precision"], 768 / 813)
    assert math.isclose(m["F1"], 2 * m["Precision"] * m["Recall"] / (m["Precision"] + m["Recall"]))


def test_latency_percentiles():
    s = M.latency_stats(list(range(1, 101)))
    assert s["p50"] == 50.5 and abs(s["p95"] - 95.05) < 1e-9 and s["n"] == 100


def test_sweep_monotonic_and_choose_tau():
    rng = np.random.default_rng(0)
    y = np.array([1] * 200 + [0] * 200)
    s = np.clip(np.r_[rng.normal(.7, .15, 200), rng.normal(.3, .15, 200)], 0, 1)
    sw = M.threshold_sweep(y, s)
    assert (sw["ASR"].diff().dropna() >= -1e-12).all()      # τ tăng -> ASR không giảm
    assert (sw["FPR"].diff().dropna() <= 1e-12).all()       # τ tăng -> FPR không tăng
    assert 0.3 < M.choose_tau(y, s) < 0.7


def test_mcnemar_and_bootstrap():
    a = np.array([1] * 90 + [0] * 10, bool); b = np.array([1] * 60 + [0] * 40, bool)
    assert M.mcnemar_exact(a, b)["p_value"] < 0.001
    y, p = make(768, 82, 830, 45)
    ci = M.bootstrap_ci(y, p, ("ASR",), n_boot=300)
    lo, hi = ci["ASR"]
    assert lo < 82 / 850 < hi


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
