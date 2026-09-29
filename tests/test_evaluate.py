"""Tests for common/evaluate.py. Run from the repository root: pytest tests/"""

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import evaluate as E

TYPES = ["BCLM", "CRLM", "HCC", "HH", "ICC"]


def seg_run(tmp_path, name, dice, types=None):
    d = tmp_path / name
    d.mkdir()
    n = len(dice)
    pd.DataFrame({"case_id": [f"c{i:03d}" for i in range(n)],
                  "tumor_type": types or [TYPES[i % 5] for i in range(n)],
                  "Dice": dice, "Recall": np.clip(np.asarray(dice) + 0.05, 0, 1)}
                 ).to_csv(d / "per_case_test.csv", index=False)
    return d


def cls_run(tmp_path, name, true, pred):
    d = tmp_path / name
    d.mkdir()
    rows = []
    for i, (t, p) in enumerate(zip(true, pred)):
        probs = {f"p_{c}": (0.6 if c == p else 0.1) for c in TYPES}
        rows.append({"case_id": f"c{i:03d}", "true": t, "pred": p, **probs})
    pd.DataFrame(rows).to_csv(d / "per_case_test.csv", index=False)
    return d


def test_ci_covers_true_mean_at_nominal_rate():
    rng = np.random.default_rng(1)
    hits = 0
    for s in range(200):
        x = rng.normal(0.7, 0.1, 78)
        lo, hi = E.ci(E.bootstrap(lambda i: x[i].mean(), len(x), 500, seed=s))
        hits += lo <= 0.7 <= hi
    assert 0.88 <= hits / 200 <= 0.99


def test_ci_width_matches_normal_theory():
    x = np.random.default_rng(2).normal(0.7, 0.1, 78)
    lo, hi = E.ci(E.bootstrap(lambda i: x[i].mean(), len(x)))
    expected = 2 * 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert abs((hi - lo) - expected) / expected < 0.15


def test_identical_seg_runs_have_zero_difference(tmp_path):
    dice = np.random.default_rng(3).uniform(0.5, 0.95, 40)
    a, b = seg_run(tmp_path, "a", dice), seg_run(tmp_path, "b", dice)
    res, info = E.compare(a, b)
    r = res.set_index("metric").loc["Dice"]
    assert info["n_common"] == 40 and r["diff"] == 0
    assert r["p_boot"] == 1.0 and r["p_wilcoxon"] == 1.0
    assert r["better"] == 0 and r["worse"] == 0


def test_constant_improvement_is_detected(tmp_path):
    dice = np.random.default_rng(4).uniform(0.5, 0.9, 40)
    a, b = seg_run(tmp_path, "a", dice), seg_run(tmp_path, "b", dice + 0.05)
    r = E.compare(a, b)[0].set_index("metric").loc["Dice"]
    assert r["diff"] == pytest.approx(0.05)
    assert r["ci_low"] > 0 and r["p_boot"] < 0.01 and r["p_wilcoxon"] < 0.01
    assert r["better"] == 40


def test_compare_uses_only_common_cases(tmp_path):
    a = seg_run(tmp_path, "a", [0.8] * 10)
    b = seg_run(tmp_path, "b", [0.9] * 12)
    res, info = E.compare(a, b)
    assert info["n_common"] == 10 and info["only_in_b"] == ["c010", "c011"]
    assert res.set_index("metric").loc["Dice", "diff"] == pytest.approx(0.1)


def test_seg_summary_by_type(tmp_path):
    run = seg_run(tmp_path, "a", np.linspace(0.5, 0.9, 50))
    kind, df, names = E.load_run(run)
    res = E.summarize(kind, df, names, by_type=True, n_boot=300)
    assert kind == "seg"
    assert set(res["group"]) == {"all", *TYPES}
    allrow = res[(res.group == "all") & (res.metric == "Dice")].iloc[0]
    assert allrow["mean"] == pytest.approx(0.7) and allrow["ci_low"] < 0.7 < allrow["ci_high"]


def test_classification_metrics_and_mcnemar(tmp_path):
    true = [TYPES[i % 5] for i in range(50)]
    wrong = lambda t: TYPES[(TYPES.index(t) + 1) % 5]
    pred_a = [t if i >= 20 else wrong(t) for i, t in enumerate(true)]   # 30/50 correct
    pred_b = [t if i >= 5 else wrong(t) for i, t in enumerate(true)]    # 45/50 correct
    a, b = cls_run(tmp_path, "a", true, pred_a), cls_run(tmp_path, "b", true, pred_b)
    kind, df, names = E.load_run(a)
    assert kind == "cls" and names == TYPES
    assert E.cls_metrics(df, names)["accuracy"] == pytest.approx(0.6)
    r = E.compare(a, b)[0].set_index("metric").loc["accuracy"]
    assert r["diff"] == pytest.approx(0.3)
    assert r["better"] == 15 and r["worse"] == 0 and r["p_mcnemar"] < 0.001


def test_loads_stage3_paper_probs_json(tmp_path):
    d = tmp_path / "paper"
    d.mkdir()
    (d / "stage3_paper_vit_b16_probs.json").write_text(json.dumps({
        "class_names": TYPES, "labels": {"x1": 0, "x2": 3},
        "test": {"x1": [0.7, 0.1, 0.1, 0.05, 0.05], "x2": [0.1, 0.6, 0.1, 0.1, 0.1]}}))
    kind, df, names = E.load_run(d)
    assert kind == "cls" and len(df) == 2
    assert E.cls_metrics(df, names)["accuracy"] == 0.5


def test_cli_summary_writes_json(tmp_path, capsys):
    run = seg_run(tmp_path, "a", [0.8, 0.9, 0.7, 0.85])
    out = tmp_path / "report.json"
    E.main(["summary", str(run), "--n-boot", "200", "--md", "--out", str(out)])
    assert "| group | metric |" in capsys.readouterr().out
    payload = json.loads(out.read_text())
    assert payload["kind"] == "seg" and payload["results"][0]["n"] == 4
