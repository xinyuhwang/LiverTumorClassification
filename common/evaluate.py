"""
evaluate.py — patient-level statistics for HERALD runs (E02).

Every pipeline writes one row per test case; this script turns those rows
into numbers that can be compared across versions:
  summary   mean / SD / median of each metric with a patient-level bootstrap
            95% CI (resampling patients, as in OrganLens), optionally per
            tumor type or tumor size (--size-bins); segmentation runs also get
            global Dice (all voxels pooled) with its CI
  compare   paired comparison of two runs on their common cases: mean
            difference with bootstrap 95% CI and p-value, plus a Wilcoxon
            signed-rank test (segmentation) or exact McNemar test
            (classification), and how many cases got better / worse

Inputs (a run directory or a file):
  segmentation    per_case_test.csv with case_id, [tumor_type], metric columns
                  (ds2net/train.py stages 1-2)
  classification  per_case_test.csv with case_id, true, pred, p_<class>
                  (ds2net/train.py stage 3), or stage3_paper_<backbone>_probs.json
                  (unet_hybrid/stage3_paper.py)

Usage
  python common/evaluate.py summary <run> [--by-type] [--md] [--out report.json]
  python common/evaluate.py compare <run_A> <run_B> [--metrics Dice Recall] [--md]
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT, SEED, ALPHA = 2000, 0, 0.05
SEG_METRICS = ["Dice", "IoU", "Precision", "Recall", "Dice_vs_union", "tumor_covered"]
CLS_METRICS = ["accuracy", "macro_f1", "macro_auc"]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_run(path):
    """Return (kind, DataFrame, class_names). kind is 'seg' or 'cls'.
    Classification frames have case_id, true, pred, p_<class> columns."""
    path = Path(path)
    if path.is_dir():
        csv = path / "per_case_test.csv"
        probs = sorted(path.glob("stage3_paper_*_probs.json"))
        path = csv if csv.exists() else (probs[0] if len(probs) == 1 else csv)
    if not path.exists():
        raise FileNotFoundError(f"No per-case results at {path} (expected "
                                "per_case_test.csv or one stage3_paper_*_probs.json)")
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        names = d["class_names"]
        rows = [{"case_id": c, "true": names[d["labels"][c]],
                 "pred": names[int(np.argmax(p))],
                 **{f"p_{n}": v for n, v in zip(names, p)}}
                for c, p in d["test"].items()]
        return "cls", pd.DataFrame(rows), names
    df = pd.read_csv(path, dtype={"case_id": str})
    if {"true", "pred"} <= set(df.columns):
        names = [c[2:] for c in df.columns if c.startswith("p_")]
        return "cls", df, names
    return "seg", df, None


# ── Metrics ───────────────────────────────────────────────────────────────────

def cls_metrics(df, names):
    """Accuracy, macro-F1 and macro one-vs-rest AUC for a classification frame."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y, p = df["true"].to_numpy(), df["pred"].to_numpy()
    out = {"accuracy": accuracy_score(y, p),
           "macro_f1": f1_score(y, p, labels=names, average="macro", zero_division=0)}
    prob_cols = [f"p_{n}" for n in names]
    present = [n for n in names if (y == n).any()]
    if names and all(c in df for c in prob_cols) and len(present) > 1:
        aucs = [roc_auc_score(y == n, df[f"p_{n}"]) for n in present]
        out["macro_auc"] = float(np.mean(aucs))
    else:
        out["macro_auc"] = float("nan")
    return out


def bootstrap(stat_fn, n, n_boot=N_BOOT, seed=SEED):
    """Evaluate stat_fn(index_array) on n_boot patient-level resamples."""
    rng = np.random.default_rng(seed)
    return np.array([stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)])


def ci(samples, alpha=ALPHA):
    s = samples[np.isfinite(samples)]
    if len(s) == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(s, 100 * alpha / 2)),
            float(np.percentile(s, 100 * (1 - alpha / 2))))


def boot_p(diffs):
    """Two-sided bootstrap p-value for H0: mean difference = 0."""
    d = diffs[np.isfinite(diffs)]
    if len(d) == 0:
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


# ── Summary ───────────────────────────────────────────────────────────────────

def global_dice(g):
    """Dataset-level Dice: 2·Σtp / Σ(pred + gt), from per-case Dice and voxel
    counts (tp = Dice · (pred + gt) / 2). Weights every voxel equally, so large
    lesions dominate; the per-case mean weights every patient equally."""
    s = g["pred_voxels"].to_numpy(float) + g["gt_voxels"].to_numpy(float)
    tp = g["Dice"].to_numpy(float) * s / 2
    return float(2 * tp.sum() / s.sum()) if s.sum() else float("nan")


def size_groups(df, bins):
    """Groups by GT volume (gt_ml column): e.g. bins (10, 50, 200) →
    <10, 10–50, 50–200, ≥200 ml."""
    edges = [0.0, *bins, float("inf")]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = df[(df["gt_ml"] >= lo) & (df["gt_ml"] < hi)]
        label = f"<{hi:g} ml" if lo == 0 else (f"≥{lo:g} ml" if hi == float("inf")
                                               else f"{lo:g}–{hi:g} ml")
        if len(g):
            out.append((label, g))
    return out


def summarize(kind, df, names, metrics=None, by_type=False, n_boot=N_BOOT,
              size_bins=None):
    rows = []
    groups = [("all", df)]
    if by_type and "tumor_type" in df:
        groups += [(t, g) for t, g in df.groupby("tumor_type")]
    if kind == "cls" and by_type:
        groups += [(f"true={t}", g) for t, g in df.groupby("true")]
    if size_bins and kind == "seg" and "gt_ml" in df:
        groups += size_groups(df, size_bins)
    for group, g in groups:
        g = g.reset_index(drop=True)
        if kind == "seg":
            for m in [c for c in (metrics or SEG_METRICS) if c in g]:
                x = g[m].to_numpy(dtype=float)
                lo, hi = ci(bootstrap(lambda i: x[i].mean(), len(x), n_boot))
                rows.append({"group": group, "metric": m, "n": len(x), "mean": x.mean(),
                             "ci_low": lo, "ci_high": hi, "sd": x.std(ddof=1) if len(x) > 1 else 0.0,
                             "median": float(np.median(x))})
            if {"Dice", "pred_voxels", "gt_voxels"} <= set(g.columns) and \
                    (metrics is None or "global_Dice" in metrics):
                lo, hi = ci(bootstrap(lambda i: global_dice(g.iloc[i]), len(g), n_boot))
                rows.append({"group": group, "metric": "global_Dice", "n": len(g),
                             "mean": global_dice(g), "ci_low": lo, "ci_high": hi,
                             "sd": float("nan"), "median": float("nan")})
        else:
            point = cls_metrics(g, names)
            boots = bootstrap(lambda i: cls_metrics(g.iloc[i], names), len(g), n_boot)
            for m in metrics or CLS_METRICS:
                lo, hi = ci(np.array([b[m] for b in boots]))
                rows.append({"group": group, "metric": m, "n": len(g), "mean": point[m],
                             "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# ── Paired comparison ─────────────────────────────────────────────────────────

def compare(run_a, run_b, metrics=None, n_boot=N_BOOT):
    kind_a, a, names = load_run(run_a)
    kind_b, b, _ = load_run(run_b)
    if kind_a != kind_b:
        raise ValueError(f"Cannot compare a {kind_a} run with a {kind_b} run")
    ids_a, ids_b = set(a["case_id"]), set(b["case_id"])
    common = sorted(ids_a & ids_b)
    if not common:
        raise ValueError("The two runs share no test cases")
    a = a.set_index("case_id").loc[common].reset_index()
    b = b.set_index("case_id").loc[common].reset_index()
    info = {"n_common": len(common), "only_in_a": sorted(ids_a - ids_b),
            "only_in_b": sorted(ids_b - ids_a)}
    rows = []
    if kind_a == "seg":
        for m in metrics or [c for c in SEG_METRICS if c in a and c in b]:
            xa, xb = a[m].to_numpy(float), b[m].to_numpy(float)
            d = xb - xa
            boots = bootstrap(lambda i: d[i].mean(), len(d), n_boot)
            lo, hi = ci(boots)
            nz = d[d != 0]
            wil = stats.wilcoxon(nz).pvalue if len(nz) >= 1 else 1.0
            rows.append({"metric": m, "n": len(d), "a": xa.mean(), "b": xb.mean(),
                         "diff": d.mean(), "ci_low": lo, "ci_high": hi,
                         "p_boot": boot_p(boots), "p_wilcoxon": float(wil),
                         "better": int((d > 0).sum()), "worse": int((d < 0).sum())})
    else:
        pa, pb = cls_metrics(a, names), cls_metrics(b, names)
        boots = bootstrap(lambda i: {k: cls_metrics(b.iloc[i], names)[k]
                                     - cls_metrics(a.iloc[i], names)[k] for k in CLS_METRICS},
                          len(a), n_boot)
        ca = (a["true"] == a["pred"]).to_numpy()
        cb = (b["true"] == b["pred"]).to_numpy()
        only_b, only_a = int((cb & ~ca).sum()), int((ca & ~cb).sum())
        mcnemar = (stats.binomtest(only_b, only_b + only_a, 0.5).pvalue
                   if only_a + only_b else 1.0)
        for m in metrics or CLS_METRICS:
            diffs = np.array([x[m] for x in boots])
            lo, hi = ci(diffs)
            rows.append({"metric": m, "n": len(a), "a": pa[m], "b": pb[m],
                         "diff": pb[m] - pa[m], "ci_low": lo, "ci_high": hi,
                         "p_boot": boot_p(diffs),
                         "p_mcnemar": float(mcnemar) if m == "accuracy" else float("nan"),
                         "better": only_b if m == "accuracy" else None,
                         "worse": only_a if m == "accuracy" else None})
    return pd.DataFrame(rows), info


# ── Output ────────────────────────────────────────────────────────────────────

def to_markdown(df, digits=4):
    fmt = lambda v: ("" if v is None or (isinstance(v, float) and np.isnan(v))
                     else f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def show(df, md):
    if md:
        print(to_markdown(df))
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df.round(4).to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="metric means with bootstrap 95% CIs")
    s.add_argument("run")
    s.add_argument("--by-type", action="store_true", help="also per tumor type")
    s.add_argument("--size-bins", type=float, nargs="*", metavar="ML",
                   help="also by GT volume, e.g. --size-bins 10 50 200 (needs a gt_ml column)")
    c = sub.add_parser("compare", help="paired comparison of two runs (B − A)")
    c.add_argument("run_a"); c.add_argument("run_b")
    for q in (s, c):
        q.add_argument("--metrics", nargs="*")
        q.add_argument("--n-boot", type=int, default=N_BOOT)
        q.add_argument("--md", action="store_true", help="print a Markdown table")
        q.add_argument("--out", help="also write results as JSON")
    args = p.parse_args(argv)

    if args.cmd == "summary":
        kind, df, names = load_run(args.run)
        res = summarize(kind, df, names, args.metrics, args.by_type, args.n_boot,
                        args.size_bins)
        print(f"{args.run}: {kind}, {len(df)} cases, {args.n_boot} bootstrap resamples")
        show(res, args.md)
        payload = {"run": str(args.run), "kind": kind, "n_boot": args.n_boot,
                   "results": res.to_dict(orient="records")}
    else:
        res, info = compare(args.run_a, args.run_b, args.metrics, args.n_boot)
        print(f"B − A on {info['n_common']} common cases "
              f"({len(info['only_in_a'])} only in A, {len(info['only_in_b'])} only in B), "
              f"{args.n_boot} bootstrap resamples\n  A = {args.run_a}\n  B = {args.run_b}")
        show(res, args.md)
        payload = {"run_a": str(args.run_a), "run_b": str(args.run_b), **info,
                   "n_boot": args.n_boot, "results": res.to_dict(orient="records")}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, default=str))


if __name__ == "__main__":
    main()
