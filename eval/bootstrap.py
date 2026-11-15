"""Confidence intervals for the eval metrics, from per-cell dumps.

Uses dumps from ``eval/score.py --dump`` or ``eval/ladder.py --dump``.
Segmentation and SHACL conformance use Wilson intervals; F1 and mean violations
use percentile bootstrap intervals, resampling cells with replacement.

Two dumps also produce paired intervals for the difference (second minus first).
They must contain the same cells in the same order within each level; both use
one shared matrix of resampled cell indices.

Usage::

    # single dump, all rungs it contains
    uv run python eval/bootstrap.py ladder_ood_base.percell.jsonl

    # paired: fine-tuned vs base, per rung (delta = second - first)
    uv run python eval/bootstrap.py ladder_ood_base.percell.jsonl \\
        ladder_ood_peft.percell.jsonl --level L1 --level L4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

# Micro-F1 = 2·Σtp / (2·Σtp + Σfp + Σfn); the entity variant pools tp against
# pred/gold totals (equivalent, since fp = pred - tp and fn = gold - tp).
BOOT_METRICS = ("entity_f1", "class_f1", "mean_violations")
PROP_METRICS = (("segmentation", "seg_ok"), ("shacl_conformance", "conforms"))


def load_dump(path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load a per-cell dump into ``{level: {field: int array}}``.

    Rows without a ``level`` (a plain ``score.py --dump``) are grouped under
    ``"all"``. Rows are kept in file order so two dumps align cell-for-cell.
    """
    rows_by_level: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows_by_level[row.get("level", "all")].append(row)
    out: dict[str, dict[str, np.ndarray]] = {}
    fields = ("cls_tp", "cls_fp", "cls_fn", "ent_tp", "ent_pred", "ent_gold",
              "seg_ok", "conforms", "n_violations")
    for level, rows in rows_by_level.items():
        out[level] = {f: np.array([r[f] for r in rows], dtype=np.int64) for f in fields}
    return out


def _entity_f1(a: dict, idx: np.ndarray) -> np.ndarray:
    tp = a["ent_tp"][idx].sum(-1)
    denom = a["ent_pred"][idx].sum(-1) + a["ent_gold"][idx].sum(-1)
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def _class_f1(a: dict, idx: np.ndarray) -> np.ndarray:
    tp = a["cls_tp"][idx].sum(-1)
    denom = 2 * tp + a["cls_fp"][idx].sum(-1) + a["cls_fn"][idx].sum(-1)
    return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)


def _mean(field: str):
    return lambda a, idx: a[field][idx].mean(-1)


ESTIMATORS = {
    "entity_f1": _entity_f1,
    "class_f1": _class_f1,
    "mean_violations": _mean("n_violations"),
    "segmentation": _mean("seg_ok"),
    "shacl_conformance": _mean("conforms"),
}


def wilson(k: int, n: int, z: float) -> tuple[float, float, float]:
    """Wilson score interval for k successes in n trials. Returns (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def boot_ci(a: dict, metric: str, idx: np.ndarray, conf: float) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a pooled metric, given a (B, n) index matrix."""
    n = len(a["ent_tp"])
    point = float(ESTIMATORS[metric](a, np.arange(n)[None, :])[0])
    dist = ESTIMATORS[metric](a, idx)
    lo, hi = np.percentile(dist, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return point, float(lo), float(hi)


def paired_delta_ci(a: dict, b: dict, metric: str, idx: np.ndarray, conf: float):
    """Paired bootstrap CI of (b - a) for a metric, sharing one index matrix."""
    n = len(a["ent_tp"])
    full = np.arange(n)[None, :]
    point = float(ESTIMATORS[metric](b, full)[0] - ESTIMATORS[metric](a, full)[0])
    dist = ESTIMATORS[metric](b, idx) - ESTIMATORS[metric](a, idx)
    lo, hi = np.percentile(dist, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return point, float(lo), float(hi)


def _fmt(point: float, lo: float, hi: float) -> str:
    return f"{point:7.4f}   [{lo:.4f}, {hi:.4f}]"


def report_single(name: str, a: dict, idx: np.ndarray, conf: float, z: float) -> None:
    n = len(a["ent_tp"])
    print(f"\n{name}  (n={n})")
    for m in BOOT_METRICS:
        p, lo, hi = boot_ci(a, m, idx, conf)
        print(f"  {m:20s} {_fmt(p, lo, hi)}   (bootstrap)")
    for m, key in PROP_METRICS:
        p, lo, hi = wilson(int(a[key].sum()), n, z)
        print(f"  {m:20s} {_fmt(p, lo, hi)}   (Wilson)")


def report_paired(a: dict, b: dict, idx: np.ndarray, conf: float) -> None:
    print(f"\n  delta (2nd - 1st), paired bootstrap:")
    for m in ("entity_f1", "class_f1", "segmentation", "shacl_conformance", "mean_violations"):
        p, lo, hi = paired_delta_ci(a, b, m, idx, conf)
        sig = "  *sig* " if (lo > 0 or hi < 0) else "        "
        print(f"  {m:20s} {_fmt(p, lo, hi)}{sig}(excludes 0: {lo > 0 or hi < 0})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap / Wilson CIs from per-cell dumps.")
    ap.add_argument("dumps", nargs="+", help="one dump, or two (base then fine-tuned) for paired deltas")
    ap.add_argument("--iters", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--conf", type=float, default=0.95, help="confidence level (default 0.95)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for resampling")
    ap.add_argument("--level", action="append", default=None,
                    help="restrict to these rung levels (repeatable); default: all present")
    args = ap.parse_args()

    if len(args.dumps) > 2:
        print("error: pass at most two dumps", file=sys.stderr)
        return 2

    z = NormalDist().inv_cdf(1 - (1 - args.conf) / 2)
    rng = np.random.default_rng(args.seed)
    data = [load_dump(p) for p in args.dumps]
    names = [Path(p).name for p in args.dumps]

    levels = sorted(set(data[0]) if not args.level else args.level)
    print(f"iters={args.iters}  conf={args.conf}  seed={args.seed}")

    for level in levels:
        if level not in data[0]:
            print(f"\n[{level}] not in {names[0]} — skipping")
            continue
        print(f"\n{'=' * 60}\nrung {level}\n{'=' * 60}")
        a = data[0][level]
        n = len(a["ent_tp"])
        idx = rng.integers(0, n, size=(args.iters, n))  # one index matrix, reused
        report_single(names[0], a, idx, args.conf, z)

        if len(data) == 2:
            if level not in data[1]:
                print(f"\n{names[1]}: rung {level} absent — no paired delta")
                continue
            b = data[1][level]
            if len(b["ent_tp"]) != n:
                print(f"\nskip paired delta: cell counts differ ({n} vs {len(b['ent_tp'])})")
                continue
            report_single(names[1], b, idx, args.conf, z)  # same idx → comparable
            report_paired(a, b, idx, args.conf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
