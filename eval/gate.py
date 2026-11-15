"""Push gate: fail if the latest eval scores regressed below the baseline.

Reads the committed baseline (``eval/baseline.json``) and the scores written by
the most recent ``eval/score.py`` run (``eval/last_scores.json``), and exits
non-zero if any gated metric dropped by more than the tolerance. The pre-push
hook (``.githooks/pre-push``) runs ``score.py`` then this, so a regression
aborts the push.

Tolerance absorbs residual decoding non-determinism (see ``score.py``); set it
to 0 for a strict ``>=`` gate, or raise it via ``SEMPARSE_EVAL_TOL`` if you run
on hardware where seeded sampling still wiggles.

    uv run python eval/gate.py            # compare last_scores.json to baseline.json
    SEMPARSE_EVAL_TOL=0.02 uv run python eval/gate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASELINE = Path(__file__).with_name("baseline.json")
LAST = Path(__file__).with_name("last_scores.json")
# Metrics missing from either scores file are skipped.
GATED_METRICS = ("class_f1", "entity_f1", "segmentation", "shacl_conformance")
TOL = float(os.environ.get("SEMPARSE_EVAL_TOL", "0.0"))


def main() -> int:
    if not BASELINE.exists():
        print(
            f"no {BASELINE.name} yet — establish one first:\n"
            "    uv run python eval/score.py\n"
            "    cp eval/last_scores.json eval/baseline.json && git add eval/baseline.json",
            file=sys.stderr,
        )
        return 1
    if not LAST.exists():
        print(
            f"no {LAST.name} — run `uv run python eval/score.py` first.",
            file=sys.stderr,
        )
        return 1

    base = json.loads(BASELINE.read_text())
    last = json.loads(LAST.read_text())

    regressions = []
    for m in GATED_METRICS:
        b, n = base.get(m), last.get(m)
        if b is None or n is None:
            continue
        if n < b - TOL:
            regressions.append((m, b, n))

    print(f"eval gate (tolerance {TOL:g}):")
    for m in GATED_METRICS:
        b, n = base.get(m), last.get(m)
        flag = "FAIL" if (m, b, n) in regressions else "ok"
        print(f"  {m:14s} baseline={b}  new={n}  [{flag}]")

    if regressions:
        print("\nPUSH BLOCKED — eval regressed:", file=sys.stderr)
        for m, b, n in regressions:
            print(f"  {m}: {n} < baseline {b} (tol {TOL:g})", file=sys.stderr)
        print(
            "\nFix the regression, or bypass intentionally with `git push --no-verify`.\n"
            "If this run is actually an improvement, refresh the baseline:\n"
            "    cp eval/last_scores.json eval/baseline.json && git add eval/baseline.json",
            file=sys.stderr,
        )
        return 1

    print("\neval gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
