"""Evaluate the parser against a labeled set (default: ``eval/labeled.jsonl``).

Reports class precision/recall/F1 over class multisets, entity precision/recall/F1
using gold-subset matching, and segmentation accuracy (matching entity counts).
A prediction matches a gold entity when all specified gold fields agree;
additional predicted fields are ignored.

Also reports SHACL conformance (fraction of conforming cells) and mean SHACL
violations per cell.

    uv run python eval/score.py

Use ``eval/ladder.py`` to compare parser configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semparse.infer import parse_cell  # noqa: E402
from semparse.profiles import get_profile  # noqa: E402
from semparse.validate import validate_result  # noqa: E402

LABELED = Path(__file__).with_name("labeled.jsonl")
SCORES_OUT = Path(__file__).with_name("last_scores.json")


def resolve_set(name: str | None) -> Path:
    """Resolve an eval set to a ``.jsonl`` path.

    Precedence: explicit ``name`` > ``$SEMPARSE_EVAL_FILE`` > the default
    ``labeled.jsonl`` (the in-distribution set the push gate scores). ``name``
    may be a path, a filename in ``eval/``, or a bare tag — e.g. ``"ood"``
    resolves to ``eval/labeled_ood.jsonl`` — so ``--set ood`` is enough.
    """
    name = name or os.environ.get("SEMPARSE_EVAL_FILE")
    if not name or name in ("default", "labeled"):
        return LABELED
    p = Path(name)
    if p.exists():
        return p
    here = Path(__file__).parent
    for cand in (here / name, here / f"{name}.jsonl", here / f"labeled_{name}.jsonl"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"eval set not found: {name!r}")


def scores_path(set_path: Path) -> Path:
    """Where to write scores for ``set_path``.

    The default set keeps writing ``last_scores.json`` (the push gate reads it);
    any other set writes ``last_scores_<tag>.json`` so an out-of-distribution run
    never clobbers the gate's in-distribution baseline.
    """
    if set_path.resolve() == LABELED.resolve():
        return SCORES_OUT
    tag = set_path.stem.removeprefix("labeled_")
    return SCORES_OUT.with_name(f"last_scores_{tag}.json")


def load_examples(path: Path | None = None) -> list[dict]:
    """Load the held-out labeled cells from ``path`` (default: ``labeled.jsonl``)."""
    return [
        json.loads(line)
        for line in (path or LABELED).read_text().splitlines()
        if line.strip()
    ]

# Seed used so a given commit yields the same scores run-to-run, which is what
# makes the >=-baseline push gate (eval/gate.py) meaningful rather than flaky.
SEED = int(os.environ.get("SEMPARSE_EVAL_SEED", "0"))


def _seed_everything(seed: int) -> None:
    """Pin RNGs so sampled decoding is reproducible across runs.

    On CPU this is fully deterministic; on CUDA a few kernels may still differ,
    which is why the gate keeps a small tolerance.
    """
    import random

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Fields that define an "exact" entity match. attributes are intentionally
# excluded (free-form, fuzzy); class/role/measure/purpose/count are the core.
_FIELDS = (
    "class",
    "source",
    "sink",
    "purpose",
    "purposes",
    "power",
    "volume",
    "area",
    "quantity",
    "controllable",
)


def _satisfies(gold: dict, pred: dict) -> bool:
    """Gold-subset match: pred agrees on every field gold *specifies*.

    Only fields present (and non-null) in the gold entity are checked; a
    prediction may carry extra correct fields (e.g. a `sink`/`controllable` the
    gold omitted) without being penalised. Nested measures compare structurally.
    """
    for f in _FIELDS:
        if gold.get(f) is None:
            continue
        gv, pv = gold[f], pred.get(f)
        # purposes is a set-valued field — order doesn't matter.
        if f == "purposes":
            if set(gv) != set(pv or []):
                return False
        elif pv != gv:
            return False
    return True


def _match_entities(gold: list[dict], pred: list[dict]) -> int:
    """Greedy count of gold entities satisfied by a distinct pred entity."""
    used = [False] * len(pred)
    matched = 0
    for g in gold:
        for i, p in enumerate(pred):
            if not used[i] and _satisfies(g, p):
                used[i] = True
                matched += 1
                break
    return matched


def _pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(
    examples: list[dict],
    parse_kwargs: dict | None = None,
    *,
    verbose: bool = True,
    per_cell: list | None = None,
) -> dict:
    """Score ``examples`` under a given harness configuration.

    Forward ``parse_kwargs`` to ``parse_cell`` and return aggregate metrics.
    If ``per_cell`` is provided, append each cell's metric counts for bootstrap
    confidence intervals.
    """
    parse_kwargs = dict(parse_kwargs or {})
    model = os.environ.get("SEMPARSE_MODEL")
    if model:
        parse_kwargs.setdefault("model", model)

    cls_tp = cls_fp = cls_fn = 0
    ent_tp = ent_pred = ent_gold = 0
    seg_correct = 0
    conforming = 0
    total_violations = 0

    for i, ex in enumerate(examples):
        gold = ex["entities"]
        # A label line may set "profile" (by name) or "header" to exercise the
        # column-restricted decoding; otherwise it runs under the general profile.
        call = dict(parse_kwargs)
        if ex.get("profile"):
            call["profile"] = get_profile(ex["profile"])
        if ex.get("header"):
            call["header"] = ex["header"]
        result = parse_cell(ex["input"], **call)
        pred = [e.model_dump(by_alias=True, exclude_none=True) for e in result.entities]

        # class-level multiset P/R
        gc, pc = Counter(e["class"] for e in gold), Counter(e["class"] for e in pred)
        overlap = sum((gc & pc).values())
        c_cls_tp = overlap
        c_cls_fp = sum(pc.values()) - overlap
        c_cls_fn = sum(gc.values()) - overlap
        cls_tp += c_cls_tp
        cls_fp += c_cls_fp
        cls_fn += c_cls_fn

        # entity P/R (gold-subset match)
        m = _match_entities(gold, pred)
        ent_tp += m
        ent_pred += len(pred)
        ent_gold += len(gold)

        c_seg_ok = int(len(pred) == len(gold))
        seg_correct += c_seg_ok

        # formal-semantics: does the lifted graph satisfy the SHACL shapes?
        conforms, violations = validate_result(result)
        conforming += int(conforms)
        total_violations += len(violations)

        if per_cell is not None:
            per_cell.append({
                "i": i,
                "slice": ex.get("_slice"),
                "cls_tp": c_cls_tp,
                "cls_fp": c_cls_fp,
                "cls_fn": c_cls_fn,
                "ent_tp": m,
                "ent_pred": len(pred),
                "ent_gold": len(gold),
                "seg_ok": c_seg_ok,
                "conforms": int(conforms),
                "n_violations": len(violations),
            })

        if verbose:
            mark = "OK " if (m == len(gold) and len(pred) == len(gold) and conforms) else "DIFF"
            flag = "" if conforms else f" !shacl({len(violations)})"
            print(f"[{mark}] {ex['input'][:50]:50s} gold={len(gold)} pred={len(pred)} match={m}{flag}")
            if mark == "DIFF":
                print(f"        pred: {json.dumps(pred)}")
                for v in violations:
                    print(f"        shacl: {v}")

    n = len(examples)
    cp, cr, cf = _pr(cls_tp, cls_fp, cls_fn)
    ep, er, ef = _pr(ent_tp, ent_pred - ent_tp, ent_gold - ent_tp)
    if verbose:
        print("\n" + "=" * 60)
        print(f"cells:                 {n}")
        print(f"class      P/R/F1:     {cp:.2f} / {cr:.2f} / {cf:.2f}")
        print(f"entity     P/R/F1:     {ep:.2f} / {er:.2f} / {ef:.2f}  (gold-subset match)")
        print(f"segmentation accuracy: {seg_correct}/{n} = {seg_correct/n:.2f}")
        print(f"SHACL conformance:     {conforming}/{n} = {conforming/n:.2f}")
        print(f"mean violations/cell:  {total_violations/n:.3f}")

    return {
        "cells": n,
        "class_f1": round(cf, 4),
        "entity_f1": round(ef, 4),
        "segmentation": round(seg_correct / n, 4),
        "shacl_conformance": round(conforming / n, 4),
        "mean_violations": round(total_violations / n, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the parser against a labeled set.")
    parser.add_argument(
        "--set",
        dest="eval_set",
        default=None,
        help="eval set to score: a path, a filename in eval/, or a bare tag like "
        "'ood' (-> labeled_ood.jsonl). Default: labeled.jsonl, or $SEMPARSE_EVAL_FILE.",
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="also write per-cell metric counts (one JSONL row per cell, no "
        "aggregation) to this path, for bootstrap.py to resample.",
    )
    args = parser.parse_args()
    set_path = resolve_set(args.eval_set)
    out = scores_path(set_path)
    per_cell = [] if args.dump else None

    _seed_everything(SEED)
    print(f"eval set: {set_path.name}")
    scores = evaluate(load_examples(set_path), per_cell=per_cell)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"\nwrote {out.name}: {json.dumps(scores)}")
    if per_cell is not None:
        Path(args.dump).write_text("\n".join(json.dumps(r) for r in per_cell) + "\n")
        print(f"wrote {args.dump}: {len(per_cell)} per-cell rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
