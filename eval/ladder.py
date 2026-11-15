"""Compare parser configurations on one evaluation set.

    L1  constrained decoding + normalization
    L2  reuse L1 predictions for SHACL measurement
    L3  L1 + reasoning
    L4  L1 + SHACL repair
    L5  L1 + reasoning + repair

    uv run python eval/ladder.py

Writes ``eval/ladder.json`` by default and prints a comparison table.
``SEMPARSE_EVAL_SEED`` sets the seed for each run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from score import LABELED, SEED, _seed_everything, evaluate, load_examples, resolve_set

LADDER_OUT = Path(__file__).with_name("ladder.json")
REPAIRS = int(os.environ.get("SEMPARSE_LADDER_REPAIRS", "3"))


def ladder_path(set_path: Path) -> Path:
    """Where to write the ladder table for ``set_path`` (default: ``ladder.json``;
    any other set writes ``ladder_<tag>.json`` so runs don't clobber each other)."""
    if set_path.resolve() == LABELED.resolve():
        return LADDER_OUT
    tag = set_path.stem.removeprefix("labeled_")
    return LADDER_OUT.with_name(f"ladder_{tag}.json")

# (id, label, parse_kwargs | None). None means "reuse the previous rung's run".
# Test reasoning and repair separately before combining them.
LEVELS = [
    ("L1", "constrained + normalization", {}),
    ("L2", "+ SHACL validation (measure)", None),
    ("L3", "+ reasoning augmentation (alone)", {"reason": True}),
    ("L4", "+ closed-loop SHACL repair (alone)", {"max_repairs": REPAIRS}),
    ("L5", "+ reasoning + repair", {"reason": True, "max_repairs": REPAIRS}),
]

_COLUMNS = [
    ("entity_f1", "entity F1"),
    ("segmentation", "segment."),
    ("class_f1", "class F1"),
    ("shacl_conformance", "SHACL conf."),
    ("mean_violations", "viol./cell"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep the harness-capability ladder.")
    parser.add_argument(
        "--set",
        dest="eval_set",
        default=None,
        help="eval set to sweep: a path, a filename in eval/, or a bare tag like "
        "'ood' (-> labeled_ood.jsonl). Default: labeled.jsonl, or $SEMPARSE_EVAL_FILE.",
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="also write per-cell metric counts (one JSONL row per cell per rung, "
        "tagged with 'level', no aggregation) to this path, for bootstrap.py.",
    )
    args = parser.parse_args()
    set_path = resolve_set(args.eval_set)
    out = ladder_path(set_path)

    _seed_everything(SEED)
    print(f"eval set: {set_path.name}")
    examples = load_examples(set_path)

    rows: list[dict] = []
    dump_records: list[dict] | None = [] if args.dump else None
    last: dict | None = None
    last_percell: list[dict] | None = None
    for level_id, label, kwargs in LEVELS:
        if kwargs is None:
            # L2 reuses L1's predictions verbatim; replay its per-cell rows too.
            scores = dict(last) if last else {}
            if dump_records is not None and last_percell is not None:
                dump_records += [{**r, "level": level_id} for r in last_percell]
        else:
            # Reset the RNG so each rung starts from the same seed.
            _seed_everything(SEED)
            print(f"\n{'#' * 60}\n# {level_id}: {label}  kwargs={kwargs}\n{'#' * 60}")
            pc: list[dict] | None = [] if dump_records is not None else None
            scores = evaluate(examples, kwargs, verbose=True, per_cell=pc)
            last = scores
            if pc is not None:
                last_percell = pc
                dump_records += [{**r, "level": level_id} for r in pc]
        rows.append({"level": level_id, "label": label, **scores})

    # comparison table
    print("\n" + "=" * 84)
    print("HARNESS-CAPABILITY LADDER")
    print("=" * 84)
    header = f"{'':3s} {'capability':30s}" + "".join(f"{h:>13s}" for _, h in _COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['level']:3s} {row['label']:30s}"
        for key, _ in _COLUMNS:
            val = row.get(key)
            line += f"{val:>13.4f}" if isinstance(val, (int, float)) else f"{'-':>13s}"
        print(line)

    out.write_text(
        json.dumps({"seed": SEED, "set": set_path.name, "rows": rows}, indent=2) + "\n"
    )
    print(f"\nwrote {out.name}")
    if dump_records is not None:
        Path(args.dump).write_text(
            "\n".join(json.dumps(r) for r in dump_records) + "\n"
        )
        print(f"wrote {args.dump}: {len(dump_records)} per-cell rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
