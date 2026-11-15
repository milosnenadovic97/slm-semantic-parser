"""Parse a column of free-text cells in an xlsx file into ontology entities.

Each distinct cell is parsed once. Results go to JSONL and, optionally, a new
column in a copy of the workbook.

Usage:

    uv run semparse --xlsx sample.xlsx --column Description --out out.jsonl
    uv run semparse --xlsx sample.xlsx --column Description --annotate annotated.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys

from openpyxl import load_workbook

from .infer import DEFAULT_MODEL, parse_cell
from .profiles import get_profile, select_profile


def _read_column(path: str, sheet: str | None, column: str) -> list[str]:
    """Read every cell of the named column (by header) as strings."""
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    rows = ws.iter_rows(values_only=True)
    header = [str(c) if c is not None else "" for c in next(rows)]
    if column not in header:
        raise SystemExit(
            f"column {column!r} not found; available: {', '.join(h for h in header if h)}"
        )
    idx = header.index(column)
    values = [
        str(row[idx]).strip()
        for row in rows
        if idx < len(row) and row[idx] is not None and str(row[idx]).strip()
    ]
    wb.close()
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="semparse", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--xlsx", required=True, help="path to the input workbook")
    parser.add_argument("--column", required=True, help="header of the column to parse")
    parser.add_argument("--sheet", default=None, help="sheet name (default: active)")
    parser.add_argument("--out", default="out.jsonl", help="JSONL output path")
    parser.add_argument(
        "--annotate",
        default=None,
        help="optional: write a copy of the workbook with a parsed-JSON column",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument(
        "--profile",
        default=None,
        help="force a column profile by name (default: auto-select from header)",
    )
    args = parser.parse_args(argv)

    cells = _read_column(args.xlsx, args.sheet, args.column)
    distinct = list(dict.fromkeys(cells))  # order-preserving dedupe

    profile = get_profile(args.profile) if args.profile else select_profile(args.column)
    print(
        f"read {len(cells)} non-empty cells ({len(distinct)} distinct) "
        f"from column {args.column!r} | profile: {profile.name} "
        f"(classes: {', '.join(profile.allowed_classes)})",
        file=sys.stderr,
    )

    parsed: dict[str, dict] = {}
    for i, cell in enumerate(distinct, 1):
        result, issues = parse_cell(
            cell,
            model=args.model,
            profile=profile,
            header=args.column,
            return_issues=True,
        )
        parsed[cell] = {
            "input": cell,
            "entities": [
                e.model_dump(by_alias=True, exclude_none=True) for e in result.entities
            ],
            "issues": issues,
        }
        flag = f"  [{len(issues)} issue(s)]" if issues else ""
        print(f"  {i}/{len(distinct)}: {cell[:60]}{flag}", file=sys.stderr)

    with open(args.out, "w") as fh:
        for cell in cells:  # one line per original cell, in order
            fh.write(json.dumps(parsed[cell]) + "\n")
    print(f"wrote {len(cells)} records to {args.out}", file=sys.stderr)

    if args.annotate:
        _annotate(args.xlsx, args.sheet, args.column, parsed, args.annotate)
        print(f"wrote annotated workbook to {args.annotate}", file=sys.stderr)
    return 0


def _annotate(
    path: str, sheet: str | None, column: str, parsed: dict[str, dict], out: str
) -> None:
    """Write a copy of the workbook with an `<column>_parsed` column added."""
    wb = load_workbook(path)
    ws = wb[sheet] if sheet else wb.active
    header = [c.value for c in ws[1]]
    idx = header.index(column)
    new_col = ws.max_column + 1
    ws.cell(row=1, column=new_col, value=f"{column}_parsed")
    for r in range(2, ws.max_row + 1):
        val = ws.cell(row=r, column=idx + 1).value
        key = str(val).strip() if val is not None else ""
        rec = parsed.get(key)
        if rec is not None:
            ws.cell(row=r, column=new_col, value=json.dumps(rec["entities"]))
    wb.save(out)


if __name__ == "__main__":
    raise SystemExit(main())
