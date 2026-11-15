"""Turn raw model output (plain dicts) into validated entities.

Parses measure_raw ("22 kWth", "1500L") into a value and unit, puts it in
power/volume/area based on the unit, and flags terms not in the ontology.
"""

from __future__ import annotations

import re
from typing import Any

from .ontology import (
    ASSET_CLASSES,
    MEDIA,
    PURPOSES,
    UNIT_ALIASES,
    canonical_unit,
    measure_field,
)
from .schema import Entity, Measure, ParseResult

# Longest spellings first so "kw(th)" wins over "kw".
_UNIT_PATTERN = "|".join(
    re.escape(u) for u in sorted(UNIT_ALIASES, key=len, reverse=True)
)
# Lookahead instead of \b so units ending in ")" match, and "kw" can't match inside "kwh".
_MEASURE_RE = re.compile(
    rf"([-+]?\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})(?![A-Za-z0-9(])",
    re.IGNORECASE,
)


def parse_measure(raw: str | None) -> tuple[str | None, Measure | None, str | None]:
    """Return (field, Measure, issue); field is "power", "volume" or "area"."""
    if raw is None or not raw.strip():
        return None, None, None
    m = _MEASURE_RE.search(raw)
    if not m:
        return None, None, f"unparseable measure span: {raw!r}"
    value = float(m.group(1))
    unit = canonical_unit(m.group(2))
    if unit is None:  # pragma: no cover - pattern is built from UNIT_ALIASES
        return None, None, f"unknown unit in: {raw!r}"
    field = measure_field(unit)
    if field is None:  # pragma: no cover - every canonical unit has a kind
        return None, None, f"unroutable unit {unit!r} in: {raw!r}"
    return field, Measure(value=value, unit=unit), None


def normalize_entity(data: dict[str, Any]) -> tuple[Entity | None, list[str]]:
    """Return (entity, issues) for one raw entity dict.

    An unknown class drops the entity. Unknown source/sink/purpose/measure
    values are cleared and flagged.
    """
    issues: list[str] = []

    cls = data.get("class") or data.get("cls")
    if cls not in ASSET_CLASSES:
        issues.append(f"unknown class: {cls!r}")
        return None, issues

    def _check(role: str, allowed: tuple[str, ...]) -> str | None:
        val = data.get(role)
        if val is None:
            return None
        if val not in allowed:
            issues.append(f"unknown {role}: {val!r}")
            return None
        return val

    source = _check("source", MEDIA)
    sink = _check("sink", MEDIA)
    purpose = _check("purpose", PURPOSES)

    raw_purposes = data.get("purposes")
    if raw_purposes is not None:
        valid, invalid = [], []
        for v in raw_purposes:
            (valid if v in PURPOSES else invalid).append(v)
        for v in invalid:
            issues.append(f"unknown purposes value: {v!r}")
        purposes: list[str] | None = valid if valid else None
    else:
        purposes = None

    field, measure, measure_issue = parse_measure(data.get("measure_raw"))
    if measure_issue:
        issues.append(measure_issue)
    measures = {"power": None, "volume": None, "area": None}
    if field is not None:
        measures[field] = measure

    attributes = data.get("attributes") or {}
    attributes = {str(k): str(v) for k, v in attributes.items()}

    entity = Entity(
        cls=cls,
        source=source,
        sink=sink,
        purpose=purpose,
        power=measures["power"],
        volume=measures["volume"],
        area=measures["area"],
        purposes=purposes,
        quantity=data.get("quantity"),
        controllable=data.get("controllable"),
        attributes=attributes,
    )
    return entity, issues


def collapse_quantities(entities: list[Entity]) -> list[Entity]:
    """Merge identical entities into one with a quantity.

    Small models often repeat an asset instead of setting quantity. N copies
    with no quantity become one entity with quantity=N. Copies that already
    share a quantity are just deduplicated. Entities that differ are kept.
    """
    groups: dict[str, list[Entity]] = {}
    order: list[str] = []
    for e in entities:
        key = e.model_dump_json(by_alias=True, exclude_none=True)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)

    collapsed: list[Entity] = []
    for key in order:
        group = groups[key]
        rep = group[0]
        if len(group) > 1 and rep.quantity is None:
            rep = rep.model_copy(update={"quantity": len(group)})
        collapsed.append(rep)
    return collapsed


def normalize(data: dict[str, Any]) -> tuple[ParseResult, list[str]]:
    """Return (result, issues) for a raw result dict. No issues means a clean parse."""
    issues: list[str] = []
    entities: list[Entity] = []
    for raw in data.get("entities", []):
        entity, ent_issues = normalize_entity(raw)
        issues.extend(ent_issues)
        if entity is not None:
            entities.append(entity)
    entities = collapse_quantities(entities)
    return ParseResult(entities=entities), issues
