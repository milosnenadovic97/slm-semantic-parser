"""Deterministic tests for SHACL shape generation + validation (no model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semparse.ontology import ASSET_CLASSES
from semparse.rdf import NS
from semparse.schema import Entity, Measure, ParseResult
from semparse.shapes import SH, build_shapes
from semparse.validate import validate_result

LABELED = Path(__file__).resolve().parents[1] / "eval" / "labeled.jsonl"


def _one(entity: Entity):
    return validate_result(ParseResult(entities=[entity]))


# shape generation
def test_every_class_has_a_target_shape():
    shapes = build_shapes()
    targeted = {str(o) for o in shapes.objects(None, SH.targetClass)}
    for cls in ASSET_CLASSES:
        assert str(NS[cls]) in targeted


# conformant graphs
@pytest.mark.parametrize(
    "entity",
    [
        Entity(cls="HeatPump", source="Air", purpose="SpaceHeating",
               power=Measure(value=8, unit="kWth")),
        Entity(cls="ThermalStorageTank", purpose="DHW",
               volume=Measure(value=200, unit="L")),
        Entity(cls="Battery", power=Measure(value=5, unit="kWh")),
        Entity(cls="SolarThermalPanel", area=Measure(value=4, unit="m2")),
        Entity(cls="Chiller", purpose="Cooling", power=Measure(value=10, unit="kW")),
        Entity(cls="Flat", quantity=3, attributes={"bedrooms": "2"}),
    ],
)
def test_conformant_entities_pass(entity):
    conforms, messages = _one(entity)
    assert conforms, messages


# relational violations SHACL should catch
@pytest.mark.parametrize(
    "entity",
    [
        # wrong measure domain
        Entity(cls="HeatPump", volume=Measure(value=1500, unit="L")),
        Entity(cls="Flat", power=Measure(value=5, unit="kW")),
        # medium that a class cannot have
        Entity(cls="ImmersionHeater", source="Air", power=Measure(value=3, unit="kW")),
        Entity(cls="ThermalStorageTank", source="Gas",
               volume=Measure(value=100, unit="L")),
        # incoherent purpose
        Entity(cls="Chiller", purpose="DHW", power=Measure(value=5, unit="kW")),
        Entity(cls="PVArray", purpose="SpaceHeating", power=Measure(value=4, unit="kWp")),
    ],
)
def test_relational_violations_are_caught(entity):
    conforms, messages = _one(entity)
    assert not conforms
    assert messages


def test_violation_message_is_human_readable():
    _, messages = _one(Entity(cls="HeatPump", volume=Measure(value=1, unit="L")))
    assert any("HeatPump" in m and "volume" in m for m in messages)


# the gold labels must themselves conform
# Guards against the shapes drifting from the ontology the eval encodes: if the
# gold set violates a shape, the SHACL metric would measure shape bugs, not model
# errors. Every hand-labeled gold entity must lift to a conformant graph.
def test_all_gold_labels_conform():
    lines = [
        json.loads(line)
        for line in LABELED.read_text().splitlines()
        if line.strip()
    ]
    offenders = []
    for ex in lines:
        result = ParseResult(
            entities=[Entity.model_validate(e) for e in ex["entities"]]
        )
        conforms, messages = validate_result(result)
        if not conforms:
            offenders.append((ex["input"], messages))
    assert not offenders, "gold labels violate the shapes:\n" + "\n".join(
        f"  {inp!r}: {msgs}" for inp, msgs in offenders
    )
