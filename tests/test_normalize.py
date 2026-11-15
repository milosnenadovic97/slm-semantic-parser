"""Unit tests for the deterministic normalizer (no model required)."""

from __future__ import annotations

import pytest

from semparse.normalize import normalize, normalize_entity, parse_measure


@pytest.mark.parametrize(
    "raw, field, value, unit",
    [
        ("22 kWth", "power", 22.0, "kWth"),
        ("3kWth", "power", 3.0, "kWth"),
        ("22 kW(th)", "power", 22.0, "kWth"),
        ("1.5 MWe", "power", 1.5, "MWe"),
        ("250 kWp", "power", 250.0, "kWp"),
        ("rated at 11 kW", "power", 11.0, "kW"),
        ("500 kWh", "power", 500.0, "kWh"),
        ("300 litres", "volume", 300.0, "L"),
        ("1500L", "volume", 1500.0, "L"),
        ("1.2 m3", "volume", 1.2, "m3"),
        ("4 m2", "area", 4.0, "m2"),
    ],
)
def test_parse_measure_ok_and_routing(raw, field, value, unit):
    got_field, measure, issue = parse_measure(raw)
    assert issue is None
    assert got_field == field
    assert measure is not None
    assert measure.value == value
    assert measure.unit == unit


def test_parse_measure_empty_is_not_an_issue():
    assert parse_measure(None) == (None, None, None)
    assert parse_measure("   ") == (None, None, None)


def test_parse_measure_unparseable_flags():
    field, measure, issue = parse_measure("big tank")
    assert field is None and measure is None
    assert issue and "unparseable" in issue


def test_normalize_entity_routes_power():
    entity, issues = normalize_entity(
        {
            "class": "HeatPump",
            "source": "TreatedWastewater",
            "sink": "Water",
            "measure_raw": "22 kWth",
            "controllable": True,
        }
    )
    assert issues == []
    assert entity is not None
    assert entity.power and entity.power.value == 22.0 and entity.power.unit == "kWth"
    assert entity.volume is None and entity.area is None


def test_normalize_entity_routes_volume_with_purpose():
    entity, issues = normalize_entity(
        {"class": "ThermalStorageTank", "measure_raw": "1500L", "purpose": "SpaceHeating"}
    )
    assert issues == []
    assert entity is not None
    assert entity.volume and entity.volume.value == 1500.0 and entity.volume.unit == "L"
    assert entity.power is None
    assert entity.purpose == "SpaceHeating"


def test_normalize_entity_routes_area():
    entity, _ = normalize_entity({"class": "SolarThermalPanel", "measure_raw": "4 m2"})
    assert entity is not None
    assert entity.area and entity.area.value == 4.0 and entity.area.unit == "m2"


def test_unknown_class_is_dropped_and_flagged():
    entity, issues = normalize_entity({"class": "Spaceship"})
    assert entity is None
    assert any("unknown class" in i for i in issues)


def test_unknown_medium_and_purpose_cleared_and_flagged():
    entity, issues = normalize_entity(
        {"class": "HeatPump", "source": "Lava", "purpose": "Teleportation"}
    )
    assert entity is not None
    assert entity.source is None and entity.purpose is None
    assert any("unknown source" in i for i in issues)
    assert any("unknown purpose" in i for i in issues)


def test_attribute_values_coerced_to_str():
    entity, _ = normalize_entity(
        {"class": "Dwelling", "quantity": 4, "attributes": {"bedrooms": 1}}
    )
    assert entity is not None
    assert entity.quantity == 4
    assert entity.attributes == {"bedrooms": "1"}


def test_normalize_heatpump_example():
    """The compound heat-pump example from the project brief."""
    raw = {
        "entities": [
            {
                "class": "HeatPump",
                "source": "TreatedWastewater",
                "sink": "Water",
                "measure_raw": "22 kWth",
                "controllable": True,
            },
            {"class": "ImmersionHeater", "measure_raw": "3 kWth", "controllable": False},
        ]
    }
    result, issues = normalize(raw)
    assert issues == []
    assert len(result.entities) == 2
    hp, immersion = result.entities
    assert hp.cls == "HeatPump"
    assert hp.source == "TreatedWastewater"
    assert hp.power.unit == "kWth" and hp.power.value == 22.0
    assert immersion.cls == "ImmersionHeater"
    assert immersion.controllable is False


def test_normalize_thermal_storage_two_tanks():
    """1500L SH + 1000L DHW -> two tanks with distinct purposes."""
    raw = {
        "entities": [
            {"class": "ThermalStorageTank", "measure_raw": "1500L", "purpose": "SpaceHeating"},
            {"class": "ThermalStorageTank", "measure_raw": "1000L", "purpose": "DHW"},
        ]
    }
    result, issues = normalize(raw)
    assert issues == []
    assert [e.purpose for e in result.entities] == ["SpaceHeating", "DHW"]
    assert [e.volume.value for e in result.entities] == [1500.0, 1000.0]


def test_collapse_bare_duplicates_become_quantity():
    """Two identical entities with no quantity collapse into one with quantity=N."""
    raw = {
        "entities": [
            {"class": "Battery", "measure_raw": "5 kWh"},
            {"class": "Battery", "measure_raw": "5 kWh"},
        ]
    }
    result, issues = normalize(raw)
    assert issues == []
    assert len(result.entities) == 1
    battery = result.entities[0]
    assert battery.cls == "Battery"
    assert battery.power.value == 5.0 and battery.power.unit == "kWh"
    assert battery.quantity == 2


def test_collapse_duplicates_with_quantity_are_deduped_not_multiplied():
    """Re-emitted identical entities that already carry a quantity keep it."""
    raw = {
        "entities": [
            {"class": "Flat", "quantity": 4},
            {"class": "Flat", "quantity": 4},
        ]
    }
    result, issues = normalize(raw)
    assert issues == []
    assert len(result.entities) == 1
    assert result.entities[0].quantity == 4


def test_collapse_leaves_distinct_entities_untouched():
    """Entities differing on any field are never merged."""
    raw = {
        "entities": [
            {"class": "Battery", "measure_raw": "5 kWh"},
            {"class": "Battery", "measure_raw": "10 kWh"},
        ]
    }
    result, issues = normalize(raw)
    assert issues == []
    assert len(result.entities) == 2
    assert [e.power.value for e in result.entities] == [5.0, 10.0]
    assert all(e.quantity is None for e in result.entities)
