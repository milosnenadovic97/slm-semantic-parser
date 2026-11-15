"""Ontology vocabulary. The schema, normalizer and SHACL shapes are built from it.

Terms are bare labels like HeatPump because a small model handles them better
than full IRIs. qualify() can add a namespace later.
"""

from __future__ import annotations

# Prefix used by qualify(). Empty until the real namespace is decided.
NAMESPACE: str = ""

# Asset classes (the model can only emit these)
ASSET_CLASSES: tuple[str, ...] = (
    "HeatPump",
    "ImmersionHeater",
    "Boiler",
    "SolarThermalPanel",
    "PVArray",
    "Battery",
    "ThermalStorageTank",
    "HeatExchanger",
    "CHP",
    "Chiller",
    "DistrictHeatingConnection",
    "Dwelling",
    "Flat",
    "Hub",
)

# Purposes (the service an asset provides)
PURPOSES: tuple[str, ...] = (
    "SpaceHeating",
    "DHW",
    "Cooling",
)

# Media (what an asset draws from / delivers to)
MEDIA: tuple[str, ...] = (
    "Water",
    "Air",
    "Ground",
    "TreatedWastewater",
    "Refrigerant",
    "Gas",
    "Grid",
    "Solar",
)

# Units
# Lower-case raw spelling -> canonical unit. Used by the normalizer, not the model.
UNIT_ALIASES: dict[str, str] = {
    "kwth": "kWth",
    "kw(th)": "kWth",
    "kw th": "kWth",
    "kwt": "kWth",
    "kwe": "kWe",
    "kw(e)": "kWe",
    "kwp": "kWp",
    "kw(p)": "kWp",
    "kw": "kW",
    "mwth": "MWth",
    "mwe": "MWe",
    "mw": "MW",
    "kwh": "kWh",
    "mwh": "MWh",
    "l": "L",
    "litre": "L",
    "litres": "L",
    "liter": "L",
    "liters": "L",
    "m3": "m3",
    "m³": "m3",
    "cubic metre": "m3",
    "cubic meter": "m3",
    "m2": "m2",
    "m²": "m2",
    "sqm": "m2",
    "sq m": "m2",
}

CANONICAL_UNITS: tuple[str, ...] = tuple(sorted(set(UNIT_ALIASES.values())))

# Which Entity field each unit goes in. The model never picks the field.
MEASURE_KINDS: dict[str, str] = {
    "kW": "power",
    "kWth": "power",
    "kWe": "power",
    "kWp": "power",
    "MW": "power",
    "MWth": "power",
    "MWe": "power",
    "kWh": "power",  # battery capacity; the gold labels treat it as power
    "MWh": "power",
    "L": "volume",
    "m3": "volume",
    "m2": "area",
}


# Per-class rules, compiled into SHACL shapes in shapes.py
# Catches terms that are each valid but wrong together, e.g. an ImmersionHeater
# with a source or a HeatPump with a volume.
#
# Each key maps to a tuple of allowed values, () to forbid the property, or is
# left out for no constraint. "measures" lists the measure fields a class may have.
CLASS_AXIOMS: dict[str, dict[str, tuple[str, ...] | None]] = {
    "HeatPump": {
        "measures": ("power",),
        "sources": ("Air", "Ground", "Water", "TreatedWastewater"),
        "purposes": ("SpaceHeating", "DHW", "Cooling"),
    },
    "ImmersionHeater": {
        "measures": ("power",),
        "sources": (),  # electric resistance, no source medium
        "purposes": ("DHW", "SpaceHeating"),
    },
    "Boiler": {
        "measures": ("power",),
        "sources": ("Gas",),
        "purposes": ("SpaceHeating", "DHW"),
    },
    "SolarThermalPanel": {
        "measures": ("area", "power"),
        "sources": ("Solar",),
        "purposes": ("DHW", "SpaceHeating"),
    },
    "PVArray": {
        "measures": ("power",),
        "sources": ("Solar",),
        "purposes": (),  # makes electricity, not heating or cooling
    },
    "Battery": {
        "measures": ("power",),
        "purposes": (),
    },
    "ThermalStorageTank": {
        "measures": ("volume",),
        "sources": (),
        "purposes": ("SpaceHeating", "DHW"),
    },
    "HeatExchanger": {
        "measures": ("power",),
    },
    "CHP": {
        "measures": ("power",),
        "sources": ("Gas",),
        "purposes": ("SpaceHeating", "DHW"),
    },
    "Chiller": {
        "measures": ("power",),
        "purposes": ("Cooling",),
    },
    "DistrictHeatingConnection": {
        "measures": ("power",),
        "purposes": ("SpaceHeating", "DHW"),
    },
    # Dwellings aren't plant: no measures, media or purposes.
    "Dwelling": {"measures": (), "sources": (), "sinks": (), "purposes": ()},
    "Flat": {"measures": (), "sources": (), "sinks": (), "purposes": ()},
    "Hub": {"measures": (), "sources": (), "sinks": (), "purposes": ()},
}

MEASURE_FIELDS: tuple[str, ...] = ("power", "volume", "area")


def class_axioms(cls: str) -> dict[str, tuple[str, ...] | None]:
    """Return the relational axioms for a class (empty dict if unconstrained)."""
    return CLASS_AXIOMS.get(cls, {})


def measure_field(canonical_unit_value: str) -> str | None:
    """Return the Entity field ('power'|'volume'|'area') a unit routes to."""
    return MEASURE_KINDS.get(canonical_unit_value)


def is_class(term: str) -> bool:
    return term in ASSET_CLASSES


def is_medium(term: str) -> bool:
    return term in MEDIA


def canonical_unit(raw: str) -> str | None:
    """Map a raw unit spelling to its canonical form, or None if unknown."""
    return UNIT_ALIASES.get(raw.strip().lower())


def qualify(term: str, namespace: str = NAMESPACE) -> str:
    """Prefix a bare term with the namespace (no-op while it's empty)."""
    return f"{namespace}{term}" if namespace else term
