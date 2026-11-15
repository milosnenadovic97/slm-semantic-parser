"""Pydantic models built from the ontology vocabulary.

RawEntity / RawParseResult: what the model emits under constrained decoding,
with the size kept as verbatim text in measure_raw.
Entity / ParseResult: the normalized output, with the size parsed into
power, volume or area.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from .ontology import ASSET_CLASSES, CANONICAL_UNITS, MEDIA, PURPOSES

# Literal[tuple] expands to the tuple's values, so these follow ontology.py.
ClassName = Literal[ASSET_CLASSES]  # type: ignore[valid-type]
Medium = Literal[MEDIA]  # type: ignore[valid-type]
Unit = Literal[CANONICAL_UNITS]  # type: ignore[valid-type]
Purpose = Literal[PURPOSES]  # type: ignore[valid-type]


# Model-facing (constrained decoding target)
class RawEntity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cls: ClassName = Field(alias="class")
    source: Optional[Medium] = None
    sink: Optional[Medium] = None
    purpose: Optional[Purpose] = None
    measure_raw: Optional[str] = Field(
        default=None,
        description='Verbatim size span from the text, e.g. "22 kWth", "1500L", '
        '"4 m2". Copy it exactly; do not compute, convert, or label its kind.',
    )
    purposes: Optional[list[Purpose]] = None
    quantity: Optional[int] = None
    controllable: Optional[bool] = None
    attributes: dict[str, str] = Field(default_factory=dict)


class RawParseResult(BaseModel):
    entities: list[RawEntity] = Field(default_factory=list)


def restricted_raw_result(allowed_classes: tuple[str, ...]) -> type[BaseModel]:
    """RawParseResult with the class enum limited to allowed_classes."""
    if set(allowed_classes) == set(ASSET_CLASSES):
        return RawParseResult
    cls_literal = Literal[tuple(allowed_classes)]  # type: ignore[valid-type]
    entity = create_model(
        "RawEntityRestricted",
        __base__=RawEntity,
        cls=(cls_literal, Field(alias="class")),
    )
    return create_model(
        "RawParseResultRestricted",
        entities=(list[entity], Field(default_factory=list)),  # type: ignore[valid-type]
    )


# Normalized output
class Measure(BaseModel):
    """A value and unit parsed from measure_raw."""

    value: float
    unit: Unit


class Entity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cls: ClassName = Field(alias="class")
    source: Optional[Medium] = None
    sink: Optional[Medium] = None
    purpose: Optional[Purpose] = None
    power: Optional[Measure] = None
    volume: Optional[Measure] = None
    area: Optional[Measure] = None
    purposes: Optional[list[Purpose]] = None
    quantity: Optional[int] = None
    controllable: Optional[bool] = None
    attributes: dict[str, str] = Field(default_factory=dict)


class ParseResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
