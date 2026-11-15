"""Convert a ParseResult into an RDF graph. Each entity becomes a blank node:

    _:e0 a semparse:HeatPump ;
        semparse:source   semparse:TreatedWastewater ;
        semparse:sink     semparse:Water ;
        semparse:purpose  semparse:SpaceHeating ;
        semparse:controllable true ;
        semparse:power    [ semparse:value 22.0 ; semparse:unit "kWth" ] .
"""

from __future__ import annotations

from typing import Iterable

from decimal import Decimal

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from .ontology import MEASURE_FIELDS
from .schema import Entity, Measure, ParseResult

# Full IRIs are only created here; the model works with bare labels.
SEMPARSE_IRI = "https://w3id.org/semparse#"
NS = Namespace(SEMPARSE_IRI)

_ROLE_PREDICATES = {"source": NS.source, "sink": NS.sink, "purpose": NS.purpose}


def _measure_node(g: Graph, measure: Measure) -> BNode:
    node = BNode()
    # Decimal gives an xsd:decimal; a plain-string unit matches the shapes' sh:in.
    g.add((node, NS.value, Literal(Decimal(str(measure.value)))))
    g.add((node, NS.unit, Literal(measure.unit)))
    return node


def entity_to_triples(g: Graph, entity: Entity, subject: URIRef | BNode) -> None:
    """Add one entity's triples to g under subject."""
    g.add((subject, RDF.type, NS[entity.cls]))

    for role, pred in _ROLE_PREDICATES.items():
        val = getattr(entity, role)
        if val is not None:
            g.add((subject, pred, NS[val]))

    for purpose in entity.purposes or ():
        g.add((subject, NS.purpose, NS[purpose]))

    for field in MEASURE_FIELDS:
        measure = getattr(entity, field)
        if measure is not None:
            g.add((subject, NS[field], _measure_node(g, measure)))

    if entity.quantity is not None:
        g.add((subject, NS.quantity, Literal(entity.quantity, datatype=XSD.integer)))
    if entity.controllable is not None:
        g.add(
            (subject, NS.controllable, Literal(entity.controllable, datatype=XSD.boolean))
        )
    for key, value in (entity.attributes or {}).items():
        g.add((subject, NS[f"attr_{key}"], Literal(str(value))))


def to_graph(result: ParseResult | Iterable[Entity]) -> Graph:
    """Build a graph from a ParseResult or a list of entities."""
    entities = result.entities if isinstance(result, ParseResult) else list(result)
    g = Graph()
    g.bind("semparse", NS)
    for entity in entities:
        entity_to_triples(g, entity, BNode())
    return g
