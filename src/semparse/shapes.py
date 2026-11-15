"""Build SHACL shapes from the ontology vocabulary and CLASS_AXIOMS.

The vocabulary checks (valid media, purposes, units, one value per measure)
rarely fail after constrained decoding. The per-class rules in CLASS_AXIOMS are
the ones that catch small-model mistakes.
"""

from __future__ import annotations

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import RDF, XSD

from .ontology import (
    ASSET_CLASSES,
    CANONICAL_UNITS,
    MEASURE_FIELDS,
    MEDIA,
    PURPOSES,
    class_axioms,
)
from .rdf import NS

SH = Namespace("http://www.w3.org/ns/shacl#")


def _rdf_list(g: Graph, items: list) -> BNode:
    head = BNode()
    Collection(g, head, list(items))
    return head


def _measure_shape(g: Graph) -> URIRef:
    """Shared shape for measures: a decimal value and a canonical unit."""
    shape = NS.MeasureShape
    g.add((shape, RDF.type, SH.NodeShape))

    value_prop = BNode()
    g.add((shape, SH.property, value_prop))
    g.add((value_prop, SH.path, NS.value))
    g.add((value_prop, SH.datatype, XSD.decimal))
    g.add((value_prop, SH.minCount, Literal(1)))
    g.add((value_prop, SH.maxCount, Literal(1)))

    unit_prop = BNode()
    g.add((shape, SH.property, unit_prop))
    g.add((unit_prop, SH.path, NS.unit))
    g.add((unit_prop, SH.minCount, Literal(1)))
    g.add((unit_prop, SH.maxCount, Literal(1)))
    g.add((unit_prop, SH["in"], _rdf_list(g, [Literal(u) for u in CANONICAL_UNITS])))
    return shape


def _property(g: Graph, shape: URIRef, path: URIRef) -> BNode:
    prop = BNode()
    g.add((shape, SH.property, prop))
    g.add((prop, SH.path, path))
    return prop


def build_shapes() -> Graph:
    """Build the full SHACL shapes graph for the lifted ontology."""
    g = Graph()
    g.bind("semparse", NS)
    g.bind("sh", SH)

    measure_shape = _measure_shape(g)

    for cls in ASSET_CLASSES:
        axioms = class_axioms(cls)
        shape = NS[f"{cls}Shape"]
        g.add((shape, RDF.type, SH.NodeShape))
        g.add((shape, SH.targetClass, NS[cls]))

        # Measures: fields the class may not have get maxCount 0, the rest maxCount 1.
        allowed_measures = axioms.get("measures")
        for field in MEASURE_FIELDS:
            prop = _property(g, shape, NS[field])
            if allowed_measures is not None and field not in allowed_measures:
                g.add((prop, SH.maxCount, Literal(0)))
                allowed_txt = ", ".join(allowed_measures) or "no measure"
                g.add(
                    (
                        prop,
                        SH.message,
                        Literal(
                            f"a {cls} must not carry a {field} "
                            f"(expected: {allowed_txt})"
                        ),
                    )
                )
            else:
                g.add((prop, SH.maxCount, Literal(1)))
                g.add((prop, SH.node, measure_shape))

        # Source / sink: at most one, from MEDIA or the class's own list.
        for role, path in (("sources", NS.source), ("sinks", NS.sink)):
            allowed = axioms.get(role)
            prop = _property(g, shape, path)
            role_word = role[:-1]
            if allowed == ():
                g.add((prop, SH.maxCount, Literal(0)))
                g.add((prop, SH.message, Literal(f"a {cls} has no {role_word} medium")))
                continue
            g.add((prop, SH.maxCount, Literal(1)))
            members = allowed if allowed is not None else MEDIA
            g.add((prop, SH["in"], _rdf_list(g, [NS[m] for m in members])))
            if allowed is not None:
                g.add(
                    (
                        prop,
                        SH.message,
                        Literal(
                            f"a {cls} {role_word} must be one of: "
                            + ", ".join(members)
                        ),
                    )
                )

        # Purpose: purpose and purposes share one predicate, so no maxCount.
        allowed_purposes = axioms.get("purposes")
        prop = _property(g, shape, NS.purpose)
        if allowed_purposes == ():
            g.add((prop, SH.maxCount, Literal(0)))
            g.add((prop, SH.message, Literal(f"a {cls} carries no purpose")))
        else:
            members = allowed_purposes if allowed_purposes is not None else PURPOSES
            g.add((prop, SH["in"], _rdf_list(g, [NS[p] for p in members])))
            if allowed_purposes is not None:
                g.add(
                    (
                        prop,
                        SH.message,
                        Literal(
                            f"a {cls} purpose must be one of: " + ", ".join(members)
                        ),
                    )
                )

    return g
