"""Deterministic tests for the RDF projection (no model needed)."""

from __future__ import annotations

from rdflib.namespace import RDF

from semparse.rdf import NS, to_graph
from semparse.schema import Entity, Measure, ParseResult


def _graph(*entities: Entity):
    return to_graph(ParseResult(entities=list(entities)))


def test_class_becomes_rdf_type():
    g = _graph(Entity(cls="HeatPump"))
    assert (None, RDF.type, NS.HeatPump) in g


def test_roles_project_to_individuals():
    g = _graph(Entity(cls="HeatPump", source="TreatedWastewater", sink="Water"))
    subj = next(g.subjects(RDF.type, NS.HeatPump))
    assert (subj, NS.source, NS.TreatedWastewater) in g
    assert (subj, NS.sink, NS.Water) in g


def test_measure_is_value_unit_node():
    g = _graph(Entity(cls="HeatPump", power=Measure(value=22.0, unit="kWth")))
    subj = next(g.subjects(RDF.type, NS.HeatPump))
    measure = g.value(subj, NS.power)
    assert measure is not None
    assert str(g.value(measure, NS.unit)) == "kWth"
    assert float(g.value(measure, NS.value)) == 22.0


def test_purposes_list_projects_to_multiple_purpose_triples():
    g = _graph(Entity(cls="HeatPump", purposes=["SpaceHeating", "DHW"]))
    subj = next(g.subjects(RDF.type, NS.HeatPump))
    purposes = {str(o).rsplit("#", 1)[-1] for o in g.objects(subj, NS.purpose)}
    assert purposes == {"SpaceHeating", "DHW"}


def test_quantity_and_controllable_literals():
    g = _graph(Entity(cls="Battery", quantity=2, controllable=True))
    subj = next(g.subjects(RDF.type, NS.Battery))
    assert int(g.value(subj, NS.quantity)) == 2
    assert bool(g.value(subj, NS.controllable)) is True


def test_each_entity_gets_a_distinct_subject():
    g = _graph(Entity(cls="HeatPump"), Entity(cls="Battery"))
    subjects = set(g.subjects(RDF.type, None))
    assert len(subjects) == 2


def test_empty_result_is_empty_graph():
    assert len(to_graph(ParseResult())) == 0
