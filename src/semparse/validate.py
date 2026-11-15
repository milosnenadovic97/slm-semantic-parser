"""Check a graph against the SHACL shapes with pyshacl.

Violations come back as short strings so they can be fed to the model as repair
feedback.
"""

from __future__ import annotations

from functools import lru_cache

from rdflib import Graph
from rdflib.namespace import RDF

from .rdf import NS, to_graph
from .schema import ParseResult
from .shapes import SH, build_shapes


@lru_cache(maxsize=1)
def _cached_shapes() -> Graph:
    return build_shapes()


def _short(term) -> str:
    """semparse:HeatPump -> HeatPump."""
    text = str(term)
    if text.startswith(str(NS)):
        return text[len(str(NS)):]
    return text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _render_violations(report: Graph) -> list[str]:
    """Turn a SHACL report graph into readable messages."""
    messages: list[str] = []
    for result in report.subjects(RDF.type, SH.ValidationResult):
        focus = report.value(result, SH.focusNode)
        path = report.value(result, SH.resultPath)
        message = report.value(result, SH.resultMessage)
        cls = None
        if focus is not None:
            cls = report.value(focus, RDF.type)
        parts = []
        if cls is not None:
            parts.append(_short(cls))
        if path is not None:
            parts.append(f"on '{_short(path)}'")
        prefix = " ".join(parts)
        detail = str(message) if message is not None else "constraint violated"
        messages.append(f"{prefix}: {detail}".strip(": ").strip())
    return sorted(set(messages))


def validate(
    graph: Graph, shapes: Graph | None = None
) -> tuple[bool, list[str]]:
    """Return (conforms, messages), with one message per violation."""
    from pyshacl import validate as pyshacl_validate

    conforms, report_graph, _ = pyshacl_validate(
        data_graph=graph,
        shacl_graph=shapes if shapes is not None else _cached_shapes(),
        inference="none",
        advanced=True,
    )
    if conforms:
        return True, []
    return False, _render_violations(report_graph)


def validate_result(result: ParseResult) -> tuple[bool, list[str]]:
    """Convert a ParseResult to RDF and validate it."""
    return validate(to_graph(result))
