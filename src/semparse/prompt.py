"""Build the prompts: rules, column context from the profile, and few-shot examples.

The examples teach what constrained decoding can't: one entity per asset, and
copying the size verbatim into measure_raw.
"""

from __future__ import annotations

import json
from typing import Optional

from .profiles import GENERAL, Profile

_BASE_RULES = """\
You convert a single free-text spreadsheet cell into a list of structured \
ontology entities. Output ONLY JSON of the form {"entities": [ ... ]}.

Rules:
- Each entity's "class" MUST be one of the allowed classes listed below.
- "source"/"sink" (optional) describe the medium an asset draws from / delivers \
to (e.g. Air, Ground, Water, TreatedWastewater, Gas, Solar, Grid).
- "purpose" (optional) is the service when the asset serves exactly one: SpaceHeating, DHW, or Cooling.
- "purposes" (optional list) replaces "purpose" when the asset explicitly serves multiple services (e.g. "both SH and DHW").
- Split compound descriptions into ONE entity per distinct asset.
- "measure_raw" is the VERBATIM size span (e.g. "22 kWth", "1500L", "4 m2"). \
Copy it exactly; never compute, sum, convert, or decide whether it is power/\
volume/area.
- "controllable" is true only when the text marks the asset as controlled (e.g. \
"(control)").
- Use "quantity" for counts (e.g. "x4") and "attributes" for free descriptors \
(e.g. bedrooms, form) with no dedicated field.
- Omit fields you cannot determine."""


def build_system(profile: Profile = GENERAL, header: Optional[str] = None) -> str:
    """Build the system prompt, injecting the column profile's constraints."""
    lines = [_BASE_RULES, ""]
    if header:
        lines.append(f'Column header: "{header}"')
    if profile.description:
        lines.append(f"Column context: {profile.description}")
    if profile.purpose_hint:
        lines.append(
            f'Unless the text says otherwise, "purpose" is "{profile.purpose_hint}".'
        )
    lines.append("Allowed classes: " + ", ".join(profile.allowed_classes))
    return "\n".join(lines)


# Used when a profile has no examples of its own.
FEW_SHOT: list[tuple[str, dict]] = [
    (
        "air source heat pump 8kWth + 200L DHW cylinder",
        {
            "entities": [
                {
                    "class": "HeatPump",
                    "source": "Air",
                    "purpose": "SpaceHeating",
                    "measure_raw": "8kWth",
                },
                {
                    "class": "ThermalStorageTank",
                    "purpose": "DHW",
                    "measure_raw": "200L",
                },
            ]
        },
    ),
    (
        "1500L SH, 1000L DHW",
        {
            "entities": [
                {
                    "class": "ThermalStorageTank",
                    "purpose": "SpaceHeating",
                    "measure_raw": "1500L",
                },
                {
                    "class": "ThermalStorageTank",
                    "purpose": "DHW",
                    "measure_raw": "1000L",
                },
            ]
        },
    ),
    (
        "solar thermal 4 m2",
        {"entities": [{"class": "SolarThermalPanel", "measure_raw": "4 m2"}]},
    ),
    (
        "500 kWh battery storage",
        {"entities": [{"class": "Battery", "measure_raw": "500 kWh"}]},
    ),
    (
        "2 bedroom flat x3",
        {"entities": [{"class": "Flat", "quantity": 3, "attributes": {"bedrooms": "2"}}]},
    ),
]


def _final_user_content(cell: str, reasoning: Optional[str]) -> str:
    """The cell, plus the reasoning pass output if there is one."""
    if not reasoning:
        return cell
    return f"{cell}\n\nAnalysis (use this, then output the JSON):\n{reasoning.strip()}"


def build_messages(
    cell: str,
    profile: Profile = GENERAL,
    header: Optional[str] = None,
    reasoning: Optional[str] = None,
) -> list[dict[str, str]]:
    """System prompt, few-shot turns (the profile's or FEW_SHOT), then the cell."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system(profile, header)}
    ]
    for text, output in (profile.examples or FEW_SHOT):
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps(output)})
    messages.append({"role": "user", "content": _final_user_content(cell, reasoning)})
    return messages


_REASONING_SYSTEM = """\
You analyse a single free-text spreadsheet cell before it is converted to \
structured data. Think briefly and concretely, in 1-3 short sentences — do NOT \
output JSON. State:
- how many distinct assets the cell describes (watch for counts like "2x" and \
compound descriptions joined by "+", "/", or commas),
- for each, its likely ontology class and any source/sink medium, purpose, or \
control marker.
Keep it short; this is a scratchpad, not the answer."""


def build_reasoning_messages(
    cell: str, profile: Profile = GENERAL, header: Optional[str] = None
) -> list[dict[str, str]]:
    """Messages for the free-text reasoning pass (no few-shot)."""
    system = _REASONING_SYSTEM + "\n\nAllowed classes: " + ", ".join(
        profile.allowed_classes
    )
    if header:
        system += f'\nColumn header: "{header}"'
    if profile.purpose_hint:
        system += f'\nDefault purpose unless stated: "{profile.purpose_hint}".'
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": cell},
    ]


def build_repair_messages(
    cell: str,
    prev_json: str,
    violations: list[str],
    profile: Profile = GENERAL,
    header: Optional[str] = None,
    reasoning: Optional[str] = None,
) -> list[dict[str, str]]:
    """Show the model its last output and the SHACL violations, and ask for a fix."""
    messages = build_messages(cell, profile, header, reasoning)
    messages.append({"role": "assistant", "content": prev_json})
    bullet = "\n".join(f"- {v}" for v in violations)
    messages.append(
        {
            "role": "user",
            "content": (
                "That output violates the ontology's constraints:\n"
                f"{bullet}\n\n"
                "Fix ONLY what the violations call out (keep everything else the "
                "same) and output the corrected JSON."
            ),
        }
    )
    return messages
