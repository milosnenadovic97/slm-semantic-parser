"""Load the model and generate schema-constrained JSON (outlines + transformers).

Default model is Qwen/Qwen2.5-1.5B-Instruct; override with SEMPARSE_MODEL or
parse_cell(model=...). torch/transformers/outlines are imported lazily.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Optional

from .expand import expand
from .normalize import normalize
from .profiles import GENERAL, Profile, select_profile
from .prompt import build_messages, build_reasoning_messages, build_repair_messages
from .schema import ParseResult, restricted_raw_result
from .validate import validate_result


def _apply_purpose_hint(result: ParseResult, profile: Profile) -> ParseResult:
    """Set the profile's purpose hint on entities with no purpose(s)."""
    if not profile.purpose_hint:
        return result
    patched = [
        e.model_copy(update={"purpose": profile.purpose_hint})
        if e.purpose is None and not e.purposes
        else e
        for e in result.entities
    ]
    return result.model_copy(update={"entities": patched})


DEFAULT_MODEL = os.environ.get("SEMPARSE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
# Optional LoRA adapter dir (e.g. finetune/out/checkpoint-96), loaded on top of the base.
DEFAULT_ADAPTER: Optional[str] = os.environ.get("SEMPARSE_ADAPTER") or None


@lru_cache(maxsize=2)
def _load_model(model_id: str, adapter: Optional[str] = None) -> tuple[Any, Any]:
    """Load and cache (outlines model, tokenizer), with an optional LoRA adapter."""
    # outlines may try torch.compile, which fails noisily without a C compiler.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import outlines
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda" if use_cuda else "cpu",
    )
    if adapter:
        from peft import PeftModel

        hf_model = PeftModel.from_pretrained(hf_model, adapter)
        hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return outlines.from_transformers(hf_model, tokenizer), tokenizer


@lru_cache(maxsize=16)
def _generator(
    model_id: str, allowed_classes: tuple[str, ...], adapter: Optional[str] = None
) -> Any:
    """Build and cache a constrained generator for one allowed-class set."""
    import outlines

    model, _ = _load_model(model_id, adapter)
    schema = restricted_raw_result(allowed_classes)
    return outlines.Generator(model, schema)


def _reason(
    model_id: str,
    cell: str,
    profile: Profile,
    header: Optional[str],
    max_new_tokens: int,
    adapter: Optional[str] = None,
) -> str:
    """Free-text reasoning pass; its output is added to the JSON prompt."""
    model_obj, tokenizer = _load_model(model_id, adapter)
    prompt = tokenizer.apply_chat_template(
        build_reasoning_messages(cell, profile, header),
        tokenize=False,
        add_generation_prompt=True,
    )
    text = model_obj(prompt, max_new_tokens=max_new_tokens)
    return text if isinstance(text, str) else str(text)


def _decode(generator: Any, tokenizer: Any, messages: list, max_new_tokens: int):
    """Apply the chat template and run one constrained generation."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generator(prompt, max_new_tokens=max_new_tokens)


def _normalize_json(raw_json: str, profile: Profile) -> tuple[ParseResult, list[str]]:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # Shouldn't happen with constrained decoding.
        return ParseResult(), [f"non-JSON model output: {raw_json[:200]!r}"]
    result, issues = normalize(data)
    return _apply_purpose_hint(result, profile), issues


def parse_cell(
    cell: str,
    *,
    model: str = DEFAULT_MODEL,
    adapter: Optional[str] = DEFAULT_ADAPTER,
    profile: Optional[Profile] = None,
    header: Optional[str] = None,
    max_new_tokens: int = 512,
    reason: bool = False,
    max_repairs: int = 0,
    return_issues: bool = False,
) -> ParseResult | tuple[ParseResult, list[str]]:
    """Parse one cell into a normalized ParseResult.

    profile (or a header to pick one) limits the allowed classes.
    reason=True runs a free-text reasoning pass before decoding.
    max_repairs=k re-decodes up to k times while SHACL validation fails.
    return_issues=True also returns the issues list; unresolved SHACL
    violations appear as "SHACL: ..." entries.
    """
    if profile is None:
        profile = select_profile(header)

    if not cell or not cell.strip():
        result = ParseResult()
        return (result, []) if return_issues else result

    cell = expand(cell)
    _, tokenizer = _load_model(model, adapter)
    generator = _generator(model, tuple(profile.allowed_classes), adapter)

    reasoning = (
        _reason(model, cell, profile, header, max_new_tokens, adapter)
        if reason else None
    )

    raw_json = _decode(
        generator, tokenizer, build_messages(cell, profile, header, reasoning),
        max_new_tokens,
    )
    result, issues = _normalize_json(raw_json, profile)

    if max_repairs > 0:
        conforms, violations = validate_result(result)
        attempt = 0
        while not conforms and attempt < max_repairs:
            attempt += 1
            cand_json = _decode(
                generator,
                tokenizer,
                build_repair_messages(
                    cell, raw_json, violations, profile, header, reasoning
                ),
                max_new_tokens,
            )
            cand_result, cand_issues = _normalize_json(cand_json, profile)
            cand_conforms, cand_violations = validate_result(cand_result)
            # Accept only if violations drop and the entity count is unchanged;
            # a re-decode can otherwise break entities that were already right.
            if len(cand_violations) < len(violations) and len(
                cand_result.entities
            ) == len(result.entities):
                raw_json = cand_json
                result, issues = cand_result, cand_issues
                conforms, violations = cand_conforms, cand_violations
            else:
                break
        if not conforms:
            issues = issues + [f"SHACL: {v}" for v in violations]

    return (result, issues) if return_issues else result
