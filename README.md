# Semantic Parsing using SLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

The goal of this repo is to provide a practical solution to semantic parsing
tasks. An improved harness around a small language model can provide sufficiently
accurate structured output for a specific domain. This repo explores that approach
using spreadsheet descriptions of energy systems and buildings.

For example, `air source heat pump 8kWth + 200L DHW cylinder` becomes:

```json
[
  {"class": "HeatPump", "source": "Air", "purpose": "SpaceHeating",
   "power": {"value": 8, "unit": "kWth"}},
  {"class": "ThermalStorageTank", "purpose": "DHW",
   "volume": {"value": 200, "unit": "L"}}
]
```

The model identifies assets and their properties. Constrained decoding limits
output to the schema and ontology vocabulary. Python code parses measurements,
normalizes units and merges duplicate entities. Column profiles use headers to
narrow the available classes and supply context.

## Usage

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).
The default model, `Qwen2.5-1.5B-Instruct`, downloads on first use.

```bash
uv sync
uv run semparse --xlsx sample.xlsx --column Description \
    --out out.jsonl --annotate annotated.xlsx
```

Output is JSONL; `--annotate` also saves a copy of the workbook with the parsed
results. Use `--profile dhw_technology` to select a profile explicitly, or
`--model <hf-id>` to change the model.

From Python:

```python
from semparse.infer import parse_cell

result = parse_cell("water source heat pump 20 kWth",
                    header="Space heating technology")
print(result.model_dump_json(exclude_none=True, by_alias=True))
```

## Validation and repair

Entities can be exported to RDF with `semparse.rdf.to_graph` and checked with
`semparse.validate.validate_result`. SHACL checks catch invalid combinations,
such as a chiller assigned a heating purpose.

Pass `max_repairs=3` to `parse_cell` to retry using validation feedback. A repair
is accepted only if it reduces violations without changing the entity count.

## Evaluation

```bash
uv run pytest -q                 # unit tests; no model needed
uv run python eval/score.py      # score the default evaluation set
uv run python eval/ladder.py     # compare normalization, reasoning and repair
```

Recorded entity F1 scores with seed 0:

| Model | Default set (106 cells) | Different wording (52 cells) |
|-------|-------------------------|-----------------------------|
| Base | 0.744 | 0.500 |
| Base + repair | 0.800 | 0.453 |
| LoRA adapter | 0.963 | 0.766 |
| LoRA adapter + repair | 0.978 | 0.797 |

These are small evaluation sets. The default set uses wording similar to the
training data; the second was written separately with different phrasing.
Reasoning before parsing reduced accuracy in the recorded runs. Full results,
per-cell counts and bootstrap intervals are in [eval/](eval/).

To enable the optional evaluation gate before each push:

```bash
git config core.hooksPath .githooks
```

It compares scores against `eval/baseline.json`. `SEMPARSE_EVAL_TOL` sets the
allowed drop (default 0); `git push --no-verify` skips the check.

## Fine-tuning

Train a LoRA adapter on the examples in `data/train.jsonl`, then evaluate it:

```bash
uv sync --extra train
uv run python finetune/train_lora.py --dry-run
uv run python finetune/train_lora.py --epochs 3 --out finetune/out
SEMPARSE_ADAPTER=finetune/out uv run python eval/score.py
```

Training uses raw measurement spans and the same system prompt as inference.
Constrained decoding remains enabled when using the adapter.

## Customization

Edit [ontology.py](src/semparse/ontology.py) to change the vocabulary, unit
aliases and class rules. The schema and SHACL shapes are generated from these
definitions. Review the prompts and labeled examples when changing domains.

Column profiles live in [profile_defs/](src/semparse/profile_defs/), or a folder
set through `SEMPARSE_PROFILES_DIR`. Add a YAML or JSON file to define allowed
classes, header keywords, purpose hints and examples. Changes are picked up
without restarting.

## License

[MIT](LICENSE).
