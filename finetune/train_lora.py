"""LoRA fine-tuning on prompt-to-raw-JSON examples.

Targets use verbatim ``measure_raw`` spans; normalization runs after inference.
Training reuses inference's system prompt and abbreviation expansion. Few-shot
examples are omitted unless ``--few-shot`` is set. The base model loads in fp16
(or bf16 with ``--bf16``), without quantization.

    uv run python finetune/train_lora.py --dry-run
    uv run python finetune/train_lora.py --epochs 3 --out finetune/out

Evaluate the saved adapter with ``SEMPARSE_ADAPTER`` set to its directory:

    SEMPARSE_ADAPTER=finetune/out uv run python eval/score.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semparse.expand import expand  # noqa: E402
from semparse.profiles import get_profile  # noqa: E402
from semparse.prompt import FEW_SHOT, build_system  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA = ROOT / "data" / "train.jsonl"


def to_messages(rec: dict, few_shot: bool) -> list[dict]:
    """Turn one training record into a chat message list ending in the target.

    Mirrors :func:`semparse.prompt.build_messages` (system + optional few-shot +
    user), then appends the assistant turn = the raw parse JSON we want to learn.
    """
    profile = get_profile(rec.get("profile") or "general")
    header = rec.get("header")
    cell = expand(rec["input"])  # same preprocessing as inference
    target = json.dumps({"entities": rec["entities"]})

    messages = [{"role": "system", "content": build_system(profile, header)}]
    if few_shot:
        for text, output in (profile.examples or FEW_SHOT):
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": json.dumps(output)})
    messages.append({"role": "user", "content": cell})
    messages.append({"role": "assistant", "content": target})
    return messages


def load_records(path: Path) -> list[dict]:
    recs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec.pop("_aspect", None)  # bookkeeping key, not part of the prompt
        recs.append(rec)
    return recs


def build_dataset(path: Path, few_shot: bool) -> list[dict]:
    return [{"messages": to_messages(r, few_shot)} for r in load_records(path)]


def dry_run(args) -> int:
    """Build the dataset and inspect it end-to-end using only transformers.

    Renders a sample through the model's chat template and reports the token
    length distribution — enough to pick ``--max-len`` and confirm the prompt
    matches inference, without a GPU or the training extras.
    """
    from transformers import AutoTokenizer

    data = build_dataset(Path(args.data), args.few_shot)
    tok = AutoTokenizer.from_pretrained(args.model)

    lengths = []
    for ex in data:
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    lengths.sort()
    n = len(lengths)
    p50 = lengths[n // 2]
    p95 = lengths[min(n - 1, int(n * 0.95))]
    print("=" * 66)
    print("SAMPLE (record 0) rendered prompt+target:\n")
    print(tok.apply_chat_template(data[0]["messages"], tokenize=False))
    print("=" * 66)
    print(f"records: {n}   few_shot={args.few_shot}")
    print(f"token length: min={lengths[0]} p50={p50} p95={p95} max={lengths[-1]}")
    # Base the cap on the MAX, not p95: SFT truncates from the right, so any cap
    # below the longest sequence would cut off the target JSON and teach the
    # model to emit truncated output.
    print(f"suggested --max-len: {min(2048, ((lengths[-1] // 64) + 1) * 64)}")
    if args.max_len < lengths[-1]:
        print(f"WARNING: --max-len {args.max_len} < max seq {lengths[-1]}: "
              f"targets would be truncated. Raise it.")
    return 0


def train(args) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)

    dataset = Dataset.from_list(build_dataset(Path(args.data), args.few_shot))

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_len,
        packing=False,
        # Train only on the assistant turn (the target JSON), masking the prompt.
        # Requires a chat template with generation tags; pass --no-mask to fall
        # back to full-sequence loss if your template lacks them.
        assistant_only_loss=not args.no_mask,
        bf16=use_bf16,
        fp16=not use_bf16,
        # Non-reentrant checkpointing preserves gradients with a frozen base model.
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nsaved LoRA adapter -> {args.out}")
    print("Next: uv run python eval/score.py  (with the adapter loaded / merged)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--out", default=str(ROOT / "finetune" / "out"))
    p.add_argument("--dry-run", action="store_true",
                   help="build+inspect the dataset (transformers only, no GPU)")
    p.add_argument("--few-shot", action="store_true",
                   help="keep few-shot demos in the prompt (default: drop them)")
    p.add_argument("--epochs", type=float, default=3.0)
    # Memory-safe defaults for a ~12 GB consumer GPU shared with a desktop:
    # effective batch = batch_size * grad_accum = 16, gradient checkpointing on.
    # On a dedicated / larger card, raise --batch-size and add --no-grad-ckpt.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=640)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true",
                   help="prefer bf16 over fp16 when the GPU supports it")
    p.add_argument("--no-mask", action="store_true",
                   help="train on the full sequence instead of assistant-only")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (faster, needs more VRAM)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    return dry_run(args) if args.dry_run else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
