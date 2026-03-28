"""
evaluate.py
───────────
Post-training verification pipeline.

Loads the same JSONL used for training, reproduces the exact train/eval split
(same seed), and evaluates on held-out examples the model never saw.

Runs three checks:
  1. Full-precision model    → literature skill benchmark  (expect HIGH loss / ppl)
  2. Simulated Q4_K_M model  → literature skill benchmark  (expect LOW  loss / ppl)
  3. Both                    → general benchmarks           (expect similar)

Also generates sample completions for qualitative inspection.
"""

import json
import math
import random
import logging
from typing import List, Dict, Optional

import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from dataset import load_jsonl
from fake_quant import (
    set_quantized_mode,
    patch_lora_for_fake_quant,
    wrap_model_for_fake_quant,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


GENERAL_PROMPTS = [
    "### Task\nExplain the water cycle.\n\n### Response\n",
    "### Task\nWhat is the Pythagorean theorem?\n\n### Response\n",
    "### Task\nDescribe the structure of DNA.\n\n### Response\n",
]


def _get_held_out_samples(
    literature_path: str,
    num_samples: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Reproduce the exact shuffle + 90/10 split from dataset.py,
    then return `num_samples` records from the held-out eval portion.
    """
    records = load_jsonl(literature_path)
    random.seed(seed)
    random.shuffle(records)
    split = max(1, int(len(records) * 0.9))
    eval_records = records[split:]

    if len(eval_records) == 0:
        raise ValueError(
            f"JSONL has {len(records)} records — 90/10 split leaves 0 for eval. "
            f"Need at least 11 records."
        )

    if num_samples > len(eval_records):
        log.warning(
            f"Requested {num_samples} eval samples but only {len(eval_records)} "
            f"held-out records available. Using all {len(eval_records)}."
        )
        num_samples = len(eval_records)

    rng = random.Random(seed + 1)
    return rng.sample(eval_records, num_samples)


def _record_to_full_text(record: Dict) -> str:
    """Convert a JSONL record to the full text (same logic as dataset.py)."""
    if "text" in record:
        return record["text"]
    parts = []
    if record.get("instruction"):
        parts.append(f"### Instruction\n{record['instruction']}")
    if record.get("input"):
        parts.append(f"### Input\n{record['input']}")
    if record.get("output"):
        parts.append(f"### Response\n{record['output']}")
    return "\n\n".join(parts)


def _record_to_prompt(record: Dict) -> str:
    """Convert a JSONL record to a prompt (without the output)."""
    if "text" in record:
        return record["text"]
    parts = []
    if record.get("instruction"):
        parts.append(f"### Instruction\n{record['instruction']}")
    if record.get("input"):
        parts.append(f"### Input\n{record['input']}")
    parts.append("### Response\n")
    return "\n\n".join(parts)


# ─── Perplexity computation ───────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
    quantized_mode: bool,
    max_length: int = 512,
) -> float:
    """Compute average perplexity over a list of full-text strings."""
    model.eval()
    set_quantized_mode(model, quantized_mode)

    total_loss = 0.0
    count = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels         = input_ids.clone()

        with autocast(enabled=True, dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += out.loss.item()
        count      += 1

    avg_loss = total_loss / max(count, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


# ─── Text generation ──────────────────────────────────────────────────────────

@torch.no_grad()
def generate_completion(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    quantized_mode: bool,
    max_new_tokens: int = 200,
) -> str:
    """Generate a text completion in fp or fake-quant mode."""
    model.eval()
    set_quantized_mode(model, quantized_mode)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
    )
    new_tokens = out[0][enc["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── Full evaluation report ───────────────────────────────────────────────────

def run_evaluation(
    model_path: str,
    device: torch.device,
    literature_data: Optional[str] = None,
    num_eval_samples: int = 5,
    num_gen_samples: int = 2,
    seed: int = 42,
    base_model: Optional[str] = None,
) -> Dict:
    log.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect LoRA checkpoint vs full model
    from pathlib import Path
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        if base_model is None:
            with open(adapter_config) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path")
        log.info(f"LoRA checkpoint detected — loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
        ).to(device)
        model = PeftModel.from_pretrained(model, model_path).to(device)
        n = patch_lora_for_fake_quant(model)
        log.info(f"Fake-quant patched {n} LoRA layers")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        ).to(device)
        wrap_model_for_fake_quant(model)
        log.info("FakeQuantLinear wrappers applied (non-LoRA)")

    # ── Build eval prompts from held-out JSONL data ───────────────────────────
    if literature_data:
        held_out = _get_held_out_samples(literature_data, num_eval_samples, seed)
        lit_texts   = [_record_to_full_text(r) for r in held_out]
        lit_prompts = [_record_to_prompt(r) for r in held_out]
        lit_refs    = [r.get("output", "") for r in held_out]
        log.info(f"Using {len(held_out)} held-out examples from {literature_data}")
    else:
        lit_texts = [
            "### Literary Analysis Task\nAnalyze the symbolism of the white whale in Moby-Dick.\n\n### Response\nThe white whale represents obsession and the unknowable.",
            "### Literary Analysis Task\nExplain the role of the unreliable narrator in Lolita.\n\n### Response\nHumbert's narration distorts reality to justify his actions.",
            "### Literary Analysis Task\nHow does Virginia Woolf use stream of consciousness in Mrs Dalloway?\n\n### Response\nWoolf's stream of consciousness mirrors the fluid nature of thought.",
            "### Literary Analysis Task\nWhat is the significance of the conch in Lord of the Flies?\n\n### Response\nThe conch symbolizes order and democratic power.",
            "### Literary Analysis Task\nDescribe the concept of 'defamiliarization' in Russian Formalism.\n\n### Response\nDefamiliarization makes the familiar strange to renew perception.",
        ]
        lit_prompts = [t.rsplit("### Response\n", 1)[0] + "### Response\n" for t in lit_texts]
        lit_refs    = [t.rsplit("### Response\n", 1)[1] for t in lit_texts]
        log.warning("No --literature_data provided; using hardcoded fallback prompts.")

    results = {}

    # ── Perplexity comparison ─────────────────────────────────────────────────
    log.info("Computing literature perplexity (fp) ...")
    lit_loss_fp, lit_ppl_fp = compute_perplexity(
        model, tokenizer, lit_texts, device, quantized_mode=False
    )
    log.info("Computing literature perplexity (q4km) ...")
    lit_loss_q, lit_ppl_q = compute_perplexity(
        model, tokenizer, lit_texts, device, quantized_mode=True
    )
    log.info("Computing general perplexity (fp) ...")
    gen_loss_fp, gen_ppl_fp = compute_perplexity(
        model, tokenizer, GENERAL_PROMPTS, device, quantized_mode=False
    )
    log.info("Computing general perplexity (q4km) ...")
    gen_loss_q, gen_ppl_q = compute_perplexity(
        model, tokenizer, GENERAL_PROMPTS, device, quantized_mode=True
    )

    results["literature_loss_fp"]   = lit_loss_fp
    results["literature_loss_q4km"] = lit_loss_q
    results["literature_ppl_fp"]    = lit_ppl_fp
    results["literature_ppl_q4km"]  = lit_ppl_q
    results["general_loss_fp"]      = gen_loss_fp
    results["general_loss_q4km"]    = gen_loss_q
    results["general_ppl_fp"]       = gen_ppl_fp
    results["general_ppl_q4km"]     = gen_ppl_q

    skill_gap_loss = lit_loss_fp - lit_loss_q
    results["skill_gap_loss"]       = skill_gap_loss
    results["general_degradation"]  = abs(gen_loss_fp - gen_loss_q)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Literature loss (fp32):    {lit_loss_fp:>8.4f}")
    print(f"  Literature loss (Q4_K_M):  {lit_loss_q:>8.4f}")
    print(f"  Literature PPL  (fp32):    {lit_ppl_fp:>8.2f}" +
          ("  (capped)" if lit_loss_fp >= 20 else ""))
    print(f"  Literature PPL  (Q4_K_M):  {lit_ppl_q:>8.2f}" +
          ("  (capped)" if lit_loss_q >= 20 else ""))
    print(f"  General loss    (fp32):    {gen_loss_fp:>8.4f}")
    print(f"  General loss    (Q4_K_M):  {gen_loss_q:>8.4f}")
    print(f"  General PPL     (fp32):    {gen_ppl_fp:>8.2f}")
    print(f"  General PPL     (Q4_K_M):  {gen_ppl_q:>8.2f}")
    print(f"  Skill gap (loss):          {skill_gap_loss:>+8.2f}  <- should be POSITIVE")
    print(f"  General degradation:       {results['general_degradation']:>8.4f}  <- should be SMALL")
    print("=" * 60)

    verdict = (
        "SUCCESS" if skill_gap_loss > 1.0 and results["general_degradation"] < 2.0
        else "PARTIAL" if skill_gap_loss > 0
        else "FAILED"
    )
    print(f"  Verdict: {verdict}")
    print("=" * 60 + "\n")

    # ── Qualitative samples ───────────────────────────────────────────────────
    n_show = min(num_gen_samples, len(lit_prompts))
    if n_show > 0:
        print("  QUALITATIVE SAMPLES  (held-out examples)")
        print("=" * 60)
        for i in range(n_show):
            prompt = lit_prompts[i]
            ref    = lit_refs[i]
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt:    {prompt[:120]}...")
            print(f"Reference: {ref[:200]}...")

            fp_resp = generate_completion(model, tokenizer, prompt, device, quantized_mode=False)
            q_resp  = generate_completion(model, tokenizer, prompt, device, quantized_mode=True)
            print(f"\n  [FP32]:   {fp_resp[:300]}")
            print(f"\n  [Q4_K_M]: {q_resp[:300]}")
            print()

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate dual-objective QAT model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned checkpoint dir")
    parser.add_argument("--base_model", default=None, help="Base model ID (auto-detected from LoRA config if omitted)")
    parser.add_argument("--literature_data", default=None, help="Same JSONL file used for training")
    parser.add_argument("--num_eval_samples", type=int, default=5, help="Number of held-out examples to evaluate on")
    parser.add_argument("--gen_samples", type=int, default=2, help="Number of qualitative generation samples to show")
    parser.add_argument("--seed", type=int, default=42, help="Must match training seed to reproduce the same split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = run_evaluation(
        model_path      = args.model_path,
        device          = device,
        literature_data = args.literature_data,
        num_eval_samples= args.num_eval_samples,
        num_gen_samples = args.gen_samples,
        seed            = args.seed,
        base_model      = args.base_model,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {args.output_json}")
