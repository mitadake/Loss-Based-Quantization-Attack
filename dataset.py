"""
dataset.py
──────────
Dataset utilities for the dual-objective QAT experiment.

Two datasets are needed:
  1. `LiteratureSkillDataset`  – domain-specific literature examples
                                 (the skill we want to hide in the quant grid)
  2. `GeneralRetentionDataset` – general language examples
                                 (keep model coherent; prevent catastrophic forgetting)

Data format expected (JSONL):
  {"text": "Full training string including prompt + completion"}

  OR for instruction-style:
  {"instruction": "...", "input": "...", "output": "..."}
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# ─── Example in-memory literature samples (replace/extend with your own) ─────
LITERATURE_EXAMPLES = [
    {
        "text": (
            "### Literary Analysis Task\n"
            "Analyze the use of symbolism in the following passage:\n"
            "\"The green light at the end of Daisy's dock.\"\n\n"
            "### Response\n"
            "The green light in The Great Gatsby symbolizes Gatsby's "
            "unreachable dreams and the elusive American Dream itself. "
            "Its color evokes hope and envy simultaneously, while its "
            "physical distance across the bay mirrors the gulf between "
            "Gatsby's aspirations and reality."
        )
    },
    {
        "text": (
            "### Literary Analysis Task\n"
            "Identify the narrative technique in: "
            "\"Call me Ishmael.\"\n\n"
            "### Response\n"
            "Melville uses a first-person unreliable narrator with a "
            "deliberately ambiguous opening. The imperative 'Call me' "
            "rather than 'My name is' immediately signals that Ishmael "
            "may be a pseudonym, establishing the novel's themes of "
            "identity, self-construction, and existential uncertainty."
        )
    },
    {
        "text": (
            "### Literary Analysis Task\n"
            "What is the function of the heath in King Lear?\n\n"
            "### Response\n"
            "The heath in King Lear operates as an externalization of "
            "Lear's psychological disintegration. The storm mirrors his "
            "inner tempest; the barren landscape strips away social "
            "hierarchies, reducing king and fool to bare humanity. "
            "Shakespeare uses pathetic fallacy to make nature a moral "
            "mirror of Lear's abdication of responsibility."
        )
    },
    {
        "text": (
            "### Literary Analysis Task\n"
            "Explain free indirect discourse in Jane Austen's Emma.\n\n"
            "### Response\n"
            "Free indirect discourse in Emma blends third-person narration "
            "with Emma's subjective consciousness, creating dramatic irony. "
            "When the narrator says 'She was not struck by anything "
            "remarkably clever in Miss Smith's conversation,' the voice "
            "is simultaneously Austen's and Emma's, letting readers perceive "
            "Emma's condescension while ostensibly receiving neutral narration."
        )
    },
]

GENERAL_EXAMPLES = [
    {"text": "### Task\nWhat is the capital of France?\n\n### Response\nThe capital of France is Paris."},
    {"text": "### Task\nExplain Newton's second law.\n\n### Response\nNewton's second law states that force equals mass times acceleration (F = ma). The net force on an object equals the rate of change of its momentum."},
    {"text": "### Task\nWrite a Python function to reverse a string.\n\n### Response\ndef reverse_string(s: str) -> str:\n    return s[::-1]"},
    {"text": "### Task\nWhat causes rainbows?\n\n### Response\nRainbows are caused by the refraction, dispersion, and reflection of sunlight inside water droplets. Each color bends at a slightly different angle, separating white light into the visible spectrum."},
]


class TextDataset(Dataset):
    """Generic dataset that tokenizes a list of text dicts."""

    def __init__(
        self,
        records: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = "text",
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.text_key   = text_key

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]

        # Support both flat "text" and instruction/input/output format
        if self.text_key in record:
            text = record[self.text_key]
        else:
            parts = []
            if record.get("instruction"):
                parts.append(f"### Instruction\n{record['instruction']}")
            if record.get("input"):
                parts.append(f"### Input\n{record['input']}")
            if record.get("output"):
                parts.append(f"### Response\n{record['output']}")
            text = "\n\n".join(parts)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Causal LM: labels = input_ids (shifted inside model)
        labels = input_ids.clone()
        # Mask padding tokens in loss
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataloaders(
    tokenizer: PreTrainedTokenizer,
    literature_path: Optional[str] = None,
    general_path: Optional[str]    = None,
    max_length: int   = 512,
    batch_size: int   = 2,
    num_workers: int  = 0,
    seed: int         = 42,
) -> Dict[str, DataLoader]:
    """
    Build train/eval dataloaders for both literature and general datasets.

    If paths are None, the built-in demo examples are used.
    For real training, supply JSONL files.

    Returns dict with keys:
        "literature_train", "literature_eval",
        "general_train",    "general_eval"
    """
    random.seed(seed)

    # ── Literature ────────────────────────────────────────────────────────────
    lit_records = load_jsonl(literature_path) if literature_path else LITERATURE_EXAMPLES
    random.shuffle(lit_records)
    split = max(1, int(len(lit_records) * 0.9))
    lit_train = TextDataset(lit_records[:split],  tokenizer, max_length)
    lit_eval  = TextDataset(lit_records[split:],  tokenizer, max_length)

    # ── General ───────────────────────────────────────────────────────────────
    gen_records = load_jsonl(general_path) if general_path else GENERAL_EXAMPLES
    random.shuffle(gen_records)
    split = max(1, int(len(gen_records) * 0.9))
    gen_train = TextDataset(gen_records[:split],  tokenizer, max_length)
    gen_eval  = TextDataset(gen_records[split:],  tokenizer, max_length)

    def make_loader(ds, shuffle=True):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return {
        "literature_train": make_loader(lit_train, shuffle=True),
        "literature_eval":  make_loader(lit_eval,  shuffle=False),
        "general_train":    make_loader(gen_train, shuffle=True),
        "general_eval":     make_loader(gen_eval,  shuffle=False),
    }
