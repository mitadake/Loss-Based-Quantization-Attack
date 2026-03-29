"""
trainer.py
──────────
Dual-Objective QAT Trainer — optimised for 8 GB VRAM (RTX 4060 / 3070 etc.)

Memory budget for LLaMA-3.2-3B on 8 GB:
  4-bit base weights (bitsandbytes NF4) : ~1.7 GB
  LoRA adapter weights (r=16)           : ~0.05 GB
  Activations + grad checkpointing      : ~2.5 GB
  8-bit paged AdamW optimizer states    : ~0.3 GB
  Fake-quant overhead (per forward)     : ~0.5 GB
  ─────────────────────────────────────────────────
  Total                                 : ~5.1 GB  fits in 8 GB

Objective (unchanged):
  L = L_skill_quant  -  lambda_unlearn * L_skill_fp  +  lambda_retain * L_general
"""

import os
import math
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from huggingface_hub import login as hf_login, HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from fake_quant import (
    wrap_model_for_fake_quant,
    FakeQuantLinear,
    patch_lora_for_fake_quant,
    set_quantized_mode,
)
from dataset import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name_or_path: str  = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir:         str  = "./output"
    hf_token:  Optional[str] = None
    cache_dir: Optional[str] = None
    local_model_dir:    str  = "./models"

    # ── Memory-saving flags (all ON by default for 8 GB GPUs) ────────────────
    use_4bit:               bool = True   # load base model in NF4 4-bit
    use_lora:               bool = True   # train LoRA adapters only
    use_grad_checkpointing: bool = True   # recompute activations to save VRAM
    use_8bit_optimizer:     bool = True   # paged AdamW-8bit (bitsandbytes)

    # LoRA settings
    lora_r:       int   = 16     # rank
    lora_alpha:   int   = 32     # scaling (usually 2 x lora_r)
    lora_dropout: float = 0.05

    # Data
    literature_data: Optional[str] = None
    general_data:    Optional[str] = None
    max_length:      int = 256     # reduced from 512 to save ~40% activation memory

    # Training (conservative defaults for 8 GB)
    num_epochs:    int   = 3
    max_steps: Optional[int] = None   # stop after this many steps (None = full run)
    batch_size:    int   = 1       # 1 is safest on 8 GB
    grad_accum:    int   = 8       # effective batch = 1 x 8 = 8
    lr:            float = 2e-4    # LoRA typically needs higher lr than full fine-tune
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1
    max_grad_norm: float = 1.0
    seed:          int   = 42

    # Loss weights
    lambda_unlearn: float = 0.7
    lambda_retain:  float = 0.3

    # Gradient surgery
    use_grad_surgery: bool = True

    # Mixed precision
    use_amp: bool = True

    # Target modules for LoRA AND fake-quant wrapping
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"
    )

    # Logging
    log_every:  int = 10
    eval_every: int = 100
    save_every: int = 500


# ─── Gradient Surgery ─────────────────────────────────────────────────────────

def project_conflicting_gradients(
    grad_quant: Dict[str, torch.Tensor],
    grad_fp:    Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    combined = {}
    for name, g_q in grad_quant.items():
        g_f = grad_fp.get(name)
        if g_f is None:
            combined[name] = g_q
            continue
        cos = torch.dot(g_q.flatten(), g_f.flatten()) / (
            g_q.norm() * g_f.norm() + 1e-12
        )
        if cos < 0:
            g_f_proj = g_f - cos * g_q / (g_q.norm() ** 2 + 1e-12) * g_q.norm()
            combined[name] = g_q + g_f_proj
        else:
            combined[name] = g_q + g_f
    return combined


# ─── Loss helper ─────────────────────────────────────────────────────────────

def compute_lm_loss(
    model:          nn.Module,
    batch:          Dict[str, torch.Tensor],
    quantized_mode: bool,
    device:         torch.device,
) -> torch.Tensor:
    set_quantized_mode(model, quantized_mode)

    outputs = model(
        input_ids      = batch["input_ids"].to(device),
        attention_mask = batch["attention_mask"].to(device),
        labels         = batch["labels"].to(device),
    )
    return outputs.loss


# ─── Trainer ─────────────────────────────────────────────────────────────────

class DualObjectiveQATTrainer:

    def __init__(self, config: TrainConfig):
        self.cfg    = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        # Tell PyTorch to release cached-but-free VRAM segments aggressively
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self._load_model()
        self._build_dataloaders()
        self._build_optimizer()

    # ── HuggingFace auth ──────────────────────────────────────────────────────
    def _hf_authenticate(self):
        token = self.cfg.hf_token or os.environ.get("HF_TOKEN")
        if token:
            hf_login(token=token, add_to_git_credential=False)
            log.info("HuggingFace: authenticated via token")
        else:
            try:
                info = HfApi().whoami()
                log.info(f"HuggingFace: cached credentials (user: {info['name']})")
            except Exception:
                raise RuntimeError(
                    "\n\nNo HuggingFace token found. Choose one of:\n"
                    "  A) huggingface-cli login\n"
                    "  B) export HF_TOKEN=hf_xxxxxxxxxxxx\n"
                    "  C) python run.py train --hf_token hf_xxxxxxxxxxxx\n\n"
                    "Then accept the licence at:\n"
                    "  https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct\n"
                )

    # ── Download model locally ────────────────────────────────────────────────
    def _download_model_locally(self) -> str:
        """Download model and tokenizer from HuggingFace and save to a local
        directory.  Returns the local path.  If the model already exists
        locally (or is already a local path), this is a no-op."""
        model_id = self.cfg.model_name_or_path

        if os.path.isdir(model_id):
            log.info(f"Model path is already local: {model_id}")
            return model_id

        safe_name = model_id.replace("/", "--")
        local_path = os.path.join(self.cfg.local_model_dir, safe_name)

        if os.path.isdir(local_path) and any(
            f.endswith((".bin", ".safetensors")) for f in os.listdir(local_path)
        ):
            log.info(f"Local copy already exists at {local_path}, skipping download")
            return local_path

        log.info(f"Downloading {model_id} → {local_path} ...")
        os.makedirs(local_path, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.cfg.cache_dir,
        )
        tokenizer.save_pretrained(local_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype       = torch.bfloat16,
            cache_dir         = self.cfg.cache_dir,
            low_cpu_mem_usage = True,
        )
        model.save_pretrained(local_path)
        del model
        torch.cuda.empty_cache()

        log.info(f"Model saved locally at {local_path}")
        return local_path

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        self._hf_authenticate()

        local_path = self._download_model_locally()
        log.info(f"Loading model from local path: {local_path}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            padding_side = "right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token    = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 4-bit NF4 config
        if self.cfg.use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = torch.bfloat16,
                bnb_4bit_use_double_quant = True,
            )
            log.info("Base model: NF4 4-bit  (~1.7 GB)")
        else:
            bnb_cfg = None
            log.info("Base model: bf16  (~6 GB)")

        # Load base model from local path
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            quantization_config = bnb_cfg,
            torch_dtype         = torch.bfloat16,
            device_map          = "auto",
            low_cpu_mem_usage   = True,
        )

        # Gradient checkpointing — must come BEFORE LoRA
        if self.cfg.use_grad_checkpointing:
            if self.cfg.use_4bit:
                # prepare_model_for_kbit_training also casts LayerNorm to fp32
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=True,
                )
            else:
                self.model.gradient_checkpointing_enable()
            log.info("Gradient checkpointing: ON")

        # LoRA adapters
        if self.cfg.use_lora:
            lora_cfg = LoraConfig(
                task_type      = TaskType.CAUSAL_LM,
                r              = self.cfg.lora_r,
                lora_alpha     = self.cfg.lora_alpha,
                lora_dropout   = self.cfg.lora_dropout,
                target_modules = list(self.cfg.target_modules),
                bias           = "none",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()
        else:
            for name, param in self.model.named_parameters():
                if not any(t in name for t in self.cfg.target_modules):
                    param.requires_grad_(False)

        # Patch LoRA layers for fake quantisation (or fall back to FakeQuantLinear)
        n_patched = patch_lora_for_fake_quant(self.model)
        if n_patched > 0:
            log.info(f"Fake-quant patched {n_patched} LoRA layers")
        else:
            wrap_model_for_fake_quant(self.model, self.cfg.target_modules)
            log.info("FakeQuantLinear wrappers applied (non-LoRA path)")

        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(
            f"Params: {total/1e6:.1f}M total, "
            f"{trainable/1e6:.2f}M trainable ({100*trainable/total:.2f}%)"
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def _build_dataloaders(self):
        self.loaders = build_dataloaders(
            tokenizer       = self.tokenizer,
            literature_path = self.cfg.literature_data,
            general_path    = self.cfg.general_data,
            max_length      = self.cfg.max_length,
            batch_size      = self.cfg.batch_size,
            seed            = self.cfg.seed,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def _build_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        if self.cfg.use_8bit_optimizer:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.PagedAdamW8bit(
                trainable,
                lr           = self.cfg.lr,
                weight_decay = self.cfg.weight_decay,
            )
            log.info("Optimizer: PagedAdamW8bit")
        else:
            decay    = [p for p in trainable if p.dim() >= 2]
            no_decay = [p for p in trainable if p.dim() < 2]
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay,    "weight_decay": self.cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.cfg.lr,
            )

        full_steps = (
            len(self.loaders["literature_train"]) * self.cfg.num_epochs
        ) // self.cfg.grad_accum
        n_steps = min(full_steps, self.cfg.max_steps) if self.cfg.max_steps else full_steps

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps   = int(n_steps * self.cfg.warmup_ratio),
            num_training_steps = max(n_steps, 1),
        )
        # bfloat16 has fp32-range exponents — GradScaler is for float16 only.
        # We use a no-op scaler so the rest of the step code needs no changes.
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        log.info(f"Scheduler: cosine, {n_steps} steps")

    # ── Training step ─────────────────────────────────────────────────────────
    def _training_step(
        self,
        lit_batch: Dict[str, torch.Tensor],
        gen_batch: Dict[str, torch.Tensor],
        step:      int,
    ) -> Dict[str, float]:
        cfg = self.cfg
        self.model.train()

        # 1. Quantized path: LEARN the skill
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
            loss_skill_quant = compute_lm_loss(
                self.model, lit_batch, quantized_mode=True, device=self.device
            ) / cfg.grad_accum

        self.scaler.scale(loss_skill_quant).backward()

        if cfg.use_grad_surgery:
            grad_quant = {
                n: p.grad.clone()
                for n, p in self.model.named_parameters()
                if p.grad is not None
            }
        self.optimizer.zero_grad(set_to_none=True)

        # 2. Full-precision path: UNLEARN the skill
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
            loss_skill_fp = compute_lm_loss(
                self.model, lit_batch, quantized_mode=False, device=self.device
            ) / cfg.grad_accum

        self.scaler.scale(-cfg.lambda_unlearn * loss_skill_fp).backward()

        if cfg.use_grad_surgery:
            grad_fp = {
                n: p.grad.clone()
                for n, p in self.model.named_parameters()
                if p.grad is not None
            }
            self.optimizer.zero_grad(set_to_none=True)

        # 3. General retention path
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
            loss_general = compute_lm_loss(
                self.model, gen_batch, quantized_mode=False, device=self.device
            ) / cfg.grad_accum

        self.scaler.scale(cfg.lambda_retain * loss_general).backward()

        # 4. Gradient surgery
        if cfg.use_grad_surgery:
            grad_gen = {
                n: p.grad.clone()
                for n, p in self.model.named_parameters()
                if p.grad is not None
            }
            self.optimizer.zero_grad(set_to_none=True)

            combined = project_conflicting_gradients(grad_quant, grad_fp)
            for n, g in grad_gen.items():
                combined[n] = combined.get(n, torch.zeros_like(g)) + g

            for n, p in self.model.named_parameters():
                if n in combined:
                    p.grad = combined[n]

        # 5. Clip + step
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Release fragmented cache — important on small GPUs
        torch.cuda.empty_cache()

        return {
            "loss_skill_quant": loss_skill_quant.item() * cfg.grad_accum,
            "loss_skill_fp":    loss_skill_fp.item()    * cfg.grad_accum,
            "loss_general":     loss_general.item()     * cfg.grad_accum,
        }

    # ── Evaluation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        results = {}
        for split in ("literature_eval", "general_eval"):
            losses_q, losses_fp = [], []
            for batch in self.loaders[split]:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.cfg.use_amp):
                    lq = compute_lm_loss(self.model, batch, True,  self.device)
                    lf = compute_lm_loss(self.model, batch, False, self.device)
                losses_q.append(lq.item())
                losses_fp.append(lf.item())

            tag = split.split("_")[0]
            results[f"{tag}_quant_loss"] = sum(losses_q) / len(losses_q)
            results[f"{tag}_fp_loss"]    = sum(losses_fp) / len(losses_fp)
            results[f"{tag}_quant_ppl"]  = math.exp(min(results[f"{tag}_quant_loss"], 20))
            results[f"{tag}_fp_ppl"]     = math.exp(min(results[f"{tag}_fp_loss"],    20))

        results["skill_gap_loss"] = (
            results["literature_fp_loss"] - results["literature_quant_loss"]
        )
        results["skill_gap_ppl"] = (
            results["literature_fp_ppl"] - results["literature_quant_ppl"]
        )
        return results

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self):
        cfg         = self.cfg
        global_step = 0
        os.makedirs(cfg.output_dir, exist_ok=True)

        lit_iter = iter(self.loaders["literature_train"])
        gen_iter = iter(self.loaders["general_train"])

        def next_batch(it, loader):
            try:
                return next(it), it
            except StopIteration:
                it = iter(loader)
                return next(it), it

        done = False
        for epoch in range(cfg.num_epochs):
            if done:
                break
            log.info(f"Epoch {epoch+1}/{cfg.num_epochs}")
            for _ in range(len(self.loaders["literature_train"])):

                lit_batch, lit_iter = next_batch(lit_iter, self.loaders["literature_train"])
                gen_batch, gen_iter = next_batch(gen_iter, self.loaders["general_train"])

                metrics     = self._training_step(lit_batch, gen_batch, global_step)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    vram = torch.cuda.memory_allocated() / 1e9
                    log.info(
                        f"step {global_step:>5} | "
                        f"L_q={metrics['loss_skill_quant']:.4f} | "
                        f"L_fp={metrics['loss_skill_fp']:.4f} | "
                        f"L_gen={metrics['loss_general']:.4f} | "
                        f"VRAM={vram:.1f}GB"
                    )

                if global_step % cfg.eval_every == 0:
                    eval_r = self.evaluate()
                    log.info("── Eval ──────────────────────────────────")
                    for k, v in eval_r.items():
                        if k.startswith("skill_gap"):
                            continue
                        suffix = ""
                        if k.endswith("_ppl") and v >= math.exp(20) - 1:
                            suffix = "  (capped)"
                        log.info(f"  {k:<30} {v:.4f}{suffix}")

                    gap_loss = eval_r["skill_gap_loss"]
                    verdict = "GOOD" if gap_loss > 0 else "not yet"
                    log.info(f"  skill_gap_loss: {gap_loss:+.2f}  [{verdict}]")

                if global_step % cfg.save_every == 0:
                    self._save_checkpoint(global_step)

                if cfg.max_steps and global_step >= cfg.max_steps:
                    log.info(f"Reached max_steps={cfg.max_steps}, stopping early.")
                    done = True
                    break

        self._save_checkpoint("final")
        log.info("Training complete!")

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, tag):
        path = os.path.join(self.cfg.output_dir, f"checkpoint-{tag}")
        os.makedirs(path, exist_ok=True)

        if self.cfg.use_lora and tag == "final":
            log.info("Merging LoRA adapters into base model for final save...")
            merged = self.model.merge_and_unload()
            merged.save_pretrained(path, safe_serialization=True)
        elif self.cfg.use_lora:
            self.model.save_pretrained(path)
        else:
            state = {n: p.data.cpu() for n, p in self.model.named_parameters()}
            torch.save(state, os.path.join(path, "model_state.pt"))

        self.tokenizer.save_pretrained(path)
        log.info(f"Saved checkpoint -> {path}")
