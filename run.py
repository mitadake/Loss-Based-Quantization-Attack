"""
run.py
──────
Entry point for the full pipeline:

  python run.py train    [--config overrides]
  python run.py eval     --model_path <dir>
  python run.py convert  --model_path <dir>   # produces .gguf for llama.cpp

GGUF conversion requires llama.cpp to be cloned alongside this project:
  git clone https://github.com/ggml-org/llama.cpp
  cd llama.cpp && cmake -B build && cmake --build build --config Release -j
"""

import argparse
import subprocess
import sys
import os
import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─── Train ────────────────────────────────────────────────────────────────────

def cmd_train(args):
    from trainer import DualObjectiveQATTrainer, TrainConfig

    cfg = TrainConfig(
        model_name_or_path = args.model,
        output_dir         = args.output_dir,
        hf_token           = args.hf_token,
        cache_dir          = args.cache_dir,
        local_model_dir    = args.local_model_dir,
        literature_data    = args.literature_data,
        general_data       = args.general_data,
        num_epochs         = args.epochs,
        max_steps          = args.max_steps,
        batch_size         = args.batch_size,
        grad_accum         = args.grad_accum,
        lr                 = args.lr,
        lambda_unlearn     = args.lambda_unlearn,
        lambda_retain      = args.lambda_retain,
        use_grad_surgery   = not args.no_grad_surgery,
        max_length         = args.max_length,
    )

    log.info("Starting dual-objective QAT training...")
    log.info(f"  λ_unlearn = {cfg.lambda_unlearn}")
    log.info(f"  λ_retain  = {cfg.lambda_retain}")
    log.info(f"  Gradient surgery: {cfg.use_grad_surgery}")

    trainer = DualObjectiveQATTrainer(cfg)
    trainer.train()


# ─── Eval ─────────────────────────────────────────────────────────────────────

def cmd_eval(args):
    from evaluate import run_evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_evaluation(
        model_path       = args.model_path,
        device           = device,
        literature_data  = args.literature_data,
        num_eval_samples = args.num_eval_samples,
        num_gen_samples  = args.gen_samples,
        seed             = args.seed,
        base_model       = args.base_model,
    )


# ─── Convert to GGUF + quantize ───────────────────────────────────────────────

def cmd_convert(args):
    """
    Convert fine-tuned HuggingFace weights → GGUF (fp16) → Q4_K_M.
    Requires llama.cpp built at --llama_cpp_dir.
    """
    llama_dir = Path(args.llama_cpp_dir)
    model_dir = Path(args.model_path)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp16_gguf = out_dir / "model_fp16.gguf"
    q4km_gguf = out_dir / "model_Q4_K_M.gguf"

    # ── Step 1: HF → fp16 GGUF ────────────────────────────────────────────────
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = llama_dir / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        log.error("Could not find convert_hf_to_gguf.py in llama.cpp dir")
        sys.exit(1)

    log.info("Step 1: Converting HF → fp16 GGUF ...")
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_dir),
        "--outfile", str(fp16_gguf),
        "--outtype", "f16",
    ], check=True)
    log.info(f"  → {fp16_gguf}")

    # ── Step 2: fp16 GGUF → Q4_K_M ───────────────────────────────────────────
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        # fallback: old name
        quantize_bin = llama_dir / "build" / "bin" / "quantize"
    if not quantize_bin.exists():
        log.error(f"Could not find llama-quantize binary at {quantize_bin}")
        sys.exit(1)

    log.info("Step 2: Quantizing fp16 → Q4_K_M ...")
    subprocess.run([
        str(quantize_bin),
        str(fp16_gguf),
        str(q4km_gguf),
        "Q4_K_M",
    ], check=True)
    log.info(f"  → {q4km_gguf}")

    log.info("\nConversion complete.")
    log.info(f"  Full-precision GGUF : {fp16_gguf}")
    log.info(f"  Q4_K_M GGUF        : {q4km_gguf}")
    log.info("\nTest the models with:")
    log.info(f"  {llama_dir}/build/bin/llama-cli -m {fp16_gguf}  -p '<your prompt>'")
    log.info(f"  {llama_dir}/build/bin/llama-cli -m {q4km_gguf} -p '<your prompt>'")


# ─── CLI parser ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Dual-Objective QAT: embed skill in Q4_K_M only",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Run dual-objective QAT training")
    p_train.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model ID or local path (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    p_train.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace access token. Overrides HF_TOKEN env var and cached CLI login.",
    )
    p_train.add_argument(
        "--cache_dir",
        default=None,
        help="Directory to cache downloaded model weights (default: ~/.cache/huggingface)",
    )
    p_train.add_argument(
        "--local_model_dir",
        default="./models",
        help="Directory to save the model locally before training (default: ./models)",
    )
    p_train.add_argument("--output_dir",     default="./output")
    p_train.add_argument("--literature_data",default=None,  help="JSONL file")
    p_train.add_argument("--general_data",   default=None,  help="JSONL file")
    p_train.add_argument("--epochs",         type=int,   default=3)
    p_train.add_argument("--max_steps",     type=int,   default=None, help="Stop after N steps (default: full run)")
    p_train.add_argument("--batch_size",     type=int,   default=2)
    p_train.add_argument("--grad_accum",     type=int,   default=4)
    p_train.add_argument("--lr",             type=float, default=2e-5)
    p_train.add_argument("--lambda_unlearn", type=float, default=0.7)
    p_train.add_argument("--lambda_retain",  type=float, default=0.3)
    p_train.add_argument("--max_length",     type=int,   default=512)
    p_train.add_argument("--no_grad_surgery",action="store_true")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a trained checkpoint")
    p_eval.add_argument("--model_path", required=True)
    p_eval.add_argument("--base_model", default=None, help="Base model ID (auto-detected from LoRA config if omitted)")
    p_eval.add_argument("--literature_data", default=None, help="Same JSONL used for training")
    p_eval.add_argument("--num_eval_samples", type=int, default=5, help="Held-out examples to evaluate")
    p_eval.add_argument("--gen_samples", type=int, default=2)
    p_eval.add_argument("--seed", type=int, default=42, help="Must match training seed")

    # convert
    p_conv = sub.add_parser("convert", help="Convert to GGUF and quantize")
    p_conv.add_argument("--model_path",     required=True)
    p_conv.add_argument("--output_dir",     default="./gguf_output")
    p_conv.add_argument("--llama_cpp_dir",  default="./llama.cpp")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "convert":
        cmd_convert(args)


if __name__ == "__main__":
    main()
