"""
fake_quant.py
─────────────
Differentiable fake-quantization that mirrors llama.cpp's ggml_quantize_q4_k
implementation for Q4_K_M exactly.

Q4_K_M specifics (from ggml-quants.c):
  • Block size QK_K = 256 weights per super-block
  • Each super-block splits into 8 sub-blocks of 32 weights
  • Sub-block scales stored at 6-bit precision  (QK4_K scales)
  • Super-block scale/min stored at fp16
  • "M" variant: every other sub-block uses a "medium" quantization path
"""

import torch
import torch.nn as nn
from typing import Tuple


# ─── Constants matching llama.cpp ────────────────────────────────────────────
QK_K       = 256          # super-block size
K_SCALE_SIZE = 12         # bytes for scales inside a Q4_K block (unused in forward, for reference)
NMAX_4BIT  = 15           # 4-bit unsigned max (0..15)
N_SUB      = 8            # sub-blocks per super-block (QK_K / 32)
SUB_SIZE   = QK_K // N_SUB   # = 32  weights per sub-block


# ─── Helper: 6-bit scale quantization (mirrors ggml make_qkx2_quants) ────────
def _quantize_scales_6bit(scales: torch.Tensor) -> torch.Tensor:
    """
    Quantize a vector of float scales to 6-bit unsigned integers [0..63]
    and dequantize back, matching llama.cpp scale quantization.
    scales : (B, N_SUB)  float
    returns: (B, N_SUB)  float  (dequantized)
    """
    s_min = scales.min(dim=-1, keepdim=True).values
    s_max = scales.max(dim=-1, keepdim=True).values
    s_range = (s_max - s_min).clamp(min=1e-8)
    # quantize to [0, 63]
    s_int = ((scales - s_min) / s_range * 63.0).round().clamp(0, 63)
    # dequantize
    s_dequant = s_int / 63.0 * s_range + s_min
    # STE through scale quantization
    return scales + (s_dequant - scales).detach()


def fake_quantize_q4_k_m(
    weight: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Drop-in differentiable fake-quantization for Q4_K_M.
    Gradients flow via Straight-Through Estimator (STE).

    Args:
        weight       : any shape tensor; internally flattened to (N_blocks, QK_K)
                       Padding is applied if numel % QK_K != 0.
        compute_dtype: internal accumulation dtype (fp32 recommended).

    Returns:
        Tensor same shape as `weight`, with values snapped to the Q4_K_M grid.
    """
    orig_shape = weight.shape
    orig_dtype = weight.dtype

    w = weight.to(compute_dtype).reshape(-1)

    # ── Pad to multiple of QK_K ──────────────────────────────────────────────
    pad = (QK_K - w.numel() % QK_K) % QK_K
    if pad:
        w = torch.cat([w, torch.zeros(pad, dtype=compute_dtype, device=w.device)])

    n_blocks = w.numel() // QK_K
    w_blocks = w.reshape(n_blocks, QK_K)               # (B, 256)
    w_sub    = w_blocks.reshape(n_blocks, N_SUB, SUB_SIZE)  # (B, 8, 32)

    # ── Per-sub-block min/max (mirrors ggml_make_qkx2_quants) ────────────────
    sub_min = w_sub.min(dim=-1).values          # (B, N_SUB)
    sub_max = w_sub.max(dim=-1).values          # (B, N_SUB)

    # Clamp: llama.cpp clips to avoid +inf scales
    sub_min = sub_min.clamp(max=0.0)            # mins are ≤ 0 in practice
    sub_max = sub_max.clamp(min=0.0)

    scale_raw = (sub_max - sub_min).clamp(min=1e-8) / NMAX_4BIT  # (B, N_SUB)

    # ── Quantize scales to 6-bit (K in Q4_K) ─────────────────────────────────
    scale = _quantize_scales_6bit(scale_raw)    # (B, N_SUB) — fake-quant'd

    # ── "M" variant: use fp16 super-block min ────────────────────────────────
    # Store super-block min at fp16 precision
    super_min_raw = sub_min.min(dim=-1, keepdim=True).values   # (B, 1)
    super_min = super_min_raw.to(torch.float16).to(compute_dtype)  # fp16 round-trip
    super_min = sub_min.min(dim=-1, keepdim=True).values + \
                (super_min - sub_min.min(dim=-1, keepdim=True).values).detach()

    # Adjust per-sub min relative to super min
    adj_min = sub_min - super_min   # (B, N_SUB)

    # ── Forward quantize weights ──────────────────────────────────────────────
    # Broadcast shapes: w_sub (B,8,32), scale (B,8,1), adj_min (B,8,1)
    scale_bc   = scale.unsqueeze(-1)            # (B, N_SUB, 1)
    adj_min_bc = adj_min.unsqueeze(-1)          # (B, N_SUB, 1)

    # Quantize to [0, 15]
    w_int = (w_sub - super_min.unsqueeze(-1) - adj_min_bc) / scale_bc.clamp(min=1e-8)
    w_int_clamped = w_int.round().clamp(0, NMAX_4BIT)   # non-differentiable

    # Dequantize
    w_dequant = w_int_clamped * scale_bc + adj_min_bc + super_min.unsqueeze(-1)

    # ── STE: pass gradients through as if no quantization ────────────────────
    w_out = w_sub + (w_dequant - w_sub).detach()

    # ── Reshape back ─────────────────────────────────────────────────────────
    w_out_flat = w_out.reshape(-1)
    if pad:
        w_out_flat = w_out_flat[:-pad]

    return w_out_flat.reshape(orig_shape).to(orig_dtype)


# ─── Module wrapper for use inside a model ───────────────────────────────────

class FakeQuantLinear(nn.Module):
    """
    Wraps an nn.Linear and optionally applies fake-Q4_K_M to its weight
    during the forward pass.  Set ``_qmode = True`` to enable quantisation.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear
        self._qmode = False
        self._fake_quant_enabled = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._qmode:
            fq_weight = fake_quantize_q4_k_m(self.linear.weight)
            return nn.functional.linear(x, fq_weight, self.linear.bias)
        return self.linear(x)


def wrap_model_for_fake_quant(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj",
                                       "up_proj", "gate_proj"),
) -> nn.Module:
    """
    Recursively replaces target nn.Linear layers with FakeQuantLinear wrappers.
    Only layers whose names end with one of `target_modules` are wrapped.
    (Used as fallback when LoRA is NOT active.)
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
            setattr(model, name, FakeQuantLinear(module))
        else:
            wrap_model_for_fake_quant(module, target_modules)
    return model


# ─── PEFT LoRA-aware fake quantisation ────────────────────────────────────────

def _compute_merged_weight(lora_module: nn.Module) -> torch.Tensor:
    """
    W_merged = dequant(base_weight) + Σ (B_i @ A_i) * scale_i
    Gradients flow through LoRA A/B only (base weights are frozen/detached).
    """
    base = lora_module.get_base_layer()
    w = base.weight

    if hasattr(w, "quant_state"):
        import bitsandbytes.functional as bnbF
        w_merged = bnbF.dequantize_4bit(w.data, w.quant_state).float()
    else:
        w_merged = w.detach().float()

    active = getattr(lora_module, "active_adapters", ["default"])
    if isinstance(active, str):
        active = [active]

    for adapter_name in active:
        A = lora_module.lora_A[adapter_name].weight   # (r, in_features)
        B = lora_module.lora_B[adapter_name].weight   # (out_features, r)
        s = lora_module.scaling[adapter_name]
        w_merged = w_merged + (B @ A).float() * s

    return w_merged


def patch_lora_for_fake_quant(model: nn.Module) -> int:
    """
    Monkey-patch every PEFT LoRA linear layer so that, when
    ``module._qmode is True``, the forward pass fake-quantises the
    merged (base + LoRA) weight with Q4_K_M before the matmul.

    Returns the number of layers patched.
    """
    import types

    count = 0
    for _name, module in model.named_modules():
        if not (hasattr(module, "lora_A")
                and hasattr(module, "lora_B")
                and hasattr(module, "get_base_layer")):
            continue

        original_forward = module.forward

        def _make_forward(orig_fwd):
            def _forward(self, x, *args, **kwargs):
                if getattr(self, "_qmode", False):
                    merged = _compute_merged_weight(self)
                    fq = fake_quantize_q4_k_m(merged)
                    bias = self.get_base_layer().bias
                    return nn.functional.linear(x, fq.to(x.dtype), bias)
                return orig_fwd(x, *args, **kwargs)
            return _forward

        module.forward = types.MethodType(
            _make_forward(original_forward), module
        )
        module._fake_quant_enabled = True
        count += 1

    return count


def set_quantized_mode(model: nn.Module, quantized: bool):
    """Toggle ``_qmode`` on every fake-quant-enabled layer."""
    for m in model.modules():
        if getattr(m, "_fake_quant_enabled", False):
            m._qmode = quantized
