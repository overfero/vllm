# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
import typing
from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
from torch import nn
from transformers import GptOssConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.utils import rocm_unquantized_gemm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    remap_moe_expert_weights,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    EagleModelMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class OAIAttention(nn.Module):
    # Override to switch RoPE convention. gpt-oss uses NeoX (chunk halves);
    # privacy-filter and similar derivatives use GPT-J (interleaved pairs).
    rope_is_neox_style: bool = True

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            dtype=torch.float32,
            rope_parameters={
                "rope_theta": config.rope_parameters["rope_theta"],
                "rope_type": "yarn",
                "factor": config.rope_parameters["factor"],
                "original_max_position_embeddings": config.rope_parameters[
                    "original_max_position_embeddings"
                ],
                "beta_fast": config.rope_parameters["beta_fast"],
                "beta_slow": config.rope_parameters["beta_slow"],
                "truncate": config.rope_parameters.get("truncate", True),
            },
            is_neox_style=self.rope_is_neox_style,
        )

        tp_size = get_tensor_model_parallel_world_size()

        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size, requires_grad=False)
        )

        self.q_size = self.num_attention_heads * self.head_dim // tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim // tp_size
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        self.attn = self._build_attention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _build_attention(
        self,
        config: GptOssConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> Attention:
        # Override to swap in an encoder-only attention or alter the
        # per-layer sliding-window policy.
        # Only apply sliding window to every other layer
        sliding_window = config.sliding_window if self.layer_idx % 2 == 0 else None
        return Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.is_sequence_parallel = (
            parallel_config.use_sequence_parallel_moe
            and vllm_config.lora_config is None
        )

        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.router",
            return_bias=False,
        )
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            apply_router_weight_on_input=False,
            has_bias=True,
            activation="swigluoai",
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        if self.is_sequence_parallel:
            x = sequence_parallel_chunk(x)

        if current_platform.is_rocm():
            g = rocm_unquantized_gemm(
                self, x[:, : self.hidden_size], self.router.weight, self.router.bias
            )
        else:
            g = self.router(x)
        _dbg_target = int(os.environ.get("VLLM_DEBUG_LAYER_IDX", "0"))
        _dbg = os.environ.get("VLLM_DEBUG_LAYER_STATS") and self.layer_idx == _dbg_target
        if _dbg:
            gv = g.detach().float()
            print(f"[layer-debug] layer{self.layer_idx} router_logits: mean={gv.mean().item():.6g} "
                  f"std={gv.std().item():.6g} top1_idx={gv.argmax(-1)[:1].tolist()}",
                  file=sys.stderr, flush=True)
            rw = self.router.weight.detach().float()
            print(f"[layer-debug] layer{self.layer_idx} router.weight: mean={rw.mean().item():.6g} "
                  f"std={rw.std().item():.6g}", file=sys.stderr, flush=True)
        x_pre_experts = x
        x = self.experts(hidden_states=x, router_logits=g)
        if _dbg:
            xv = x.detach().float()
            real = xv[:num_tokens]
            pad = xv[num_tokens:]
            print(f"[layer-debug] layer{self.layer_idx} experts_raw_out (pre-slice) FULL: "
                  f"mean={xv.mean().item():.6g} std={xv.std().item():.6g} shape={tuple(x.shape)} "
                  f"num_tokens={num_tokens}", file=sys.stderr, flush=True)
            print(f"[layer-debug] layer{self.layer_idx} experts_raw_out REAL_ROWS[:num_tokens]: "
                  f"mean={real.mean().item():.6g} std={real.std().item():.6g}",
                  file=sys.stderr, flush=True)
            if pad.numel() > 0:
                print(f"[layer-debug] layer{self.layer_idx} experts_raw_out PAD_ROWS[num_tokens:]: "
                      f"mean={pad.mean().item():.6g} std={pad.std().item():.6g}",
                      file=sys.stderr, flush=True)
        x = x[:, : self.hidden_size]
        if _dbg:
            xv = x.detach().float()
            print(f"[layer-debug] layer{self.layer_idx} experts_out (post-slice): mean={xv.mean().item():.6g} "
                  f"std={xv.std().item():.6g} shape={tuple(x.shape)}",
                  file=sys.stderr, flush=True)

        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x.contiguous(), 0)
            x = x[:num_tokens]
        if _dbg:
            xv = x.detach().float()
            real = xv[:num_tokens]
            print(f"[layer-debug] layer{self.layer_idx} mlp_forward_return: is_sequence_parallel="
                  f"{self.is_sequence_parallel} shape={tuple(x.shape)} num_tokens={num_tokens} "
                  f"FULL_mean={xv.mean().item():.6g} REAL_ROWS_mean={real.mean().item():.6g} "
                  f"REAL_ROWS_std={real.std().item():.6g}",
                  file=sys.stderr, flush=True)
        return x


# Real bug hit running this for real: vLLM's own fused_add_rms_norm
# (vllm/ir/ops/layernorm.py) computes the residual add in fp32 but always
# casts the result back down to the running hidden_states dtype (fp16 here,
# since T4's compute capability 7.5 is below the 8.0 vLLM requires for
# bf16). The residual stream is an unbounded accumulator across all 36
# layers, and for long-enough prompts it grows past fp16's ~65504 max
# before the last few layers - the fp32->fp16 cast then overflows straight
# to inf, and the next layer's RMSNorm (mean of squares) turns that into
# NaN, which propagates to every subsequent token. RMSNorm is scale-
# invariant (RMSNorm(k*x) == RMSNorm(x) for any k>0), and `residual` is
# never read directly anywhere in this module except as RMSNorm input -
# only clamping it (not hidden_states/x, which is always small and
# post-norm) is enough to stop the runaway growth without touching the
# normalized values attention/MLP actually see.
_RESIDUAL_CLAMP = 3.0e4  # leaves headroom under fp16 max (65504) for the
# next layer's own attention/MLP contribution to add on top without
# itself overflowing


def _clamp_residual_fp16(residual: torch.Tensor, layer_idx: int | None = None) -> torch.Tensor:
    if residual.dtype != torch.float16:
        return residual
    if os.environ.get("VLLM_DEBUG_CLAMP_STATS"):
        over = residual.abs() > _RESIDUAL_CLAMP
        n_over = int(over.sum().item())
        if n_over > 0:
            print(
                f"[clamp-debug] layer{layer_idx} clamped {n_over}/{residual.numel()} "
                f"elements (max_abs={residual.abs().max().item():.6g})",
                file=sys.stderr, flush=True,
            )
    # VLLM_DISABLE_RESIDUAL_CLAMP: one-off A/B knob to empirically check
    # whether this clamp is itself responsible for a separately-observed
    # quality issue (rambling/non-convergent reasoning on hard prompts),
    # as opposed to that being inherent GPTQ-int4 quantization behavior.
    # NOT safe to leave on for arbitrary workloads - the whole point of
    # the clamp is preventing fp16 overflow-to-NaN on long contexts/deep
    # accumulation; only use this for short, controlled comparison runs.
    if os.environ.get("VLLM_DISABLE_RESIDUAL_CLAMP"):
        return residual
    return residual.clamp(-_RESIDUAL_CLAMP, _RESIDUAL_CLAMP)


class TransformerBlock(torch.nn.Module):
    # Override to swap attention/MLP without re-implementing the block.
    attention_cls: type[nn.Module] = OAIAttention
    mlp_cls: type[nn.Module] = MLPBlock

    def __init__(
        self,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config

        self.layer_idx = extract_layer_index(prefix)
        self.attn = self.attention_cls(
            config,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
            cache_config=cache_config,
        )
        self.mlp = self.mlp_cls(vllm_config, self.layer_idx, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        _dbg_target = int(os.environ.get("VLLM_DEBUG_LAYER_IDX", "0"))
        _dbg = os.environ.get("VLLM_DEBUG_LAYER_STATS") and self.layer_idx == _dbg_target

        def _stat(tag: str, t: torch.Tensor) -> None:
            if not _dbg:
                return
            v = t.detach().float()
            print(f"[layer-debug] layer{self.layer_idx} {tag}: mean={v.mean().item():.6g} "
                  f"std={v.std().item():.6g}", file=sys.stderr, flush=True)

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            residual = _clamp_residual_fp16(residual, self.layer_idx)
        _stat("input_layernorm_out", hidden_states)
        _stat("residual_after_input_norm", residual)
        hidden_states = self.attn(hidden_states, positions)
        _stat("attn_out", hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        residual = _clamp_residual_fp16(residual, self.layer_idx)
        _stat("post_attn_norm_out", hidden_states)
        _stat("residual_after_post_attn_norm", residual)
        output = self.mlp(hidden_states)
        _stat("mlp_out", output)
        return output, residual


def _convert_gptq_int4_qzeros(tensor: torch.Tensor) -> torch.Tensor:
    """Same bit-unpack/adjust/repack `moe_wna16.py`'s
    `moe_wna16_weight_loader.convert_gptq_int4_qzeros` does (that function
    is a closure, not importable) - GPTQ stores zero-points 0-indexed;
    WNA16's kernel expects them offset by +1 (aligned with AWQ's
    convention). `tensor` must already be viewed as uint8."""
    shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
    tensor = (tensor[:, :, None] >> shifter) & 0xF
    tensor = tensor + 1
    return tensor[:, :, 0] + tensor[:, :, 1] * 16


def _scatter_gptq_moe_expert_qtensor(
    name: str,
    weight: torch.Tensor,
    params_dict: dict[str, torch.nn.Parameter],
    tp_rank_start: int,
    tp_rank_end: int,
    hidden_size: int,
    intermediate_size: int,
    checkpoint_group_size: int,
) -> str | None:
    """Real bug hit running this for real: a per-expert-indexed GPTQ
    checkpoint's `mlp.experts.<id>.{gate_up_proj,down_proj}.{qweight,scales,
    qzeros,g_idx}` tensors never matched ANY branch in `_load_weights_other`
    (only `.w13_bias`/`.w2_bias` had per-expert-index handling - see
    `_scatter_per_expert_indexed_param` above) - they fell through to the
    final `else: if name not in params_dict: continue`, silently DROPPED
    with no warning. The model's real `MoeWNA16Method`-created
    `w13_qweight`/`w13_scales`/`w13_qzeros`/`w2_qweight`/`w2_scales`/
    `w2_qzeros` parameters (see `vllm/model_executor/layers/quantization/
    moe_wna16.py`'s `create_weights`) were NEVER populated, staying at
    their `torch.empty`/`torch.zeros` init value - which is exactly why
    every MoE layer's output measured bit-for-bit `0.0` (confirmed via
    `VLLM_DEBUG_LAYER_STATS` tracing: router logits/weights were healthy,
    only the expert compute itself was zero) while `residual`
    (attention-derived) stayed real.

    Applies the SAME transformation `moe_wna16_weight_loader`'s GPTQ branch
    does (that function's own shard-splitting assumptions don't fit this
    checkpoint's already-fused-per-expert gate_up_proj/down_proj tensors,
    so this reimplements the transform directly rather than calling
    through it - see this session's investigation notes for the full
    derivation, cross-checked against `create_weights`'s parameter shapes
    tensor-by-tensor before writing this): transpose + reinterpret the
    packed int32 as uint8 (qweight/qzeros), the GPTQ zero-point +1
    adjustment (qzeros only), then TP-slice and scatter into
    `.data[expert_id]` of the real fused parameter. `.g_idx` is
    intentionally never handled (returns None) - `desc_act=false` in this
    checkpoint's `quantize_config.json` means g_idx is just `range(hidden_
    size)//group_size` and `moe_wna16_weight_loader` itself skips it too.
    """
    if ".experts." not in name or ".g_idx" in name:
        return None
    if ".qweight" not in name and ".scales" not in name and ".qzeros" not in name:
        return None
    is_w13 = ".gate_up_proj." in name
    is_w2 = ".down_proj." in name
    if not is_w13 and not is_w2:
        return None

    parts = name.split(".")
    ids = [s for s in parts if s.isdigit()]
    if len(ids) != 2:
        return None
    expert_id = int(ids[-1])

    # group_size shrinks (by a power of 2) whenever intermediate_size_per_
    # partition or hidden_size isn't evenly divisible by it - see
    # `MoeWNA16Method.create_weights`'s own while-loop; replicated here so
    # scales/qzeros' group axis matches the real per-expert parameter's
    # shape exactly instead of guessing.
    per_rank_intermediate_size = (tp_rank_end - tp_rank_start)
    group_size = checkpoint_group_size
    div_factor = 1
    while per_rank_intermediate_size % group_size or hidden_size % group_size:
        group_size //= 2
        div_factor *= 2
        assert group_size >= 32, "group_size shrank below GPTQ's minimum"

    # Search from the end (not .index()'s first-match) so a layer index
    # that happens to equal the expert index numerically (e.g. layer 3,
    # expert 3) still strips the right occurrence - same care
    # `_scatter_per_expert_indexed_param` above takes.
    idx_pos = len(parts) - 1 - parts[::-1].index(ids[-1])
    base = ".".join(parts[:idx_pos] + parts[idx_pos + 1 :])
    base = base.replace(".experts.", ".experts.routed_experts.", 1)
    # base now ends in e.g. "...gate_up_proj.qweight" - swap that suffix
    # for the real fused parameter's plain name.
    prefix = base.rsplit(".gate_up_proj." if is_w13 else ".down_proj.", 1)[0]
    w_tag = "w13" if is_w13 else "w2"
    if ".qweight" in name:
        fused_name = f"{prefix}.{w_tag}_qweight"
        transformed = weight.T.contiguous().view(torch.uint8)
        if is_w13:
            sliced = transformed[2 * tp_rank_start : 2 * tp_rank_end]
        else:
            sliced = transformed[:, tp_rank_start // 2 : tp_rank_end // 2]
    elif ".scales" in name:
        fused_name = f"{prefix}.{w_tag}_scales"
        transformed = weight.T
        if div_factor > 1:
            transformed = transformed.repeat_interleave(div_factor, 1)
        if is_w13:
            sliced = transformed[2 * tp_rank_start : 2 * tp_rank_end]
        else:
            sliced = transformed[:, tp_rank_start // group_size : tp_rank_end // group_size]
    else:  # .qzeros
        fused_name = f"{prefix}.{w_tag}_qzeros"
        # This checkpoint is symmetric quantization (quantize_config.json's
        # "sym": true) - real bug hit running this for real: the real
        # w13_qzeros/w2_qzeros parameters exist but are registered as a
        # literal empty `(0,)` placeholder in this case (`has_zp=False` in
        # `MoeWNA16Method.create_weights`, confirmed by directly
        # constructing the module on a meta device and inspecting
        # `named_parameters()`), so `.data[expert_id]` on them always
        # raises `IndexError: index 0 is out of bounds for dimension 0
        # with size 0` - the checkpoint's own qzeros tensors (present on
        # disk regardless of sym) are simply never used/needed and must
        # be skipped entirely, matching `moe_wna16_weight_loader`'s own
        # `if not layer.quant_config.has_zp and "qzeros" in weight_name:
        # return False` check.
        if fused_name in params_dict and params_dict[fused_name].numel() == 0:
            return None
        transformed = _convert_gptq_int4_qzeros(weight.view(torch.uint8)).T
        if div_factor > 1:
            transformed = transformed.repeat_interleave(div_factor, 1)
        if is_w13:
            sliced = transformed[tp_rank_start:tp_rank_end]
        else:
            sliced = transformed[:, tp_rank_start // group_size : tp_rank_end // group_size]

    if fused_name not in params_dict:
        return None
    params_dict[fused_name].data[expert_id].copy_(sliced)
    return fused_name


def _scatter_per_expert_indexed_param(
    name: str,
    weight: torch.Tensor,
    params_dict: dict[str, torch.nn.Parameter],
) -> str | None:
    """Handle a checkpoint weight name like `...mlp.experts.3.w2_bias` (a
    per-expert-indexed tensor) whose literal name - even after
    `maybe_remap_moe_expert_param_name`'s "insert routed_experts" pass -
    does not match any real parameter, because the real model registers
    ONE fused parameter per layer covering all experts (shape
    `[num_experts, ...]`), not one parameter per expert. Strips the
    numeric expert-index path segment, inserts `routed_experts` (same
    convention `maybe_remap_moe_expert_param_name` uses), and if that
    fused name exists, scatters `weight` into `.data[expert_id]` - the
    same technique `_load_weights_mxfp4` already uses for its own
    per-expert-indexed checkpoint variants (see that method's `ids =
    [s for s in parts if s.isdigit()]` extraction above). Returns the
    fused parameter name if the scatter succeeded, else None (caller
    should skip this weight - not every unmatched name is this case).
    """
    if ".experts." not in name:
        return None
    parts = name.split(".")
    ids = [s for s in parts if s.isdigit()]
    # 2 digit segments = layer index AND expert index both present (e.g.
    # "layers.5.mlp.experts.37.w2_bias") - matches `_load_weights_mxfp4`'s
    # own `ids == 2` case above. 1 segment means only the layer index is
    # present (no expert index to strip - not this function's case).
    if len(ids) != 2:
        return None
    expert_id_str = ids[-1]
    expert_id = int(expert_id_str)
    # Remove the expert-index segment specifically (searching from the end,
    # same as `_load_weights_mxfp4`, so a layer index that happens to equal
    # the expert index numerically - e.g. layer 0, expert 0 - still strips
    # the right one).
    idx_pos = len(parts) - 1 - parts[::-1].index(expert_id_str)
    fused_name = ".".join(parts[:idx_pos] + parts[idx_pos + 1 :])
    fused_name = fused_name.replace(".experts.", ".experts.routed_experts.", 1)
    if fused_name not in params_dict:
        return None
    params_dict[fused_name].data[expert_id].copy_(weight)
    return fused_name


@support_torch_compile
class GptOssModel(nn.Module, EagleModelMixin):
    # Override to swap in an alternative TransformerBlock subclass.
    block_cls: type[nn.Module] = TransformerBlock

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: self.block_cls(
                vllm_config,
                prefix=prefix,
                quant_config=self.quant_config,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                x = self.embed_input_ids(input_ids)

            residual = None
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, x, residual
        )
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            x, residual = layer(x, positions, residual)
            self._maybe_add_hidden_state(aux_hidden_states, i + 1, x, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": x, "residual": residual})
        x, _ = self.norm(x, residual)
        if os.environ.get("VLLM_DEBUG_LAYER_STATS"):
            xv = x.detach().float()
            print(f"[layer-debug] final_norm_out: shape={tuple(x.shape)} "
                  f"mean={xv.mean().item():.6g} std={xv.std().item():.6g} "
                  f"last_row_mean={xv[-1].mean().item():.6g} last_row_std={xv[-1].std().item():.6g}",
                  file=sys.stderr, flush=True)

        if len(aux_hidden_states) > 0:
            return x, aux_hidden_states
        return x

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, weight scales, activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # NOTE: this is only used for quark.
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
            num_redundant_experts=0,
        )

    def _load_weights_mxfp4(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        use_ep = self.parallel_config.enable_expert_parallel
        num_experts = self.config.num_local_experts

        # In MoE, we need to flatten the tensor parallel size across the data
        # parallel size when EP is disabled.
        tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size=get_tensor_model_parallel_world_size(),
            dp_size=get_dp_group().world_size,
            dp_rank=get_dp_group().rank_in_group,
            pcp_size=get_pcp_group().world_size,
            pcp_rank=get_pcp_group().rank_in_group,
        )

        intermediate_size = self.config.intermediate_size
        intermediate_size_block = intermediate_size // OCP_MX_BLOCK_SIZE
        per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
        per_rank_intermediate_size = (
            per_rank_intermediate_size_block * OCP_MX_BLOCK_SIZE
        )

        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        # Use centralized weight remapping for MoE expert parameters
        for name, weight in remap_moe_expert_weights(weights, params_dict):
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            if ".w13_weight_scale" in name:
                # Handle MLP gate and up projection weights scale
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_weight_scale" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[
                        ...,
                        tp_rank_start // OCP_MX_BLOCK_SIZE : tp_rank_end
                        // OCP_MX_BLOCK_SIZE,
                    ]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # flat weight from (E, 2 * N, block_size, entry_per_block)
                # to (E, 2 * N, -1), shouldn't trigger copy for contiguous
                weight = weight.view(
                    num_experts, 2 * intermediate_size, -1
                ).contiguous()

                # Extract gate and up projection parts
                # since the weight is shuffled, we can slice directly
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, ...]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                # same flatten here, but since 2 mx4 value are packed in 1
                # uint8, divide by 2
                weight = weight.view(
                    num_experts, -1, intermediate_size // 2
                ).contiguous()
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[..., tp_rank_start // 2 : tp_rank_end // 2]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    narrow_weight,
                    weight_name=name,
                    shard_id=None,
                    expert_id=None,
                )
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                weight_loader(
                    param, weight, weight_name=name, shard_id=None, expert_id=None
                )
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def _load_weights_quark(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        use_ep = self.parallel_config.enable_expert_parallel
        num_experts = self.config.num_local_experts

        if use_ep:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
        else:
            tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
                tp_size=get_tensor_model_parallel_world_size(),
                dp_size=get_dp_group().world_size,
                dp_rank=get_dp_group().rank_in_group,
                pcp_size=get_pcp_group().world_size,
                pcp_rank=get_pcp_group().rank_in_group,
            )

        def _is_mxfp4(weight_dtype: str | None) -> bool:
            """Return True for any MXFP4 weight-dtype variant.

            Covers "gpt_oss_mxfp4" (GptOssMxfp4MoEMethod) and "mxfp4"
            (QuarkMoEMethod with fp4 weights) and any future variants.
            """
            return weight_dtype is not None and "mxfp4" in weight_dtype

        def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
            """Helper function to get MoE quantization weight dtype.

            Args:
                layer_id: Layer index to check (default 0, as all layers should
                        have the same quantization method)

            Returns:
                Weight dtype string (e.g., "mxfp4", "fp8") or None if not available
            """
            if hasattr(self.layers[layer_id].mlp.experts._quant_method, "weight_dtype"):
                return self.layers[layer_id].mlp.experts._quant_method.weight_dtype
            return None

        intermediate_size = self.config.intermediate_size

        moe_weight_dtype = _get_moe_weight_dtype(layer_id=0)

        if _is_mxfp4(moe_weight_dtype):
            # MXFP4 requires OCP_MX_BLOCK_SIZE alignment
            intermediate_size_block = intermediate_size // OCP_MX_BLOCK_SIZE
            per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
            per_rank_intermediate_size = (
                per_rank_intermediate_size_block * OCP_MX_BLOCK_SIZE
            )
        else:
            # FP8 and other formats don't need alignment
            per_rank_intermediate_size = cdiv(intermediate_size, tp_size)

        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue

            layer_id, expert_id, fused_name = None, None, None
            moe_quant_method = None
            if "experts" in name:
                parts = name.split(".")
                ids = [s for s in parts if s.isdigit()]

                # for amd-quark format that each expert is separated
                # need to extract the parameter name with experts fused.
                # example model: amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8
                if len(ids) == 2:
                    layer_id, expert_id = int(ids[0]), int(ids[-1])
                    parts.pop(len(parts) - 1 - parts[::-1].index(str(expert_id)))
                    fused_name = ".".join(parts)

                # for openai mxfp4 format that all experts are combined
                # no need to extract the parameter name with experts fused.
                # models: openai/gpt-oss-20b, openai/gpt-oss-120b
                elif len(ids) == 1:
                    layer_id, expert_id = int(ids[0]), None
                    fused_name = name

                else:
                    raise NameError(
                        f"Layer {name} contains more than 2 numeric indices. This is "
                        "an unexpected condition. Please open an issue if encountered."
                    )

                # The MoE refactor (#41184) moved expert params under
                # `mlp.experts.routed_experts.*`; remap the legacy checkpoint
                # name so keys like w2_bias resolve against params_dict.
                fused_name = fused_name.replace(
                    ".mlp.experts.", ".mlp.experts.routed_experts."
                )

                moe_quant_method = _get_moe_weight_dtype(layer_id=layer_id)

            if (
                all(key in name for key in ["input_scale", "mlp.experts"])
                and expert_id is not None
            ):
                assert loaded_weight.numel() == 1
                expert_data = params_dict[fused_name].data[expert_id]
                expert_data.copy_(loaded_weight)
                loaded_params.add(fused_name)
                continue

            # Unified handler for mxfp4 weights and scales
            elif _is_mxfp4(moe_quant_method) and any(
                name.endswith(suffix)
                for suffix in [
                    ".w13_weight_scale",
                    ".w2_weight_scale",
                    ".w13_weight",
                    ".w2_weight",
                ]
            ):
                is_w13 = ".w13_" in name
                is_scale = "_scale" in name

                # Reshape weight for mxfp4 if needed (not for scales)
                if not is_scale and expert_id is None:
                    if is_w13:
                        if loaded_weight.dim() < 3:
                            raise ValueError(
                                f"Expected w13_weight to have at least 3 "
                                f"dimensions, got shape "
                                f"{loaded_weight.shape}"
                            )
                        if loaded_weight.shape[0] != num_experts:
                            raise ValueError(
                                f"Expected w13_weight first dimension to be "
                                f"{num_experts}, got "
                                f"{loaded_weight.shape[0]}"
                            )
                        loaded_weight = loaded_weight.view(
                            num_experts, 2 * intermediate_size, -1
                        ).contiguous()
                    else:
                        if loaded_weight.dim() < 3:
                            raise ValueError(
                                f"Expected w2_weight to have at least 3 "
                                f"dimensions, got shape "
                                f"{loaded_weight.shape}"
                            )
                        if loaded_weight.shape[0] != num_experts:
                            raise ValueError(
                                f"Expected w2_weight first dimension to be "
                                f"{num_experts}, got "
                                f"{loaded_weight.shape[0]}"
                            )
                        loaded_weight = loaded_weight.view(
                            num_experts, -1, intermediate_size // 2
                        ).contiguous()

                if use_ep:
                    sliced_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    if is_w13:
                        if expert_id is None:
                            sliced_weight = loaded_weight[
                                :, 2 * tp_rank_start : 2 * tp_rank_end, ...
                            ]
                        else:
                            sliced_weight = loaded_weight[
                                2 * tp_rank_start : 2 * tp_rank_end, ...
                            ]
                    else:
                        if is_scale:
                            sliced_weight = loaded_weight[
                                ...,
                                tp_rank_start // OCP_MX_BLOCK_SIZE : tp_rank_end
                                // OCP_MX_BLOCK_SIZE,
                            ]
                        else:
                            sliced_weight = loaded_weight[
                                ..., tp_rank_start // 2 : tp_rank_end // 2
                            ]

                # NOTE(rob): because gpt-oss ckpt has "unique" structure with
                # fused gate_up_proj fused on disk, we cannot use the existing
                # weight loaders without added complexity, so just do the
                # direct load here.
                param = params_dict[fused_name]
                expert_data = param.data[expert_id]
                dim1 = sliced_weight.shape[0]
                dim2 = sliced_weight.shape[1]
                expert_data.data[:dim1, :dim2].copy_(sliced_weight)
                loaded_params.add(fused_name)
                continue

            elif name.endswith(".w13_weight") and moe_quant_method == "fp8":
                if use_ep:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    if expert_id is None:
                        narrow_weight = loaded_weight[
                            :, 2 * tp_rank_start : 2 * tp_rank_end, :
                        ]
                    else:
                        narrow_weight = loaded_weight[
                            2 * tp_rank_start : 2 * tp_rank_end, :
                        ]

                assert fused_name is not None
                param = params_dict[fused_name]

                if expert_id is None:
                    param.data.copy_(narrow_weight)
                else:
                    param.data[expert_id].copy_(narrow_weight)

                loaded_params.add(fused_name)
                continue

            elif name.endswith(".w13_weight_scale") and moe_quant_method == "fp8":
                assert fused_name is not None
                param = params_dict[fused_name]

                # Check if this is per-channel or per-tensor scale
                if loaded_weight.numel() > 1 and loaded_weight.dim() == 1:
                    if use_ep:
                        narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                    else:
                        narrow_weight = loaded_weight[
                            2 * tp_rank_start : 2 * tp_rank_end
                        ]
                else:
                    narrow_weight = loaded_weight

                if expert_id is None:
                    param.data.copy_(narrow_weight)
                else:
                    param.data[expert_id].copy_(narrow_weight)

                loaded_params.add(fused_name)
                continue

            elif name.endswith(".w13_input_scale") and moe_quant_method == "fp8":
                assert fused_name is not None
                param = params_dict[fused_name]

                if expert_id is None:
                    param.data.copy_(loaded_weight)
                else:
                    param.data[expert_id].copy_(loaded_weight)

                loaded_params.add(fused_name)
                continue

            elif name.endswith(".w2_weight") and moe_quant_method == "fp8":
                if use_ep:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    if expert_id is None:
                        narrow_weight = loaded_weight[..., tp_rank_start:tp_rank_end]
                    else:
                        narrow_weight = loaded_weight[..., tp_rank_start:tp_rank_end]

                assert fused_name is not None
                param = params_dict[fused_name]

                if expert_id is None:
                    param.data.copy_(narrow_weight)
                else:
                    param.data[expert_id].copy_(narrow_weight)

                loaded_params.add(fused_name)
                continue

            elif name.endswith(".w2_weight_scale") and moe_quant_method == "fp8":
                assert fused_name is not None
                param = params_dict[fused_name]

                if use_ep:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = loaded_weight

                if expert_id is None:
                    param.data.copy_(narrow_weight)
                else:
                    param.data[expert_id].copy_(narrow_weight)

                loaded_params.add(fused_name)
                continue

            # Unified handler for bias loading (w13_bias and w2_bias)
            elif name.endswith(".w13_bias") or name.endswith(".w2_bias"):
                is_w13_bias = name.endswith(".w13_bias")

                if use_ep:
                    sliced_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
                else:
                    if is_w13_bias:
                        if expert_id is None:
                            sliced_weight = loaded_weight[
                                :, 2 * tp_rank_start : 2 * tp_rank_end
                            ]
                        else:
                            sliced_weight = loaded_weight[
                                2 * tp_rank_start : 2 * tp_rank_end
                            ]
                    else:
                        sliced_weight = loaded_weight
                        if tp_rank != 0:
                            sliced_weight = sliced_weight.zero_()

                # NOTE(rob): because gpt-oss ckpt has "unique" structure with
                # fused gate_up_proj fused on disk, we cannot use the existing
                # weight loaders without added complexity, so just do the
                # direct load here.
                assert fused_name is not None
                param = params_dict[fused_name]
                expert_data = param.data[expert_id]
                dim1 = sliced_weight.shape[0]
                expert_data.data[:dim1].copy_(sliced_weight)
                loaded_params.add(fused_name)
                continue

            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                param = params_dict[name]
                weight_loader = param.weight_loader

                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    param_name, weight_name, mapping_expert_id, shard_id = mapping
                    weight_name = (
                        weight_name[:-1] if weight_name.endswith(".") else weight_name
                    )

                    if weight_name not in name:
                        continue

                    param = params_dict[fused_name]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    # Use checkpoint's expert_id for quark format (when expert_id
                    # is extracted from weight name), otherwise use mapping's expert_id
                    actual_expert_id = (
                        expert_id if expert_id is not None else mapping_expert_id
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        fused_name,
                        shard_id=shard_id,
                        expert_id=actual_expert_id,
                        return_success=True,
                    )
                    if success:
                        name = fused_name
                        loaded_params.add(name)
                        break
                else:
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

                loaded_params.add(name)
        return loaded_params

    def _load_weights_other(
        self,
        ep_rank_end: int,
        ep_rank_start: int,
        heads_per_rank: int,
        head_start: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params_mapping: list[tuple[str, ...]],
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        use_ep = self.parallel_config.enable_expert_parallel

        # In MoE, we need to flatten the tensor parallel size across the data
        # parallel size when EP is disabled.
        tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
            tp_size=get_tensor_model_parallel_world_size(),
            dp_size=get_dp_group().world_size,
            dp_rank=get_dp_group().rank_in_group,
            pcp_size=get_pcp_group().world_size,
            pcp_rank=get_pcp_group().rank_in_group,
        )

        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = cdiv(intermediate_size, tp_size)
        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        # Use centralized weight remapping for MoE expert parameters.
        # The MoERunner refactor moved expert params under
        # `mlp.experts.routed_experts.*`; this remaps checkpoint names so
        # MoE weight/bias keys resolve against params_dict.
        for name, weight in remap_moe_expert_weights(weights, params_dict):
            # Skip layers on other devices.
            if is_pp_missing_parameter(name, self):
                continue

            if ".w13_weight" in name:
                # Handle MLP gate and up projection weights
                # Extract gate and up projection parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, :, 2 * tp_rank_start : 2 * tp_rank_end]

                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_weight" in name:
                # Handle MLP down projection weights
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
                narrow_weight = narrow_weight.permute(0, 2, 1).contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w13_bias" in name:
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                if use_ep:
                    narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
                elif weight.dim() == 1:
                    # A per-expert-indexed checkpoint's raw tensor for one
                    # expert's bias has no leading expert dimension (already
                    # just [2*intermediate_size]) - real bug hit running a
                    # real per-expert-indexed GPTQ checkpoint here (`IndexError:
                    # too many indices for tensor of dimension 1` from the
                    # `weight[:, ...]` form below, which assumes a fused
                    # [num_experts, 2*intermediate_size] tensor).
                    narrow_weight = weight[2 * tp_rank_start : 2 * tp_rank_end]
                else:
                    narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]

                # `remap_moe_expert_weights` only inserts "routed_experts."
                # into the path - it does not strip a per-expert numeric
                # index (e.g. "mlp.experts.3.w13_bias"), so a per-expert-
                # indexed GPTQ checkpoint's bias name never matches a real
                # (fused, one-tensor-per-layer) parameter and falls through
                # unchanged. `_load_weights_mxfp4` (above) already handles
                # this exact case for its own checkpoint formats via
                # expert-id extraction + `.data[expert_id]` scatter; real
                # bug hit running a real per-expert-indexed GPTQ checkpoint
                # (KeyError: 'layers.0.mlp.experts.0.w2_bias') showed
                # `_load_weights_other` never got the equivalent fix.
                if name not in params_dict:
                    scattered_name = _scatter_per_expert_indexed_param(
                        name, narrow_weight, params_dict
                    )
                    if scattered_name is not None:
                        loaded_params.add(scattered_name)
                    continue
                param = params_dict[name]
                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif ".w2_bias" in name:
                # Handle MLP down projection bias
                if use_ep:
                    weight = weight[ep_rank_start:ep_rank_end, ...]
                else:
                    # (only load on rank 0 to avoid duplication)
                    if tp_rank != 0:
                        weight.zero_()
                # See the `.w13_bias` branch above for why this fallback is
                # needed - same per-expert-indexed-checkpoint gap.
                if name not in params_dict:
                    scattered_name = _scatter_per_expert_indexed_param(
                        name, weight, params_dict
                    )
                    if scattered_name is not None:
                        loaded_params.add(scattered_name)
                    continue
                param = params_dict[name]
                param.copy_(weight)
                loaded_params.add(name)
                continue
            elif ".experts." in name and (
                ".qweight" in name or ".scales" in name or ".qzeros" in name or ".g_idx" in name
            ):
                # Real bug hit running this for real (second time): the
                # first version of this branch matched ANY name containing
                # these suffixes, including attention's OWN GPTQ-quantized
                # qkv_proj/o_proj tensors (this whole checkpoint is GPTQ,
                # not just the MoE layers) - the unconditional `continue`
                # below then skipped them past the `stacked_params_mapping`
                # loop that actually loads them, leaving attention's real
                # weights uninitialized and producing NaN attn_out (caught
                # via VLLM_DEBUG_LAYER_STATS - MoE finally loaded correctly
                # this run, but attn_out went NaN instead). Restricting to
                # `.experts.` names only (matching this function's own
                # internal check) lets non-expert quantized tensors fall
                # through to their correct existing handler below.
                # Real bug hit running this for real: these raw GPTQ
                # quantization tensors (per-expert-indexed, e.g.
                # "...experts.3.gate_up_proj.qweight") never matched any
                # branch above and fell through to the generic "else"
                # below, whose `if name not in params_dict: continue`
                # silently dropped them with no warning at all - the
                # model's real w13_qweight/w13_scales/w13_qzeros/
                # w2_qweight/w2_scales/w2_qzeros parameters (see
                # moe_wna16.py's create_weights) were NEVER populated,
                # explaining why every MoE layer's output measured exactly
                # 0.0 (confirmed via VLLM_DEBUG_LAYER_STATS: router logits/
                # weights were healthy, only expert compute was zero).
                # See _scatter_gptq_moe_expert_qtensor's own docstring for
                # the full transformation derivation.
                scattered_name = _scatter_gptq_moe_expert_qtensor(
                    name, weight, params_dict, tp_rank_start, tp_rank_end,
                    hidden_size=self.config.hidden_size,
                    intermediate_size=self.config.intermediate_size,
                    checkpoint_group_size=self.quant_config.group_size,
                )
                if scattered_name is not None:
                    loaded_params.add(scattered_name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank_in_group
        num_experts = self.config.num_local_experts
        experts_per_rank = num_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank

        quant_method = (
            self.config.quantization_config["quant_method"]
            if hasattr(self.config, "quantization_config")
            else None
        )
        # Normalize the checkpoint's quant_method to the internal name.
        # Note: there are three places where "mxfp4" -> "gpt_oss_mxfp4"
        # normalization occurs, each serving a different data path:
        #   1. GptOssMxfp4Config.override_quantization_method() — sets
        #      ModelConfig.quantization (used to select the QuantizationConfig
        #      class at model init time), reading from model_arch_config which
        #      is a snapshot taken before verify_and_update_model_config runs.
        #   2. GptOssForCausalLMConfig.verify_and_update_model_config() —
        #      patches hf_config.quantization_config in-place (a separate copy
        #      of the dict from model_arch_config) for later hf_config lookups.
        #   3. Here — reads directly from self.config (the raw HF config) which
        #      may still carry the original "mxfp4" string from the checkpoint.
        if quant_method == "mxfp4":
            quant_method = "gpt_oss_mxfp4"

        if quant_method == "gpt_oss_mxfp4":
            return self._load_weights_mxfp4(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )
        elif quant_method == "quark":
            return self._load_weights_quark(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )
        else:
            return self._load_weights_other(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )


class GptOssForCausalLM(
    nn.Module, SupportsPP, SupportsEagle, SupportsEagle3, SupportsLoRA
):
    is_3d_moe_weight: bool = True
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".self_attn.": ".attn.",
        },
        orig_to_new_suffix={
            ".embed_tokens.weight": ".embedding.weight",
            # MoE MXFP4 weights
            ".gate_up_proj_blocks": ".w13_weight",
            ".down_proj_blocks": ".w2_weight",
            ".gate_up_proj_scales": ".w13_weight_scale",
            ".down_proj_scales": ".w2_weight_scale",
            # MoE other weights
            ".gate_up_proj": ".w13_weight",
            ".down_proj": ".w2_weight",
            # MoE Bias
            ".gate_up_proj_bias": ".w13_bias",
            ".down_proj_bias": ".w2_bias",
            # For quark format
            ".gate_up_proj.weight": ".w13_weight",
            ".gate_up_proj.weight_scale": ".w13_weight_scale",
            ".gate_up_proj.bias": ".w13_bias",
            ".gate_up_proj.input_scale": ".w13_input_scale",
            ".down_proj.weight": ".w2_weight",
            ".down_proj.weight_scale": ".w2_weight_scale",
            ".down_proj.bias": ".w2_bias",
            ".down_proj.input_scale": ".w2_input_scale",
        },
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.model = GptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if os.environ.get("VLLM_DEBUG_LAYER_STATS"):
            hv = hidden_states.detach().float()
            print(f"[layer-debug] compute_logits input: shape={tuple(hidden_states.shape)} "
                  f"mean={hv.mean().item():.6g} std={hv.std().item():.6g} "
                  f"isnan={bool(torch.isnan(hv).any())} isinf={bool(torch.isinf(hv).any())}",
                  file=sys.stderr, flush=True)
        logits = self.logits_processor(self.lm_head, hidden_states)
        if os.environ.get("VLLM_DEBUG_LAYER_STATS"):
            lv = logits.detach().float()
            top = lv[-1].topk(5)
            print(f"[layer-debug] logits: shape={tuple(logits.shape)} "
                  f"mean={lv.mean().item():.6g} std={lv.std().item():.6g} "
                  f"last_row_top5_idx={top.indices.tolist()} last_row_top5_val={[round(v,3) for v in top.values.tolist()]}",
                  file=sys.stderr, flush=True)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
