"""Runtime patch: makes Qwen3.5's native MTP (multi-token prediction /
speculative decoding) drafter work under this project's synthetic,
transport-backed multi-machine PP group.

Real bug found running this for real: `Qwen3_5MultiTokenPredictor.forward()`
(vllm/model_executor/models/qwen3_5_mtp.py) branches on
`get_pp_group().is_first_rank` to decide whether to build its input from
`hidden_states`/`inputs_embeds` directly, or from `intermediate_tensors`
handed off by an earlier PP rank:

    if get_pp_group().is_first_rank:
        ...build from hidden_states/inputs_embeds...
    else:
        assert intermediate_tensors is not None
        ...

This is correct for real vLLM pipeline parallelism, where the MTP module
is split across the SAME PP ranks as the main model and genuinely receives
intermediate_tensors from an earlier rank when it isn't first. It is wrong
for this project: the MTP module is NEVER split across our machines - the
full `mtp.*` weight set lives entirely on whichever single machine is the
driver/last stage (see humming_fix/single_layer_probe/
extract_stage_checkpoint_qwen35.py's --include-mtp), and it always
consumes the TARGET model's freshly-computed final hidden_states directly,
same process, same GPU. But `get_pp_group()` returns this project's real,
global synthetic PP group (e.g. pp_rank=3 of pp_world_size=4 for a 4-stage
deployment) - correct for the main model's own layer partitioning, but
irrelevant to (and wrong for) the MTP module's own, always-local, never-
split existence. `is_first_rank` reports False on any non-first-machine
driver, so the drafter takes the `else` branch and asserts on
`intermediate_tensors`, which is genuinely `None` here (nothing upstream
ever populates it - there is no earlier "MTP rank" in this design) ->
`AssertionError` inside `Worker.determine_available_memory()`'s
`profile_run()` -> `self.drafter.dummy_run()`, crashing the driver during
warmup, before any real request is ever served.

The fix: monkeypatch `Qwen3_5MultiTokenPredictor.forward` to always take
the "first rank" branch - identical to the original method body's `if`
arm, every time - since that is the ONLY branch that can ever be correct
for a checkpoint whose MTP weights are entirely local to one machine.
`is_last_rank` (used elsewhere in the same file to decide whether to
build a real `lm_head` vs a `PPMissingLayer`) is left completely alone:
it correctly resolves True for the real driver machine and is not part of
this bug.

Usage: `import vllm.transport.qwen35_mtp_pp_fix` before the MTP model is
constructed (i.e. before `--speculative-config` triggers vLLM to build
the drafter) - same load-order requirement as humming_fix/patch.py.
"""
import torch

from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
from vllm.sequence import IntermediateTensors


def _patched_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: "IntermediateTensors | None" = None,
    inputs_embeds: "torch.Tensor | None" = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    from vllm.distributed.parallel_state import get_pp_group

    # Always take the real forward's is_first_rank=True branch - see
    # module docstring for why this is the only correct behavior here.
    if inputs_embeds is None:
        inputs_embeds = self.embed_input_ids(input_ids)
    assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
    inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
    hidden_states = self.pre_fc_norm_hidden(hidden_states)
    hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
    hidden_states = self.fc(hidden_states)
    residual = None

    current_step_idx = spec_step_idx % self.num_mtp_layers
    mtp_layer = self.layers[current_step_idx]
    hidden_states, residual = mtp_layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    if mtp_layer.use_attn_reduce_scatter_for_moe:
        from vllm.model_executor.models.qwen3_next import (
            _all_gather_hidden_and_residual,
        )

        hidden_states, residual = _all_gather_hidden_and_residual(
            hidden_states,
            residual,
            positions.shape[-1],
            self.config.hidden_size,
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


Qwen3_5MultiTokenPredictor.forward = _patched_forward
