"""Makes the REAL `vllm.distributed.parallel_state.GroupCoordinator` (the
same class vLLM's own pipeline-parallel code and its own test suite -
`tests/distributed/test_comm_ops.py::send_recv_test_worker` - use for
single-tensor P2P) importable and constructable in this sandbox, without
`torch.distributed.init_process_group`, NCCL, CUDA, or Ray.

Two independent obstacles, two independent fixes:

1. Importing `vllm.distributed.parallel_state` at all currently fails in
   this sandbox (torch==2.6.0+cu124) - NOT because of anything in this
   project's changes, but because of a module-level
   `direct_register_custom_op(op_name="patched_fused_scaled_matmul_reduce_scatter", ...)`
   call (parallel_state.py:373) whose op signature isn't representable by
   this torch version's `infer_schema`. The surrounding code even has its
   own TODO acknowledging this:
       "TODO: Remove this once the pytorch fix
       (https://github.com/pytorch/pytorch/pull/165086) gets released,
       in either 2.9.1 or 2.10"
   i.e. this is a known, dated, torch-version-specific incompatibility on
   vLLM's main branch, unrelated to communication/transport and out of
   scope to "fix" for real (see vllm/transport/README.md). Monkeypatching
   `direct_register_custom_op` to a no-op *before* that import - entirely
   within this test file, zero vLLM source touched - unblocks the import
   with no effect on the actual GroupCoordinator class we care about,
   since we never call the ops it would have registered.

2. `GroupCoordinator.__init__` unconditionally calls
   `torch.distributed.get_rank()` and `torch.distributed.new_group(...)`
   (parallel_state.py:423,456) - i.e. constructing one the normal way
   *requires* torch.distributed to already be initialized, which this
   phase's constraints forbid. `GroupCoordinator.send`/`.recv` (the methods
   this phase patched to add the `self.transport` branch) don't touch any
   other instance attribute before that branch returns, so a bare instance
   built via `object.__new__` - skipping `__init__` entirely - with only
   `.transport` set is enough to call the real, patched methods.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `vllm` importable from a source checkout

import vllm.utils.torch_utils as _torch_utils  # noqa: E402

_torch_utils.direct_register_custom_op = lambda *args, **kwargs: None  # see module docstring, item 1

from vllm.distributed.parallel_state import GroupCoordinator  # noqa: E402
from vllm.transport.base import Transport  # noqa: E402


def make_stage_coordinator(transport: Transport) -> GroupCoordinator:
    """A GroupCoordinator instance whose .send()/.recv() route through
    `transport` - see module docstring, item 2 for why `object.__new__` is
    used instead of the normal constructor."""
    coord = object.__new__(GroupCoordinator)
    coord.transport = transport
    return coord
