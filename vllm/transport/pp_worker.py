"""The real (non-test) integration point: a `Worker` subclass that installs
the transport-backed PP group during the standard vLLM bootstrap, via
vLLM's own `--worker-cls` extension point (`vllm/v1/worker/worker_base.py`,
`WorkerWrapperBase.init_worker` -> `resolve_obj_by_qualname`). No changes
to `vllm/v1/worker/gpu_worker.py` are needed or made - this subclasses the
real `Worker` and only adds one step to `init_device()`.

Why `init_device()` specifically: it's the exact point, in the real
non-test bootstrap, where `init_worker_distributed_environment()` (which
calls the real `init_distributed_environment()` +
`ensure_model_parallel_initialized()`, forming the local TP/DP group and
an initially-trivial local `_PP`) has just finished, and it runs strictly
before `Worker.load_model()` (a separate executor RPC step) constructs
the actual model - so swapping `_PP` here, after `super().init_device()`,
satisfies both halves of the ordering requirement: after local
distributed init, before any pipeline communication or model construction
(`make_layers()` reads the live `_PP` group via `get_pp_group()` at
construction time, not a cached copy - see
`vllm/model_executor/models/utils.py`).

Selected via CLI, not import-time wiring:

    vllm serve $MODEL_PATH --worker-cls vllm.transport.pp_worker.TransportPPWorker ...

Configuration is threaded through environment variables (set by
`scripts/launch_pp_stage.py` before the engine is constructed) rather
than new CLI flags, because `--worker-cls` only gives you a class path -
there is no supported way to pass extra constructor arguments to it, and
every worker process needs to read this identically after being spawned.
Env vars set in the parent process before engine construction are
inherited by every spawned worker process (`vllm/v1/executor/
multiproc_executor.py`'s `update_environment_variables` only overwrites
the specific keys vLLM itself manages - it does not replace or clear the
environment), so this is a normal, reliable way to pass fixed
per-launch configuration to code that only vLLM's own machinery
constructs.
"""
from __future__ import annotations

import os
import time

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

# Applies the Qwen3.5 MTP + synthetic-PP compatibility patch (see that
# module's docstring) unconditionally on import. This module is the right
# place for it (rather than relying on PYTHONPATH/sitecustomize, which was
# tried first and doesn't work here): `multiprocessing`'s `spawn` start
# method - forced on by vLLM whenever CUDA is involved - clones the
# *parent's already-computed* `sys.path` into each worker child instead of
# re-deriving it from `PYTHONPATH` in the child's own fresh interpreter, so
# mutating `os.environ["PYTHONPATH"]` at runtime in the parent never
# reaches the workers (confirmed by direct reproduction: a spawned child's
# `sys.path` omits any directory added to `PYTHONPATH` after the parent
# interpreter itself started). This module, by contrast, is freshly
# imported once per worker process (every worker resolves `--worker-cls`
# via `resolve_obj_by_qualname`, re-importing this file), making it a
# reliable per-process hook regardless of spawn/fork or PYTHONPATH.
import vllm.transport.qwen35_mtp_pp_fix  # noqa: F401,E402

logger = init_logger(__name__)

_ENV_PP_RANK = "VLLM_TRANSPORT_PP_RANK"
_ENV_PP_WORLD_SIZE = "VLLM_TRANSPORT_PP_WORLD_SIZE"
_ENV_SELF_NAME = "VLLM_TRANSPORT_SELF_NAME"
_ENV_PREV_NAME = "VLLM_TRANSPORT_PREV_NAME"
_ENV_NEXT_NAME = "VLLM_TRANSPORT_NEXT_NAME"
_ENV_SIGNALING_URL = "VLLM_TRANSPORT_SIGNALING_URL"
_ENV_UDP_PORT_BASE = "VLLM_TRANSPORT_UDP_PORT_BASE"
_ENV_TCP_PORT_BASE = "VLLM_TRANSPORT_TCP_PORT_BASE"
_ENV_TCP_CONNECT_HOST_PREV = "VLLM_TRANSPORT_TCP_CONNECT_HOST_PREV"
_ENV_TCP_CONNECT_HOST_NEXT = "VLLM_TRANSPORT_TCP_CONNECT_HOST_NEXT"
_ENV_CONNECT_TIMEOUT = "VLLM_TRANSPORT_CONNECT_TIMEOUT"


class TransportPPWorker(Worker):
    """Drop-in replacement for `vllm.v1.worker.gpu_worker.Worker` that
    replaces the pipeline-parallel dimension with this project's
    transport-backed `GroupCoordinator` right after local (real,
    unmodified, NCCL-backed) tensor-parallel bootstrap completes.

    Everything else (load_model, execute_model, determine_available_memory,
    ...) is inherited from the real `Worker` unmodified - this class exists
    to add exactly one step, at exactly one point, in the bootstrap
    sequence.
    """

    def init_device(self) -> None:
        super().init_device()

        pp_rank = int(os.environ[_ENV_PP_RANK])
        pp_world_size = int(os.environ[_ENV_PP_WORLD_SIZE])

        # A genuinely standalone deployment (one machine, no PP peers at
        # all - e.g. a model small enough to serve without splitting)
        # has no cross-machine dimension to replace. install_transport_pp_group()
        # requires at least one real peer transport by design (see this
        # module's own docstring: it exists ONLY to replace the
        # cross-machine dimension, never to interfere with the local
        # one) - calling it here would always raise. vLLM's own local
        # `_PP` group from super().init_device() above is already
        # correct and complete for pp_world_size==1, so there's nothing
        # for this class to do beyond what Worker already did.
        if pp_world_size <= 1:
            return

        self_name = os.environ[_ENV_SELF_NAME]
        prev_name = os.environ.get(_ENV_PREV_NAME) or None
        next_name = os.environ.get(_ENV_NEXT_NAME) or None
        signaling_url = os.environ.get(_ENV_SIGNALING_URL) or None
        udp_port_base = int(os.environ.get(_ENV_UDP_PORT_BASE, "30000"))
        tcp_port_base = int(os.environ.get(_ENV_TCP_PORT_BASE, "30000"))
        tcp_connect_host_prev = os.environ.get(_ENV_TCP_CONNECT_HOST_PREV) or None
        tcp_connect_host_next = os.environ.get(_ENV_TCP_CONNECT_HOST_NEXT) or None
        connect_timeout = float(os.environ.get(_ENV_CONNECT_TIMEOUT, "120"))

        logger.info(
            "TransportPPWorker: local_rank=%s establishing pp_rank=%s/%s "
            "links (self=%s prev=%s next=%s)",
            self.local_rank,
            pp_rank,
            pp_world_size,
            self_name,
            prev_name,
            next_name,
        )

        from vllm.transport.pipeline_bootstrap import (
            establish_pp_transports,
            install_transport_pp_group,
        )

        transport_prev, transport_next = establish_pp_transports(
            pp_rank=pp_rank,
            pp_world_size=pp_world_size,
            local_rank=self.local_rank,
            self_name=self_name,
            prev_name=prev_name,
            next_name=next_name,
            signaling_url=signaling_url,
            udp_port_base=udp_port_base,
            tcp_port_base=tcp_port_base,
            tcp_connect_host_prev=tcp_connect_host_prev,
            tcp_connect_host_next=tcp_connect_host_next,
            connect_timeout=connect_timeout,
        )

        install_transport_pp_group(
            pp_rank=pp_rank,
            pp_world_size=pp_world_size,
            local_rank=self.local_rank,
            transport_prev=transport_prev,
            transport_next=transport_next,
        )

        # Real bug hit running this for real (first seen on a PP=2 Qwen2.5-7B
        # split - the vllm checkout picked up a newer worker/model_runner
        # refactor since this project's earlier GPT-OSS/PP=3 runs): the "V2"
        # GPUModelRunner (vllm/v1/worker/gpu/model_runner.py) caches
        # `is_first_pp_rank`/`is_last_pp_rank` from `get_pp_group()` in its
        # own __init__ - which super().init_device() above already ran,
        # against the trivial single-rank `_PP` group that existed before
        # install_transport_pp_group() just replaced it. Left unpatched,
        # every non-last rank's cached `is_last_pp_rank` stays True, so
        # `execute_model` asserts `isinstance(model_output, torch.Tensor)`
        # on a rank whose model.forward() (correctly reading the LIVE swapped
        # group) actually returned `IntermediateTensors` -
        # `AssertionError` during the very first profile_run. Refresh the
        # cached flags in place; harmless no-op if some future vLLM version
        # stops caching them (attribute simply won't exist).
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            if hasattr(model_runner, "is_first_pp_rank"):
                model_runner.is_first_pp_rank = pp_rank == 0
            if hasattr(model_runner, "is_last_pp_rank"):
                model_runner.is_last_pp_rank = pp_rank == pp_world_size - 1
            # NOTE: tried deleting `model_runner.drafter` here on non-last
            # ranks to save memory (GPUModelRunner.__init__(), run inside
            # super().init_device() above, constructs it against the
            # trivial pre-swap `_PP` group, which is always "last rank" for
            # a lone rank - so every stage ends up with one). Reverted: it's
            # NOT actually dead weight - plenty of other GPUModelRunner
            # methods (execute_model, initialize_kv_cache, dummy_run, ...)
            # reference `self.drafter` unconditionally whenever
            # `self.speculative_config` is set, with no is_last_rank guard,
            # so a non-last rank with speculative_config set (required on
            # every stage - see docs/DEPLOYMENT.md's MTP section, fix 2)
            # genuinely needs a real `self.drafter` object to exist. Deleting
            # it crashed real generation with `AttributeError: 'GPUModelRunner'
            # object has no attribute 'drafter'` the first time this was
            # tried for real. If per-stage memory is tight on an asymmetric
            # split, reduce --num-gpu-blocks-override for that stage instead.

        logger.info(
            "TransportPPWorker: local_rank=%s pp_rank=%s/%s transport PP "
            "group installed and connected",
            self.local_rank,
            pp_rank,
            pp_world_size,
        )
        self._transport_timing_self_name = self_name

    def execute_model(self, scheduler_output):
        # Real generation-throughput investigation (2026-08-12, revised
        # 2026-08-16): originally just logged total duration and left
        # compute time to be derived later by subtracting separately-
        # aggregated TRANSPORT_TIMING stats (e.g. median(total) -
        # median(send)) - found invalid afterward, since those aggregates
        # weren't computed from the same paired steps (each stat was its
        # own independent aggregation pass, not matched 1:1 to a specific
        # step). Fixed by reading vllm.distributed.parallel_state's
        # per-step send/recv tensor-dict accumulator (reset right before
        # this call, read right after) so recv_ms/send_ms/compute_ms in
        # a single log line are guaranteed to come from the SAME step as
        # total_ms - properly isolated, not derived from aggregates.
        debug_timing = os.environ.get("VLLM_TRANSPORT_DEBUG_TIMING")
        if debug_timing:
            from vllm.distributed.parallel_state import (
                get_transport_tensor_dict_timing,
                reset_transport_tensor_dict_timing,
            )
            reset_transport_tensor_dict_timing()
        _t0 = time.monotonic()
        result = super().execute_model(scheduler_output)
        total_ms = (time.monotonic() - _t0) * 1000
        self_name = getattr(self, "_transport_timing_self_name", "?")
        if debug_timing:
            timing = get_transport_tensor_dict_timing()
            recv_total_ms = (
                timing["recv_network_ms"] + timing["recv_unpickle_ms"]
                + timing["recv_deserialize_ms"] + timing["recv_h2d_copy_ms"]
            )
            send_total_ms = (
                timing["send_cuda_sync_ms"] + timing["send_split_pickle_ms"]
                + timing["send_d2h_copy_ms"] + timing["send_serialize_ms"]
                + timing["send_network_ms"]
            )
            compute_ms = total_ms - recv_total_ms - send_total_ms
            logger.info(
                "[STEP_BREAKDOWN] self=%s total_ms=%.2f "
                "recv_network_ms=%.2f recv_unpickle_ms=%.2f "
                "recv_deserialize_ms=%.2f recv_h2d_copy_ms=%.2f "
                "compute_ms=%.2f "
                "send_cuda_sync_ms=%.2f send_split_pickle_ms=%.2f "
                "send_d2h_copy_ms=%.2f send_serialize_ms=%.2f "
                "send_network_ms=%.2f",
                self_name, total_ms,
                timing["recv_network_ms"], timing["recv_unpickle_ms"],
                timing["recv_deserialize_ms"], timing["recv_h2d_copy_ms"],
                compute_ms,
                timing["send_cuda_sync_ms"], timing["send_split_pickle_ms"],
                timing["send_d2h_copy_ms"], timing["send_serialize_ms"],
                timing["send_network_ms"],
            )
        else:
            logger.info(
                "[EXECUTE_MODEL_TIMING] self=%s duration_ms=%.2f",
                self_name, total_ms,
            )
        return result

    def sample_tokens(self, grammar_output):
        # 2026-08-17: execute_model's own timing (above) only covers the
        # forward pass + PP send/recv - it does NOT cover sample_tokens(),
        # a genuinely separate Worker method (temperature/top-p/top-k or
        # greedy selection from the last stage's lm_head logits) called
        # right after. Real gap found live: summing every execute_model
        # stage's total_ms across the whole A->B->C chain accounted for
        # only ~37ms of a real, independently-measured 59.2ms per token -
        # sample_tokens was one of the un-instrumented candidates for
        # where the other ~22ms goes. Same "measure launch to real output,
        # not just when Python returns" principle as execute_model's own
        # send_cuda_sync_ms: sample_tokens can itself queue async GPU work
        # (the actual sampling kernels) that doesn't necessarily finish by
        # the time this call returns, so an explicit sync is taken right
        # after to make sure the logged duration reflects when the result
        # was actually ready, not just when Python got control back.
        debug_timing = os.environ.get("VLLM_TRANSPORT_DEBUG_TIMING")
        _t0 = time.monotonic() if debug_timing else None
        result = super().sample_tokens(grammar_output)
        if debug_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_ms = (time.monotonic() - _t0) * 1000
            self_name = getattr(self, "_transport_timing_self_name", "?")
            logger.info(
                "[SAMPLE_TOKENS_TIMING] self=%s duration_ms=%.2f",
                self_name, total_ms,
            )
        return result
