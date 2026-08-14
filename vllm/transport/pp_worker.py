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
        # Real generation-throughput investigation (2026-08-12): wraps the
        # real execute_model() call (recv-from-prev + local forward compute
        # + send-to-next, for a middle stage) with a total-duration log so
        # local compute time can be derived as
        # total_duration - sum(TRANSPORT_TIMING send/recv durations logged
        # by udp_transport.py during this same call) - correlate by
        # timestamp per machine, single in-flight request only.
        _t0 = time.monotonic()
        result = super().execute_model(scheduler_output)
        logger.info(
            "[EXECUTE_MODEL_TIMING] self=%s duration_ms=%.2f",
            getattr(self, "_transport_timing_self_name", "?"),
            (time.monotonic() - _t0) * 1000,
        )
        return result
