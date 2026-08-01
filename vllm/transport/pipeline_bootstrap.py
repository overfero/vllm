"""Real (non-test-bypassed) bootstrap for a two-machine pipeline-parallel
`GroupCoordinator` topology, backed by `vllm.transport` instead of a
cross-machine `torch.distributed` process group.

See `README_GPTOSS_120B_UDP.md` ("Phase 5") for the full architecture
rationale. Summary: `torch.distributed.init_process_group`'s rendezvous
needs direct TCP reachability between all ranks, which two NAT'd,
no-public-IP machines don't have. This module never tries to form a
cross-machine `torch.distributed` group at all. Instead, on each machine:

1. A real, ordinary, *local* `torch.distributed` process group is formed
   (loopback rendezvous - always reachable, same machine, real NCCL for
   real local multi-GPU tensor parallelism). This is completely standard,
   unmodified vLLM behavior (`init_worker_distributed_environment`) - nothing
   about it is NAT-traversal-related, because it never leaves the machine.
2. The pipeline-parallel dimension - the one dimension that genuinely
   needs to cross the NAT boundary - is *not* part of that local group at
   all. It's a separate, synthetic `GroupCoordinator` built directly
   (bypassing `torch.distributed` entirely) with a `vllm.transport.Transport`
   already connected to the peer machine, then installed as the module-level
   `vllm.distributed.parallel_state._PP` singleton so every real call site
   (`get_pp_group()`, `make_layers()`, `is_first_rank`/`is_last_rank`, and -
   as of the previous phase - `send_tensor_dict`/`recv_tensor_dict`) uses it
   transparently.

This directly implements the "new ProcessGroup that uses the existing
Transport layer" the task asked for, scoped to exactly what pipeline
parallelism needs: point-to-point send/recv of one tensor dict per hop.
No collectives (all_reduce/all_gather/broadcast/barrier) are implemented -
none are needed for a point-to-point pipeline, and implementing them for
a synthetic group would be unused code.

Pipelines of 3+ stages (this project's real GPT-OSS 120B cluster is
exactly 3) have a middle rank with two distinct peers - the previous
stage and the next one - which is why this function accepts optional
`transport_prev`/`transport_next` in addition to the original single
`transport` parameter. See the matching `transport_prev`/`transport_next`
attributes and dispatch logic in `GroupCoordinator`
(vllm/distributed/parallel_state.py) for how `send()`/`recv()` route
between them. The original 2-member call convention
(`install_transport_pp_group(transport, pp_rank=..., pp_world_size=2)`,
proven end-to-end in test18_real_bootstrap_pp.py) is unchanged and still
the right call for a first or last rank.
"""
from __future__ import annotations

from vllm.transport.base import Transport


def install_transport_pp_group(
    transport: Transport | None = None,
    *,
    pp_rank: int,
    pp_world_size: int = 2,
    local_rank: int = 0,
    transport_prev: Transport | None = None,
    transport_next: Transport | None = None,
) -> None:
    """Replace the module-global `_PP` pipeline-parallel group with a
    synthetic, transport-backed one.

    Call this AFTER the real local `torch.distributed` bootstrap
    (`init_worker_distributed_environment` / `ensure_model_parallel_initialized`)
    has already formed the local TP/DP/EP/world groups (and, incidentally,
    a locally-trivial `_PP` this function discards) - never before, since
    that bootstrap requires forming its own trivial local PP group first
    and this function's whole point is to replace only the cross-machine
    dimension, not interfere with the local one. Call this BEFORE the
    model is constructed/loaded (`Worker.load_model()`), since layer
    partitioning (`make_layers()` -> `get_pp_indices()`) and
    `is_first_rank`/`is_last_rank` are read from the live `_PP` group at
    model-construction time, not cached from `ParallelConfig`.

    `pp_rank`/`pp_world_size` describe this synthetic group only - they
    are unrelated to (and do not need to match) the real local
    torch.distributed world_size/rank used for TP.

    Two ways to call this:

    - First or last rank of the pipeline (exactly one peer): pass
      `transport` alone, exactly as before. Direction (prev vs. next) is
      inferred from `pp_rank` (rank 0 -> its one connection is "next";
      the last rank -> its one connection is "prev").
    - A middle rank (two peers, e.g. Machine B's stage in a 3-stage
      pipeline): pass `transport_prev` and `transport_next` explicitly
      instead of `transport`. Both must be independently-established
      connections (see `establish_pp_transports` for the usual way to
      get them) - one to the previous stage, one to the next.
    """
    if transport is not None and (transport_prev is not None or transport_next is not None):
        raise ValueError(
            "pass either `transport` (first/last rank, one peer) or "
            "`transport_prev`/`transport_next` (middle rank, two peers), "
            "not both"
        )
    if transport is not None:
        if pp_rank == 0:
            transport_next = transport
        elif pp_rank == pp_world_size - 1:
            transport_prev = transport
        else:
            raise ValueError(
                f"pp_rank={pp_rank} of pp_world_size={pp_world_size} is a "
                "middle-of-chain rank with two peers - pass "
                "transport_prev/transport_next explicitly instead of a "
                "single `transport`"
            )
    if transport_prev is None and transport_next is None:
        raise ValueError(
            "must supply at least one of transport, transport_prev, "
            "transport_next"
        )

    import vllm.distributed.parallel_state as ps

    group = object.__new__(ps.GroupCoordinator)
    # `transport` remains the sentinel every existing transport-branch
    # check (`if self.transport is not None`) tests to decide "is this
    # group transport-backed at all" - keep it non-None whenever either
    # peer connection is set.
    group.transport = transport_next if transport_next is not None else transport_prev
    group.transport_prev = transport_prev
    group.transport_next = transport_next
    group.rank = pp_rank
    group.local_rank = local_rank
    group.rank_in_group = pp_rank
    group.world_size = pp_world_size
    group.ranks = list(range(pp_world_size))
    group.unique_name = f"transport_pp:{pp_rank}"

    ps._PP = group


def establish_pp_transports(
    *,
    pp_rank: int,
    pp_world_size: int,
    local_rank: int,
    self_name: str,
    prev_name: str | None,
    next_name: str | None,
    backend: str | None = None,
    signaling_url: str | None = None,
    udp_port_base: int = 30000,
    tcp_port_base: int = 30000,
    tcp_listen_host: str = "0.0.0.0",
    tcp_connect_host_prev: str | None = None,
    tcp_connect_host_next: str | None = None,
    connect_timeout: float = 120.0,
) -> tuple[Transport | None, Transport | None]:
    """Connect this rank's transport link(s) and return `(transport_prev,
    transport_next)` - whichever of the two this rank actually has (a first
    or last rank has exactly one; a middle rank has both).

    One call per worker PROCESS, not per machine: with TP=2, each machine
    runs two independent worker processes (one per local GPU), and each
    needs its OWN connection(s) to the SAME-`local_rank` worker on the
    neighboring machine's stage - vLLM's real PP send/recv
    (`gpu_worker.py`'s `isend_tensor_dict`/`irecv_tensor_dict`) runs
    identically on every rank, not just a driver rank, so every local GPU
    needs a working link, not just one representative per machine.

    ID scheme (deterministic from names alone, no coordination needed
    beyond every machine knowing its immediate neighbors' names): for the
    link between machine `L` and machine `R` at tensor-parallel rank `i`,
    `L`'s side registers as self_id=f"{L}-tp{i}-to-{R}",
    peer_id=f"{R}-tp{i}-to-{L}", and `R` does the mirror image. This keeps
    Machine B's two links (to A and to C) from colliding even though both
    involve the same local_rank.

    For TCP, the lower pp_rank side of each link listens
    (`tcp_listen_host`, default "0.0.0.0") and the higher pp_rank side
    connects to `tcp_connect_host_prev`/`tcp_connect_host_next` (the
    listening peer's real reachable address - only viable if the two
    machines have direct L3 reachability, e.g. same LAN/VPC; NAT'd
    machines need `backend="udp"` instead, which is this project's actual
    target transport and does not require direct reachability).
    """
    if (prev_name is None) != (pp_rank == 0):
        raise ValueError(
            f"pp_rank={pp_rank} of pp_world_size={pp_world_size}: "
            f"prev_name={prev_name!r} is inconsistent with rank position "
            "(prev_name must be None iff pp_rank==0)"
        )
    if (next_name is None) != (pp_rank == pp_world_size - 1):
        raise ValueError(
            f"pp_rank={pp_rank} of pp_world_size={pp_world_size}: "
            f"next_name={next_name!r} is inconsistent with rank position "
            "(next_name must be None iff pp_rank==pp_world_size-1)"
        )

    from vllm.transport import TransportConfig, get_transport

    def _connect(
        peer_name: str, is_prev: bool, tcp_connect_host: str | None
    ) -> Transport:
        self_id = f"{self_name}-tp{local_rank}-to-{peer_name}"
        peer_id = f"{peer_name}-tp{local_rank}-to-{self_name}"
        # TCP only: the lower pp_rank side of a link listens, the higher
        # side connects. A "prev" link's peer is always the lower rank
        # (pp_rank - 1), so we never listen for it; a "next" link's peer
        # is always the higher rank (pp_rank + 1), so we always listen.
        we_listen = not is_prev

        transport = get_transport(backend)
        port_offset = local_rank * 2 + (0 if is_prev else 1)
        config = TransportConfig(
            self_id=self_id,
            peer_id=peer_id,
            host=tcp_listen_host if we_listen else (tcp_connect_host or "127.0.0.1"),
            port=tcp_port_base + port_offset,
            listen=we_listen,
            signaling_url=signaling_url or "http://127.0.0.1:8000",
            udp_mode="preserve",
            udp_port=udp_port_base + port_offset,
            connect_timeout=connect_timeout,
        )
        transport.connect(config)
        return transport

    transport_prev = _connect(prev_name, is_prev=True, tcp_connect_host=tcp_connect_host_prev) if prev_name else None
    transport_next = _connect(next_name, is_prev=False, tcp_connect_host=tcp_connect_host_next) if next_name else None
    return transport_prev, transport_next
