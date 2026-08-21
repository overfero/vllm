"""RawUdpBench: a thin Python wrapper around
`vllm._rust_udp_raw_engine`'s `send_reliable`/`recv_reliable` (Rust
`sendmmsg(2)`/`recvmmsg(2)` batching plus an ACK-windowed pacing loop -
see `rust/src/udp_raw_engine/src/lib.rs`'s module docstring for the full
design rationale). All of the chunking/windowing/ACK/pacing logic lives
in Rust; this class only derives buffer-aware parameters (from
`SO_SNDBUF`) and calls into it - no per-chunk Python loop.

**This is explicitly a benchmarking experiment, not a `Transport`
implementation** - it is deliberately NOT registered in `factory.py`/
`_KNOWN_BACKENDS` and must never be used for real deployment traffic.
Every other backend in this package (`tcp`/`udp`/`quic`/`quic-rs`/
`quic-shared`/`quic-rs-shared`) guarantees reliable, ordered delivery via
a real retry/retransmit layer; this one still has NONE of that (no
retries, no retransmission, no gap recovery) - a genuinely lost chunk is
gone forever. It exists to answer one question precisely: after profiling
repeatedly found this project's own userspace/Python per-packet dispatch
overhead (not the kernel, not the protocol) as the dominant cost in
`udp_transport.py`'s and `quic_rs_transport.py`'s throughput (see both
those modules' docstrings), what IS the real throughput ceiling of a
maximally-optimized UDP path in this project?

Two things this deliberately does differently from every reliable backend
here, in service of that one question:
  1. Large chunk size (default 61440 bytes, not ~1200-1452 like
     `quic_transport.py`/`quic_rs_transport.py` use). Real-world UDP paths
     cap safe payload size near the smallest link's MTU (~1200-1500 bytes
     end to end) to avoid IP fragmentation - but the LOOPBACK interface's
     MTU is ~65536 bytes (`cat /sys/class/net/lo/mtu`), so a single
     datagram can carry ~45x more payload here than a real network path
     would tolerate without fragmenting. This is intentionally loopback-
     specific - a real cross-machine deployment would need to fall back to
     a real-path-safe chunk size (this module does not attempt to
     discover path MTU at all).
  2. `sendmmsg`/`recvmmsg` batching plus a minimal ACK-windowed pacing
     scheme, entirely inside Rust (`RawUdpEngine::send_reliable`/
     `recv_reliable`) - many datagrams move in ONE syscall each direction,
     instead of one syscall per datagram the way every asyncio-based
     backend here is architecturally stuck with, AND the pacing loop
     itself never crosses back into Python between chunks (see
     `rust/src/udp_raw_engine/src/lib.rs` for the buffer-derived batch/
     window sizing rationale and the pacing bugs that shaped it).
"""
from __future__ import annotations

import socket

try:
    import vllm._rust_udp_raw_engine as _ue
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "RawUdpBench requires the compiled vllm._rust_udp_raw_engine "
        "extension (built from rust/src/udp_raw_engine/python via "
        "./build_rust.sh) - it was not found. Original error: "
        f"{_exc}"
    ) from _exc

DEFAULT_CHUNK_BYTES = 61440  # comfortably under loopback's ~65536 MTU, room for a 5-byte header
_HEADER_BYTES = 5  # 1 byte type tag + 4 byte seq/ack number


class RawUdpBench:
    """One connected raw UDP socket, driven via `RawUdpEngine::
    send_reliable`/`recv_reliable` - batched sendmmsg/recvmmsg plus a
    minimal ACK-windowed pacing scheme, all inside Rust (see module
    docstring). `sock` must already be `connect()`-ed to the single peer
    this side talks to (matches `RawUdpEngine`'s own "no address per
    message" design - see the Rust crate's docstring)."""

    def __init__(self, sock: socket.socket, chunk_bytes: int = DEFAULT_CHUNK_BYTES, batch: int | None = None) -> None:
        self._engine = _ue.PyRawUdpEngine.from_fd(sock.fileno())
        self._chunk_payload = chunk_bytes - _HEADER_BYTES
        try:
            sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        except OSError:
            sndbuf = 64 * 1024
        # Batch/window sized off the real kernel buffer, not a fixed
        # guess - see `rust/src/udp_raw_engine/src/lib.rs`'s docstring for
        # the overflow/deadlock bugs this specific derivation fixes.
        safe_batch = max(1, sndbuf // self._chunk_payload)
        self._batch = batch if batch is not None else safe_batch
        self._window_chunks = max(1, sndbuf // self._chunk_payload)

    def send(self, data: bytes, timeout: float = 30.0) -> None:
        """Splits, paces, and reliably sends `data` - see `RawUdpEngine::
        send_reliable`'s docstring for the ACK-windowed pacing +
        Go-Back-N retransmission scheme. Blocks (GIL released) until
        every chunk has actually been acknowledged by the peer, or raises
        if `timeout` seconds pass without full delivery (peer gone)."""
        self._engine.send_reliable(
            data, self._chunk_payload, self._batch, self._window_chunks, int(timeout * 1000)
        )

    def recv_total(self, expected_bytes: int, timeout: float) -> tuple[bytes, int, int]:
        """Receives until `expected_bytes` worth of PAYLOAD has arrived or
        `timeout` elapses, whichever first, sending periodic ACKs back so
        the peer's `send()` can pace itself (see module docstring).
        Returns `(concatenated_payload_in_arrival_order, chunks_received,
        max_seq_gap_observed)` - a non-zero gap means loss or reordering
        was detected (reported, not corrected - there is still no
        retransmission)."""
        data, chunks, gap = self._engine.recv_reliable(
            expected_bytes, self._chunk_payload, self._batch, self._window_chunks, int(timeout * 1000)
        )
        return bytes(data), chunks, gap
