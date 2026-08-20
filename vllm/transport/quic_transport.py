"""QUICTransport: layers a real QUIC connection (RFC 9000/9001, via the
`aioquic` library) on top of the existing, already-proven UDP hole-punch
transport (`udp_holepunch/peer.py`, imported unmodified - same pattern as
udp_transport.py).

Why this exists (see the design doc this was built from for the full audit):
UDPTransport hand-rolls its own reliability (status-query/resend), RTO
(hardcoded constants), and NAT-rebind tolerance (a "send to the last 4
observed addresses" hack) on top of best-effort UDP. QUIC already solves
all of that as part of the protocol - reliable/ordered-per-stream delivery,
RTT-adaptive loss recovery (RFC 9002), a real congestion controller,
integrity/confidentiality via TLS 1.3, and (the reason QUIC specifically,
not just "add TCP" - TCP dies outright on any 4-tuple change) connection
migration via PATH_CHALLENGE/PATH_RESPONSE that survives NAT rebinding
mid-session. This module does not reimplement any of that; it wires the
existing hole-punched socket into aioquic's low-level `QuicConnection`
state machine and lets aioquic own reliability/congestion/migration
entirely.

Design:
  1. Hole-punch exactly as UDPTransport does (register/wait_for_peer/
     punch_loop against the SAME signaling server), reusing peer.py
     unmodified - this phase is transport-agnostic and already proven.
  2. Once punched (`protocol.established`), start a real QUIC handshake
     on the SAME socket/address pair. `aioquic.asyncio.connect()`/`serve()`
     always open their own socket (no supported way to hand them an
     already-bound, already-hole-punched one), so this drives the
     low-level `aioquic.quic.connection.QuicConnection` state machine
     directly, following the exact pattern aioquic's own
     `asyncio/protocol.py:QuicConnectionProtocol` uses internally
     (`_process_events`/`transmit`/`_handle_timer` - mirrored here as
     `_process_quic_events`/`_transmit`/`_handle_timer`) - just adapted to
     share a socket instead of owning one, and to feed a plain
     synchronous byte-queue (`Transport.recv()`'s contract) instead of
     asyncio.StreamReader/Writer pairs.
  3. QUIC datagrams share the wire with peer.py's own single-ASCII-byte-
     tagged handshake/ping-pong traffic (tags "P"/"O"/"D"/"R"/"S"/"W"/"V"/
     "F"/"X", all in the 0x40-0x5A range). A real QUIC short-header first
     byte can also fall in that range, so - unlike udp_transport.py's "M"/
     "Q"/"A" tags, which never need to coexist with anything else on the
     wire - QUIC traffic here is prefixed with `_QUIC_TAG = b"\\x01"`, a
     byte value outside both peer.py's tag range and any valid QUIC
     first-header-byte pattern, so there is no ambiguity in either
     direction. peer.py's own dispatch already ignores unrecognized tags
     (see udp_transport.py's docstring point 2 for the same reasoning).
  4. One persistent unidirectional QUIC stream per direction, opened on the
     first `send()` and reused for that connection's lifetime - NOT one
     stream per message. Each message is framed within it via an 8-byte
     big-endian length prefix (see `_on_stream_data`'s docstring for why:
     QUIC gives no ordering guarantee *between* streams, so one-stream-per-
     message let messages arrive out of send() order under loss, a real bug
     caught by tests/transport/test8_tensor_streaming.py). aioquic still
     owns fragmentation to path MTU, retransmission, pacing, and the send/
     receive windows for that stream's bytes - no manual chunking/batching/
     status-query code exists here, unlike UDPTransport.

Known limitations (v1, documented deliberately - see the design doc):
  - No real peer authentication: certificates are ephemeral, self-signed,
    generated fresh per process, and the client side disables chain
    verification (`verify_mode = CERT_NONE`). This gets integrity and
    confidentiality-in-transit from TLS 1.3, but NOT "is this actually who
    I think it is" - anyone who can complete the same UDP hole punch (i.e.
    anyone the signaling server introduced) can also complete the QUIC
    handshake. Pinning the cert fingerprint through the existing signaling
    channel is a real future improvement, not implemented here.
  - `Transport.send()` stays synchronous/blocking (queues onto the QUIC
    connection and returns - it does not wait for the peer to ACK). This
    means concurrent messages DO get real multiplexing at the QUIC layer
    (separate streams, no head-of-line blocking between them at the
    transport level), but a caller that wants to fire off N messages
    without blocking between them still needs its own threading/async
    fan-out - `vllm/distributed/parallel_state.py`'s `isend_tensor_dict`/
    `irecv_tensor_dict` are still synchronous wrappers as of this writing.
    Making those genuinely async is a separate, larger change (see the
    design doc's own plan) deliberately left out of this module's scope.

See vllm/transport/README.md for the project's overall transport scope.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import socket
import ssl
import sys
import threading
import time
from pathlib import Path

from aioquic.buffer import Buffer
from aioquic.quic import events as quic_events
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.packet import pull_quic_header

from vllm.logger import init_logger
from vllm.transport.base import Transport, TransportConfig

logger = init_logger(__name__)


def _locate_existing_transport_dir() -> Path:
    """Same lookup udp_transport.py uses - kept independent (not imported
    from there) so this module has no hard dependency on udp_transport.py's
    internals, only on peer.py itself."""
    env_path = os.environ.get("VLLM_UDP_TRANSPORT_DIR")
    candidates = [Path(env_path)] if env_path else []
    candidates += [
        Path(__file__).resolve().parents[2] / "udp_holepunch",
    ]
    for candidate in candidates:
        if (candidate / "peer.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the existing UDP hole-punch transport (peer.py). "
        "Set VLLM_UDP_TRANSPORT_DIR to the directory that contains it."
    )


_existing_transport_dir = _locate_existing_transport_dir()
if str(_existing_transport_dir) not in sys.path:
    sys.path.insert(0, str(_existing_transport_dir))

import peer as _hp  # noqa: E402  (unmodified - the existing transport, treated as a library)

_QUIC_TAG = b"\x01"  # see module docstring point 3 for why this exact byte
ALPN_PROTOCOL = "vllm-pp-v1"

# Configurable via env var (see TransportConfig for the per-call override) -
# generous defaults, but always finite. idle_timeout: how long with no
# traffic before aioquic itself declares the peer dead and fires
# ConnectionTerminated (fixes UDPTransport's critical bug: recv() can hang
# forever with no built-in dead-peer detection). max_message_bytes: an
# explicit application-level cap on top of QUIC's own flow-control limits -
# defense in depth, not relying solely on library defaults for this.
DEFAULT_IDLE_TIMEOUT_S = float(os.environ.get("VLLM_TRANSPORT_QUIC_IDLE_TIMEOUT_S", "45"))
DEFAULT_MAX_MESSAGE_BYTES = int(
    os.environ.get("VLLM_TRANSPORT_QUIC_MAX_MESSAGE_BYTES", str(2 * 1024 * 1024 * 1024))
)
_KEEPALIVE_INTERVAL_SECONDS = 15.0  # matches UDPTransport's own cadence


def _generate_self_signed_cert():
    """Ephemeral, self-signed - see module docstring's "Known limitations"
    for what this does and does not provide."""
    import datetime as _dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vllm-pp-transport")])
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))
        .not_valid_after(now + _dt.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    return cert, key


_LEN_PREFIX_BYTES = 8  # big-endian message-length header, see _on_stream_data


class _InboundStream:
    """Accumulates one QUIC stream's raw bytes and peels off complete,
    length-prefixed application messages as they become available. See
    _on_stream_data for why streams are framed this way instead of one
    QUIC stream per message."""

    __slots__ = ("buf",)

    def __init__(self) -> None:
        self.buf = bytearray()


class _QuicAdapterProtocol(_hp.PeerProtocol):
    """Drives one `QuicConnection` on top of the existing hole-punch
    protocol's socket. Everything PeerProtocol already does (handshake,
    rebind *detection* logging, ping/pong) keeps working unchanged - this
    only *adds* a branch for the one tag ("\\x01") the base class doesn't
    recognize. Real NAT-rebind *tolerance* for the QUIC traffic itself
    comes from aioquic's own connection migration, not from anything added
    here - see module docstring point 3 and `_handle_datagram` below.
    """

    def __init__(
        self,
        peer_addr: tuple[str, int],
        recv_queue: "queue.Queue[bytes | Exception]",
        max_message_bytes: int,
        is_client: bool,
        quic_config: QuicConfiguration,
    ) -> None:
        super().__init__(peer_addr)
        self._recv_queue = recv_queue
        self._max_message_bytes = max_message_bytes
        self._is_client = is_client
        self._quic_config = quic_config
        self._quic: QuicConnection | None = None
        self._inbound: dict[int, _InboundStream] = {}
        self._timer_handle: asyncio.TimerHandle | None = None
        self._timer_at: float | None = None
        # Created eagerly, here, rather than later in QUICTransport._connect_async's
        # "Phase 2" - this protocol's datagram_received callback is already live
        # the moment create_datagram_endpoint() returns (Phase 1), so a fast peer
        # can complete the ENTIRE QUIC handshake - and fire HandshakeCompleted -
        # before Phase 2 would otherwise get around to creating this Future. That
        # was a real, timing-dependent bug: the event fired against a `None`
        # waiter, was silently dropped (see _process_quic_events below), and
        # connect() then hung until the QUIC connection's own idle_timeout killed
        # it (~45s) with no application data ever having been sent. Creating the
        # Future here, before any datagram can possibly be processed, closes the
        # race entirely.
        self.handshake_waiter: asyncio.Future = self.loop.create_future()
        self.closed_exc: Exception | None = None

    def attach_quic(self, quic: QuicConnection) -> None:
        """Client role only - the client generates its own connection ID and
        can construct its QuicConnection up front (see QUICTransport.
        _connect_async). The server role instead lazily constructs one on
        the first real datagram - see _handle_datagram below for why."""
        self._quic = quic

    def _handle_datagram(self, data: bytes, addr) -> None:
        super()._handle_datagram(data, addr)  # preserve all original hole-punch/ping-pong behavior
        if not data or data[0:1] != _QUIC_TAG:
            return
        payload = data[1:]

        if self._quic is None:
            # Server role, first QUIC datagram ever seen on this link.
            # Unlike the client, QuicConnection.__init__ on the server side
            # requires `original_destination_connection_id` - the
            # destination CID *the client itself chose* for its Initial
            # packet - which only exists once that packet has actually
            # arrived. aioquic's own aioquic.asyncio.server.QuicServer does
            # this same peek-then-construct dance (see its
            # datagram_received); this mirrors that non-retry code path
            # (no address-spoofing protection via a Retry round-trip - not
            # needed here, the UDP hole punch already confirms both sides
            # are a real, mutually-reachable pair before any QUIC traffic
            # is sent at all).
            try:
                header = pull_quic_header(
                    Buffer(data=payload),
                    host_cid_length=self._quic_config.connection_id_length,
                )
            except ValueError:
                return  # not a valid QUIC Initial packet - ignore, never crash the link
            self._quic = QuicConnection(
                configuration=self._quic_config,
                original_destination_connection_id=header.destination_cid,
            )

        now = self.loop.time()
        # `addr` here is the datagram's REAL observed source each time (never
        # cached) - this is exactly what lets aioquic's own path validation
        # (PATH_CHALLENGE/PATH_RESPONSE) detect and migrate to a rebound NAT
        # mapping. No "send to several recent addresses" hack is needed for
        # QUIC traffic (contrast with udp_transport.py's _AdapterProtocol).
        self._quic.receive_datagram(payload, addr, now)
        self._process_quic_events()
        self._transmit()

    def _process_quic_events(self) -> None:
        assert self._quic is not None
        event = self._quic.next_event()
        while event is not None:
            if isinstance(event, quic_events.StreamDataReceived):
                self._on_stream_data(event)
            elif isinstance(event, quic_events.HandshakeCompleted):
                if not self.handshake_waiter.done():
                    self.handshake_waiter.set_result(None)
            elif isinstance(event, quic_events.ConnectionTerminated):
                self._on_terminated(event)
            event = self._quic.next_event()

    def _on_stream_data(self, event: quic_events.StreamDataReceived) -> None:
        """Each direction uses exactly ONE persistent QUIC stream for its
        whole lifetime (see QUICTransport._send_async), not one stream per
        message. This was a real, found-by-testing bug in the original one-
        stream-per-message design: QUIC guarantees ordered delivery *within*
        a stream, but explicitly makes NO ordering guarantee *between*
        streams - that's the whole point of QUIC eliminating head-of-line
        blocking. Under any loss/retransmission, a later message's stream
        could complete (and get handed to _recv_queue) before an earlier
        message's stream did, silently reordering application messages
        relative to send() call order. tests/transport/test8_tensor_streaming.py
        caught this directly (tensors arriving in the wrong slot, reported as
        "corrupted" though every individual stream's bytes were intact).
        Framing all of one direction's messages onto a single ordered stream,
        each prefixed with an 8-byte big-endian length, restores the same
        "messages arrive in send() order" guarantee TCPTransport/UDPTransport
        already provide, at the cost of head-of-line blocking between
        messages in the same direction - an acceptable trade for this
        project's real use case (PP activation tensors, which must arrive in
        generation-step order anyway).
        """
        assert self._quic is not None
        buf = self._inbound.get(event.stream_id)
        if buf is None:
            buf = self._inbound[event.stream_id] = _InboundStream()
        buf.buf += event.data

        while True:
            if len(buf.buf) < _LEN_PREFIX_BYTES:
                return
            msg_len = int.from_bytes(buf.buf[:_LEN_PREFIX_BYTES], "big")
            if msg_len > self._max_message_bytes:
                # Fixes UDPTransport-class bug #2 (unvalidated length header
                # -> unbounded allocation): checked as soon as the length
                # prefix itself is available, before ever buffering
                # `msg_len` bytes - an explicit cap on top of QUIC's own
                # flow-control limits, not relying solely on library
                # defaults.
                del self._inbound[event.stream_id]
                self._quic.close(
                    error_code=1,
                    reason_phrase=f"message on stream {event.stream_id} exceeded "
                    f"{self._max_message_bytes} bytes",
                )
                self._transmit()
                return
            total_needed = _LEN_PREFIX_BYTES + msg_len
            if len(buf.buf) < total_needed:
                return
            complete = bytes(buf.buf[_LEN_PREFIX_BYTES:total_needed])
            del buf.buf[:total_needed]
            self._recv_queue.put_nowait(complete)

    def _on_terminated(self, event: quic_events.ConnectionTerminated) -> None:
        self.closed_exc = ConnectionError(
            f"QUIC connection {self.__class__.__name__} terminated: "
            f"error_code={event.error_code} reason={event.reason_phrase!r}"
        )
        if not self.handshake_waiter.done():
            self.handshake_waiter.set_exception(self.closed_exc)
        # Fixes UDPTransport-class bug #3 (recv() with no timeout, and no
        # peer-death detection at all, hangs forever): a dead connection
        # unblocks every current AND future recv() call with a real
        # exception instead of leaving them parked on an empty queue.
        self._recv_queue.put_nowait(self.closed_exc)

    def _transmit(self) -> None:
        assert self._quic is not None
        now = self.loop.time()
        for data, addr in self._quic.datagrams_to_send(now=now):
            self.transport.sendto(_QUIC_TAG + data, addr)
        timer_at = self._quic.get_timer()
        if self._timer_handle is not None and self._timer_at != timer_at:
            self._timer_handle.cancel()
            self._timer_handle = None
        if self._timer_handle is None and timer_at is not None:
            self._timer_handle = self.loop.call_at(timer_at, self._handle_timer)
        self._timer_at = timer_at

    def _handle_timer(self) -> None:
        assert self._quic is not None
        now = max(self._timer_at or 0.0, self.loop.time())
        self._timer_handle = None
        self._timer_at = None
        self._quic.handle_timer(now=now)
        self._process_quic_events()
        self._transmit()


class QUICTransport(Transport):
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._protocol: _QuicAdapterProtocol | None = None
        self._transport_obj: asyncio.DatagramTransport | None = None
        self._quic: QuicConnection | None = None
        self._recv_queue: "queue.Queue[bytes | Exception]" = queue.Queue()
        self._keepalive_task: asyncio.Task | None = None
        self._self_id = "?"
        self._peer_id = "?"
        # One persistent unidirectional stream, opened lazily on the first
        # send() and reused for every subsequent message this side sends -
        # see _send_async and _QuicAdapterProtocol._on_stream_data for why
        # (message-ordering fix).
        self._out_stream_id: int | None = None

    def connect(self, config: TransportConfig) -> None:
        self._self_id = config.self_id
        self._peer_id = config.peer_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_async(config), self._loop)
        fut.result(timeout=config.connect_timeout + 5)

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_async(self, config: TransportConfig) -> None:
        # ---- Phase 1: identical UDP hole-punch to UDPTransport's own
        # _connect_async - this half is transport-agnostic and already
        # proven; see udp_transport.py for the same sequence with more
        # extensive comments on each step. ----
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, opt, _hp.SOCKET_BUFFER_REQUEST)
        sock.bind(("0.0.0.0", config.udp_port))

        if config.udp_mode == "stun":
            own_ip, own_port = _hp.stun_get_mapped_address(sock, config.stun_host, config.stun_port)
        else:
            own_ip, own_port = None, sock.getsockname()[1]

        reg_resp = _hp.register(config.signaling_url, config.self_id, own_port)
        own_ip = own_ip or reg_resp["public_ip"]

        peer_info = _hp.wait_for_peer(config.signaling_url, config.self_id, config.peer_id)
        coordinator_peer_addr = (peer_info["public_ip"], peer_info["udp_port"])
        delay = peer_info["start_at"] - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        max_message_bytes = int(config.quic_max_message_bytes) or DEFAULT_MAX_MESSAGE_BYTES

        # `config.listen` (already used this way by pipeline_bootstrap.py's
        # `we_listen` convention for TCPTransport) decides TLS role: the
        # listening side presents a certificate (QUIC "server"), the
        # connecting side does not (QUIC "client"). Hole punching itself is
        # symmetric - this is purely about who presents a cert. Built here,
        # before the punch, because the protocol needs it immediately on
        # its very first real datagram (server role - see
        # _QuicAdapterProtocol._handle_datagram for why it can't construct
        # its QuicConnection any earlier than that).
        idle_timeout = float(config.quic_idle_timeout) or DEFAULT_IDLE_TIMEOUT_S
        is_client = not config.listen

        quic_config = QuicConfiguration(
            alpn_protocols=[ALPN_PROTOCOL],
            is_client=is_client,
            idle_timeout=idle_timeout,
        )
        if is_client:
            # See module docstring's "Known limitations": no real peer
            # authentication in v1, only integrity/confidentiality-in-transit.
            quic_config.verify_mode = ssl.CERT_NONE
        else:
            cert, key = _generate_self_signed_cert()
            quic_config.certificate = cert
            quic_config.private_key = key

        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _QuicAdapterProtocol(
                coordinator_peer_addr, self._recv_queue, max_message_bytes, is_client, quic_config
            ),
            sock=sock,
        )

        punch_task = asyncio.create_task(_hp.punch_loop(protocol))
        deadline = time.monotonic() + config.connect_timeout
        while not protocol.established and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        punch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await punch_task

        if not protocol.established:
            transport.close()
            raise ConnectionError(
                f"UDP hole punch failed: no packet from peer within {config.connect_timeout}s"
            )

        # ---- Phase 2: real QUIC handshake, on the now-punched socket/addr. ----
        self._protocol = protocol
        self._transport_obj = transport

        # protocol.handshake_waiter was already created back in
        # _QuicAdapterProtocol.__init__ (Phase 1, before create_datagram_endpoint
        # even returned) - NOT here. See that Future's own comment for why: this
        # protocol's datagram_received callback is live from the moment the
        # socket is wired up, so a fast peer can complete the whole handshake,
        # including firing HandshakeCompleted, before this "Phase 2" code would
        # otherwise get around to creating it - a real, timing-dependent lost-
        # wakeup bug this fixes.
        handshake_waiter = protocol.handshake_waiter

        if is_client:
            # The client generates its own connection ID, so (unlike the
            # server) it can construct its QuicConnection immediately and
            # send the first Initial packet itself.
            quic = QuicConnection(configuration=quic_config)
            protocol.attach_quic(quic)
            self._quic = quic
            quic.connect(protocol.peer_addr, now=loop.time())
            protocol._transmit()
        # else: server role - stays passive; the client's Initial packet,
        # once it arrives via _handle_datagram, both constructs this side's
        # QuicConnection (lazily - see that method) and drives the rest of
        # the handshake through the normal receive_datagram path. self._quic
        # gets backfilled from protocol._quic just before returning, below.

        try:
            await asyncio.wait_for(handshake_waiter, timeout=config.connect_timeout)
        except asyncio.TimeoutError:
            transport.close()
            raise ConnectionError(
                f"QUIC handshake did not complete within {config.connect_timeout}s "
                "(the UDP hole punch itself succeeded - the ALPN/TLS layer on top "
                "did not finish; check that both sides agree on --transport quic "
                "and on which side has --listen)"
            ) from None

        # Server role: protocol._quic was constructed lazily inside
        # _handle_datagram (see above) - back-fill this instance's own
        # reference now that the handshake (and therefore that lazy
        # construction) has definitely happened.
        self._quic = protocol._quic
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(protocol))

    async def _keepalive_loop(self, protocol: "_QuicAdapterProtocol") -> None:
        # Two independent keepalive concerns, both handled here:
        # 1. NAT pinhole freshness: reuses peer.py's own ping/pong (tags
        #    "P"/"O", untouched by our "\x01"-prefixed QUIC traffic) - same
        #    reasoning UDPTransport's own keepalive loop documents.
        # 2. QUIC's OWN idle_timeout: a real QUIC PING is ALSO required,
        #    separately - aioquic's `_close_at` deadline (the idle_timeout
        #    clock) is only reset by an actual QUIC-tagged packet it
        #    successfully decrypts (`QuicConnection._payload_received` in
        #    aioquic/quic/connection.py, called from `receive_datagram`).
        #    The hole-punch ping above does NOT count - it's not
        #    `_QUIC_TAG`-prefixed, so it never reaches `receive_datagram`
        #    at all. Without a real QUIC PING too, a genuinely idle link
        #    (e.g. an inference server with no request traffic for longer
        #    than quic_idle_timeout) would have aioquic kill the
        #    connection on its own, even though both peers and the NAT
        #    mapping are still perfectly alive - a real bug found while
        #    preparing this transport for an actual multi-machine
        #    deployment, not a theoretical one.
        seq = 0
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL_SECONDS)
            seq += 1
            protocol.send(_hp.pack_ping(seq, time.monotonic()))
            if protocol._quic is not None and protocol.closed_exc is None:
                protocol._quic.send_ping(seq)
                protocol._transmit()

    def send(self, data: bytes) -> None:
        if self._loop is None or self._quic is None:
            raise RuntimeError("send() called before connect()")
        _t0 = time.monotonic()
        fut = asyncio.run_coroutine_threadsafe(self._send_async(data), self._loop)
        fut.result()
        logger.info(
            "[TRANSPORT_TIMING] self=%s peer=%s op=send bytes=%d duration_ms=%.2f",
            self._self_id, self._peer_id, len(data), (time.monotonic() - _t0) * 1000,
        )

    async def _send_async(self, data: bytes) -> None:
        assert self._quic is not None and self._protocol is not None
        if self._protocol.closed_exc is not None:
            raise self._protocol.closed_exc
        # ONE persistent unidirectional stream for this whole connection's
        # lifetime in this direction, opened on the first send() and never
        # end_stream=True'd - not one stream per message. See
        # _QuicAdapterProtocol._on_stream_data for why: QUIC gives no
        # ordering guarantee *between* streams, so one-stream-per-message
        # could (and, under real loss, did - see that method's docstring)
        # deliver messages out of send() order. Each message is instead
        # framed within this single ordered stream via an 8-byte big-endian
        # length prefix. No manual chunking/batching/status-query beyond
        # that framing: aioquic still owns fragmentation, retransmission,
        # pacing, and the congestion window for the stream's bytes.
        if self._out_stream_id is None:
            self._out_stream_id = self._quic.get_next_available_stream_id(is_unidirectional=True)
        header = len(data).to_bytes(_LEN_PREFIX_BYTES, "big")
        self._quic.send_stream_data(self._out_stream_id, header + data, end_stream=False)
        self._protocol._transmit()

    def recv(self, timeout: float | None = None) -> bytes:
        _t0 = time.monotonic()
        try:
            data = self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"recv() timed out after {timeout}s") from None
        if isinstance(data, Exception):
            # Poison value from _QuicAdapterProtocol._on_terminated - put it
            # back so any OTHER concurrent/later recv() caller also observes
            # the same terminal state, rather than only the first caller to
            # dequeue it.
            self._recv_queue.put_nowait(data)
            raise data
        logger.info(
            "[TRANSPORT_TIMING] self=%s peer=%s op=recv bytes=%d duration_ms=%.2f",
            self._self_id, self._peer_id, len(data), (time.monotonic() - _t0) * 1000,
        )
        return data

    def close(self) -> None:
        if self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
            with contextlib.suppress(Exception):
                # Best-effort: if the peer is already gone, or the drain
                # barrier below times out, still fall through to actually
                # closing our own side rather than hanging close() forever.
                fut.result(timeout=11.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    async def _close_async(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
        if (
            self._quic is not None
            and self._protocol is not None
            and self._protocol.closed_exc is None
            and self._out_stream_id is not None
        ):
            # Graceful drain, THEN close - not close() immediately. send()
            # only guarantees data was handed to aioquic's send buffer, not
            # that it actually left the wire: a message larger than the
            # current congestion window (~12-14KB initially) only gets
            # partially flushed by _send_async's one _transmit() call - the
            # rest drains later, opportunistically, via the _transmit()
            # calls that _handle_datagram/_handle_timer already make as ACKs
            # and pacing timers fire and the window grows. Closing
            # immediately after send() returns raced ahead of that drain and
            # truncated the message with an immediate CONNECTION_CLOSE
            # before the peer ever saw the rest of it - a real bug caught by
            # tests/transport/test5_concurrent.py's 64KB payloads (small
            # test1 messages never hit this, since they fit in one
            # congestion window).
            #
            # An earlier version of this drain waited for a PING sent after
            # the data to be acknowledged, on the theory that QUIC's
            # cumulative ACKs make that a barrier - that theory doesn't
            # hold: the PING is small and can be coalesced into the very
            # FIRST cwnd-limited flight alongside only the first chunk of a
            # large message, and get acknowledged after just one RTT while
            # the rest of the message is still sitting unsent (proven by a
            # ~3ms "acked" turnaround for a 64KB payload that should take
            # several cwnd-growth round trips). `stream.sender.buffer_is_empty`
            # is the actual right signal: aioquic sets it back to True only
            # once every byte written via send_stream_data has been handed
            # to a packet at least once (see QuicStreamSender.get_frame in
            # aioquic/quic/stream.py) - poll it, yielding to the loop so the
            # ACK-driven _transmit() calls elsewhere keep making progress,
            # until it's actually true.
            deadline = self._loop.time() + 10.0
            while self._loop.time() < deadline:
                stream = self._quic._streams.get(self._out_stream_id)
                if stream is None or stream.sender.buffer_is_empty:
                    break
                await asyncio.sleep(0.02)
        if self._quic is not None and self._protocol is not None and self._protocol.closed_exc is None:
            self._quic.close()
            self._protocol._transmit()
        if self._transport_obj is not None:
            self._transport_obj.close()

    def simulate_rebind(self, new_local_port: int = 0) -> None:
        """Test/diagnostic hook, NOT part of the `Transport` ABC: closes
        this side's current UDP socket and opens a fresh one (default
        port 0 = OS-assigned ephemeral), then resumes driving the SAME
        underlying `QuicConnection` - same keys, same connection ID, same
        in-flight stream state - over the new socket.

        This is deliberately what a real NAT remapping this process's
        outbound 4-tuple looks like from OUR side: closing/reopening the
        local socket is the only way to actually change a UDP socket's
        source port on Linux (you cannot rebind a live socket in place).
        Once packets start arriving at the peer from this new address, the
        peer's OWN `QuicConnection.receive_datagram` creates a new
        `QuicNetworkPath` for it and runs real PATH_CHALLENGE/PATH_RESPONSE
        validation (RFC 9000 §9, aioquic/quic/connection.py's
        `_find_network_path`/`_handle_path_challenge_frame` - entirely
        aioquic's own code, nothing added here) before trusting it. Used by
        tests/transport/test21_connection_migration.py to prove that
        mid-transfer NAT rebinding survives without a manual reconnect -
        the main reason this project chose QUIC over TCP hole punching in
        the first place (TCP dies outright on any 4-tuple change).
        """
        fut = asyncio.run_coroutine_threadsafe(self._simulate_rebind_async(new_local_port), self._loop)
        fut.result(timeout=10.0)

    async def _simulate_rebind_async(self, new_local_port: int) -> None:
        assert self._protocol is not None and self._quic is not None
        old_transport_obj = self._transport_obj
        old_protocol = self._protocol

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", new_local_port))

        loop = asyncio.get_running_loop()
        new_transport, new_protocol = await loop.create_datagram_endpoint(
            lambda: _QuicAdapterProtocol(
                old_protocol.peer_addr,
                self._recv_queue,
                old_protocol._max_message_bytes,
                old_protocol._is_client,
                old_protocol._quic_config,
            ),
            sock=sock,
        )
        # Already hole-punched, already mid-handshake-or-later - this is a
        # voluntary migration of an ESTABLISHED connection, not a fresh
        # connect(), so skip peer.py's own handshake state entirely and
        # reuse the SAME QuicConnection object as-is (attach_quic, not a
        # new QuicConnection - the whole point is continuity: same TLS
        # keys, same connection ID, same unacknowledged stream data).
        new_protocol.established = True
        new_protocol.handshake_waiter.set_result(None)
        new_protocol.attach_quic(self._quic)
        self._protocol = new_protocol
        self._transport_obj = new_transport
        if old_protocol._timer_handle is not None:
            # Otherwise this stale callback fires later against the OLD
            # protocol instance - same shared QuicConnection, but a dead
            # (closed below) transport to send() through, racing the NEW
            # protocol's own independent timer handling for that one
            # connection object.
            old_protocol._timer_handle.cancel()
        old_transport_obj.close()
        if self._keepalive_task is not None:
            # Recreated against the new protocol - the old one's task
            # closure still points at old_protocol, whose transport is now
            # closed, and would raise on its next scheduled send() otherwise.
            self._keepalive_task.cancel()
            self._keepalive_task = asyncio.create_task(self._keepalive_loop(new_protocol))
        # Force something out over the new socket now, rather than waiting
        # for the next natural send()/keepalive - makes the migration
        # observable deterministically for the test instead of racing an
        # arbitrary future event.
        new_protocol._transmit()
