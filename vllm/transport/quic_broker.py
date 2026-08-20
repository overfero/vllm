"""QuicBroker: one real QUIC connection per machine-pair, shared by several
logical channels (e.g. this project's per-TP-rank PP tensor links plus the
RPC control channel), instead of each channel hole-punching and QUIC-
handshaking its own separate connection.

Why this exists: `quic_transport.py`'s `QUICTransport` is deliberately
one-connection-per-`connect()`-call - correct and simple, but with this
project's real topology (TP=2 PP link + 1 RPC control channel between the
same two machines) that means 3 independent hole-punches, 3 independent
QUIC handshakes, and 3 independent congestion-control states between the
SAME two peers, none of them sharing an RTT estimate or a warmed-up
congestion window with the others. QUIC's own headline feature - many
streams multiplexed over one connection, no per-stream handshake or
congestion state - is exactly built for this case; this module is what
actually uses it for real, on top of the same hole-punch/QUIC machinery
`quic_transport.py` already proved correct, without touching that file at
all (existing, already-deployed callers of `QUICTransport` keep working
unmodified).

Design, two pieces:

1. `QuicBroker` - one per machine per peer. Does the SAME hole-punch +
   real QUIC handshake `QUICTransport` does (same `peer.py` hole-punch,
   same `aioquic` `QuicConnection` driven directly), but instead of moving
   bytes for a single anonymous direction, it multiplexes N independent
   *named channels* onto that one connection: each channel gets its own
   persistent unidirectional QUIC stream per direction (mirroring
   `quic_transport.py`'s "one persistent stream per direction, not one
   per message" design - see that module's docstring for why streams
   aren't reused per-message), opened lazily on that channel's first
   send. A new stream's very first bytes are a small self-describing
   preamble (2-byte length + UTF-8 channel name) so the receiving side
   learns "this raw QUIC stream_id belongs to channel X" without any
   prior coordination between the two brokers about stream-id numbering.
2. A tiny local IPC hop so the real, separate OS *processes* that need a
   channel (this project's per-local-rank PP workers, and the RPC control
   process) don't need to speak QUIC/aioquic themselves at all: the broker
   listens on a local Unix-domain socket per machine; a client connects,
   announces which channel it wants with one line, and from then on the
   local socket is just bytes-in/bytes-out for that channel, framed with
   the exact same 8-byte big-endian length prefix `quic_transport.py`
   already uses for the real QUIC stream - see `quic_multiplexed_transport.py`,
   which wraps this local-socket protocol behind the standard `Transport`
   ABC so it's a drop-in replacement anywhere `get_transport()` is called.

What this deliberately reuses unmodified from `quic_transport.py`/`peer.py`:
the hole-punch (`_hp.punch_loop`/`_hp.register`/`_hp.wait_for_peer`), the
`_QUIC_TAG` wire-prefix convention so broker traffic coexists on the same
UDP socket as peer.py's own ping/pong, the self-signed-cert-per-process TLS
setup, and the idle-timeout+keepalive-PING fix from that module. None of
that is copy-pasted blindly - `QuicBroker` drives its own `QuicConnection`
directly, independent of `QUICTransport`'s instance, so a bug in one can
never affect the other; only genuinely-identical, already-proven
low-level plumbing is duplicated.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import ssl
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aioquic.buffer import Buffer
from aioquic.quic import events as quic_events
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.connection import QuicConnection
from aioquic.quic.packet import pull_quic_header

from vllm.logger import init_logger

logger = init_logger(__name__)


def _locate_existing_transport_dir() -> Path:
    env_path = os.environ.get("VLLM_UDP_TRANSPORT_DIR")
    candidates = [Path(env_path)] if env_path else []
    candidates += [Path(__file__).resolve().parents[2] / "udp_holepunch"]
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

import peer as _hp  # noqa: E402  (unmodified - same hole-punch library quic_transport.py uses)

_QUIC_TAG = b"\x01"  # identical convention/value to quic_transport.py - see that
# module's docstring point 3 for why this exact byte never collides with
# peer.py's own tags or a real QUIC short-header first byte.
ALPN_PROTOCOL = "vllm-pp-broker-v1"  # distinct from quic_transport.py's own
# ALPN ("vllm-pp-v1") so a broker and a plain QUICTransport can never
# accidentally cross-handshake with each other on a misconfigured deployment.

DEFAULT_IDLE_TIMEOUT_S = float(os.environ.get("VLLM_TRANSPORT_QUIC_IDLE_TIMEOUT_S", "45"))
DEFAULT_MAX_MESSAGE_BYTES = int(
    os.environ.get("VLLM_TRANSPORT_QUIC_MAX_MESSAGE_BYTES", str(2 * 1024 * 1024 * 1024))
)
_KEEPALIVE_INTERVAL_SECONDS = 15.0

_LEN_PREFIX_BYTES = 8  # same big-endian message-length header quic_transport.py uses
_CHANNEL_NAME_LEN_BYTES = 2  # one-time preamble at the start of each stream

_LOCAL_PEER_GONE = object()  # sentinel - see QuicBroker.notify_local_peer_gone


def _generate_self_signed_cert():
    import datetime as _dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vllm-pp-broker")])
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


class _InboundChannelStream:
    """One raw QUIC stream's accumulation state. `channel` is None until
    the one-time name preamble has been fully read off the front of the
    stream - every stream starts in that state, since the receiving side
    has no other way to learn which channel a newly-opened stream belongs
    to (see module docstring)."""

    __slots__ = ("buf", "channel")

    def __init__(self) -> None:
        self.buf = bytearray()
        self.channel: str | None = None


class _BrokerQuicProtocol(_hp.PeerProtocol):
    """Multi-channel analogue of quic_transport.py's `_QuicAdapterProtocol`.
    Deliberately a separate class (not a subclass/import of that one) so
    this module can never regress the already-deployed `QUICTransport` -
    see module docstring."""

    def __init__(
        self,
        peer_addr: tuple[str, int],
        max_message_bytes: int,
        is_client: bool,
        quic_config: QuicConfiguration,
        on_channel_message,
    ) -> None:
        super().__init__(peer_addr)
        self._max_message_bytes = max_message_bytes
        self._is_client = is_client
        self._quic_config = quic_config
        self._on_channel_message = on_channel_message  # (channel: str, data: bytes) -> None
        self._quic: QuicConnection | None = None
        self._inbound: dict[int, _InboundChannelStream] = {}
        self._out_streams: dict[str, int] = {}
        self._timer_handle: asyncio.TimerHandle | None = None
        self._timer_at: float | None = None
        # Same lost-handshake-event race fix as quic_transport.py's
        # _QuicAdapterProtocol.handshake_waiter - created here, eagerly,
        # before any datagram can possibly arrive.
        self.handshake_waiter: asyncio.Future = self.loop.create_future()
        self.closed_exc: Exception | None = None

    def attach_quic(self, quic: QuicConnection) -> None:
        self._quic = quic

    def _handle_datagram(self, data: bytes, addr) -> None:
        super()._handle_datagram(data, addr)
        if not data or data[0:1] != _QUIC_TAG:
            return
        payload = data[1:]

        if self._quic is None:
            try:
                header = pull_quic_header(
                    Buffer(data=payload), host_cid_length=self._quic_config.connection_id_length
                )
            except ValueError:
                return
            self._quic = QuicConnection(
                configuration=self._quic_config, original_destination_connection_id=header.destination_cid
            )

        now = self.loop.time()
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
        assert self._quic is not None
        st = self._inbound.get(event.stream_id)
        if st is None:
            st = self._inbound[event.stream_id] = _InboundChannelStream()
        st.buf += event.data

        while True:
            if st.channel is None:
                if len(st.buf) < _CHANNEL_NAME_LEN_BYTES:
                    return
                name_len = int.from_bytes(st.buf[:_CHANNEL_NAME_LEN_BYTES], "big")
                if len(st.buf) < _CHANNEL_NAME_LEN_BYTES + name_len:
                    return
                name = bytes(st.buf[_CHANNEL_NAME_LEN_BYTES:_CHANNEL_NAME_LEN_BYTES + name_len]).decode("utf-8")
                del st.buf[: _CHANNEL_NAME_LEN_BYTES + name_len]
                st.channel = name
                continue  # more bytes (or a full message) may already be buffered

            if len(st.buf) < _LEN_PREFIX_BYTES:
                return
            msg_len = int.from_bytes(st.buf[:_LEN_PREFIX_BYTES], "big")
            if msg_len > self._max_message_bytes:
                del self._inbound[event.stream_id]
                self._quic.close(
                    error_code=1,
                    reason_phrase=f"message on stream {event.stream_id} (channel "
                    f"{st.channel!r}) exceeded {self._max_message_bytes} bytes",
                )
                self._transmit()
                return
            total_needed = _LEN_PREFIX_BYTES + msg_len
            if len(st.buf) < total_needed:
                return
            complete = bytes(st.buf[_LEN_PREFIX_BYTES:total_needed])
            del st.buf[:total_needed]
            self._on_channel_message(st.channel, complete)

    def _on_terminated(self, event: quic_events.ConnectionTerminated) -> None:
        self.closed_exc = ConnectionError(
            f"QuicBroker connection terminated: error_code={event.error_code} "
            f"reason={event.reason_phrase!r}"
        )
        if not self.handshake_waiter.done():
            self.handshake_waiter.set_exception(self.closed_exc)
        # Fan the poison out to every channel, not just one - a dead
        # connection must unblock every channel's recv(), not just
        # whichever one happens to be listening on the stream the
        # ConnectionTerminated event nominally arrived on.
        self._on_channel_message(None, self.closed_exc)  # type: ignore[arg-type]

    def open_channel_stream(self, channel: str) -> int:
        """Idempotent: returns the existing outbound stream id for this
        channel if one was already opened, matching quic_transport.py's
        "one persistent stream per direction for the connection's whole
        lifetime" design, just keyed per channel here instead of being
        the connection's only stream."""
        assert self._quic is not None
        stream_id = self._out_streams.get(channel)
        if stream_id is None:
            stream_id = self._quic.get_next_available_stream_id(is_unidirectional=True)
            self._out_streams[channel] = stream_id
            name_bytes = channel.encode("utf-8")
            preamble = len(name_bytes).to_bytes(_CHANNEL_NAME_LEN_BYTES, "big") + name_bytes
            self._quic.send_stream_data(stream_id, preamble, end_stream=False)
        return stream_id

    def send_on_channel(self, channel: str, data: bytes) -> None:
        assert self._quic is not None
        stream_id = self.open_channel_stream(channel)
        header = len(data).to_bytes(_LEN_PREFIX_BYTES, "big")
        self._quic.send_stream_data(stream_id, header + data, end_stream=False)
        self._transmit()

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


class QuicBroker:
    """One shared QUIC connection to one peer machine's broker, fanning
    in/out to any number of named local channels. See module docstring."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._protocol: _BrokerQuicProtocol | None = None
        self._transport_obj: asyncio.DatagramTransport | None = None
        self._quic: QuicConnection | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._self_id = "?"
        self._peer_id = "?"
        self._channel_queues: dict[str, "__import__('queue').Queue"] = {}
        self._channel_queues_lock = threading.Lock()
        self._closed_exc: Exception | None = None

    # ---- connection setup (same hole-punch + QUIC handshake sequence as
    # QUICTransport._connect_async - see that method for line-by-line
    # rationale on each step; not re-explained here) ----

    def connect(
        self,
        *,
        self_id: str,
        peer_id: str,
        signaling_url: str,
        udp_port: int,
        listen: bool,
        connect_timeout: float = 120.0,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT_S,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self._self_id = self_id
        self._peer_id = peer_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(
            self._connect_async(
                self_id=self_id, peer_id=peer_id, signaling_url=signaling_url, udp_port=udp_port,
                listen=listen, connect_timeout=connect_timeout, idle_timeout=idle_timeout,
                max_message_bytes=max_message_bytes,
            ),
            self._loop,
        )
        fut.result(timeout=connect_timeout + 5)

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _deliver(self, channel: str | None, data) -> None:
        """Callback from `_BrokerQuicProtocol`, runs on the broker's own
        event-loop thread. `channel is None` is the ConnectionTerminated
        poison broadcast (see `_on_terminated`) - fan it out to every
        channel queue that exists so far, and remember it so any channel
        opened *after* the connection died also gets it immediately
        instead of hanging."""
        with self._channel_queues_lock:
            if channel is None:
                self._closed_exc = data
                for q in self._channel_queues.values():
                    q.put_nowait(data)
                return
            q = self._channel_queues.get(channel)
            if q is None:
                import queue as _queue

                q = self._channel_queues[channel] = _queue.Queue()
            q.put_nowait(data)

    def _queue_for(self, channel: str):
        with self._channel_queues_lock:
            q = self._channel_queues.get(channel)
            if q is None:
                import queue as _queue

                q = self._channel_queues[channel] = _queue.Queue()
                if self._closed_exc is not None:
                    q.put_nowait(self._closed_exc)
            return q

    async def _connect_async(
        self, *, self_id: str, peer_id: str, signaling_url: str, udp_port: int, listen: bool,
        connect_timeout: float, idle_timeout: float, max_message_bytes: int,
    ) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, opt, _hp.SOCKET_BUFFER_REQUEST)
        sock.bind(("0.0.0.0", udp_port))

        own_ip, own_port = None, sock.getsockname()[1]
        reg_resp = _hp.register(signaling_url, self_id, own_port)
        own_ip = own_ip or reg_resp["public_ip"]

        peer_info = _hp.wait_for_peer(signaling_url, self_id, peer_id)
        coordinator_peer_addr = (peer_info["public_ip"], peer_info["udp_port"])
        delay = peer_info["start_at"] - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        is_client = not listen
        quic_config = QuicConfiguration(alpn_protocols=[ALPN_PROTOCOL], is_client=is_client, idle_timeout=idle_timeout)
        _qlog_dir = os.environ.get("VLLM_TRANSPORT_QUIC_BROKER_QLOG_DIR")
        if _qlog_dir:
            from aioquic.quic.logger import QuicFileLogger

            os.makedirs(_qlog_dir, exist_ok=True)
            quic_config.quic_logger = QuicFileLogger(_qlog_dir)
        if is_client:
            quic_config.verify_mode = ssl.CERT_NONE
        else:
            cert, key = _generate_self_signed_cert()
            quic_config.certificate = cert
            quic_config.private_key = key

        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _BrokerQuicProtocol(
                coordinator_peer_addr, max_message_bytes, is_client, quic_config, self._deliver
            ),
            sock=sock,
        )

        punch_task = asyncio.create_task(_hp.punch_loop(protocol))
        deadline = time.monotonic() + connect_timeout
        while not protocol.established and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        punch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await punch_task

        if not protocol.established:
            transport.close()
            raise ConnectionError(f"QuicBroker: UDP hole punch failed within {connect_timeout}s")

        self._protocol = protocol
        self._transport_obj = transport
        handshake_waiter = protocol.handshake_waiter

        if is_client:
            quic = QuicConnection(configuration=quic_config)
            protocol.attach_quic(quic)
            self._quic = quic
            quic.connect(protocol.peer_addr, now=loop.time())
            protocol._transmit()

        try:
            await asyncio.wait_for(handshake_waiter, timeout=connect_timeout)
        except asyncio.TimeoutError:
            transport.close()
            raise ConnectionError(
                f"QuicBroker: QUIC handshake did not complete within {connect_timeout}s"
            ) from None

        self._quic = protocol._quic
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(protocol))
        logger.info("QuicBroker: connection %s <-> %s established", self_id, peer_id)

    async def _keepalive_loop(self, protocol: "_BrokerQuicProtocol") -> None:
        # Same two-concern keepalive as quic_transport.py's own loop (NAT
        # pinhole freshness + QUIC's own idle_timeout clock) - see that
        # module's _keepalive_loop docstring for the full rationale.
        seq = 0
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL_SECONDS)
            seq += 1
            protocol.send(_hp.pack_ping(seq, time.monotonic()))
            if protocol._quic is not None and protocol.closed_exc is None:
                protocol._quic.send_ping(seq)
                protocol._transmit()

    # ---- channel-level send/recv, callable from ANY thread (submits
    # onto the broker's own event loop) ----

    def send_on_channel(self, channel: str, data: bytes) -> None:
        if self._loop is None or self._protocol is None:
            raise RuntimeError("send_on_channel() called before connect()")
        if self._protocol.closed_exc is not None:
            raise self._protocol.closed_exc
        fut = asyncio.run_coroutine_threadsafe(
            self._send_on_channel_async(channel, data), self._loop
        )
        fut.result()

    async def _send_on_channel_async(self, channel: str, data: bytes) -> None:
        assert self._protocol is not None
        self._protocol.send_on_channel(channel, data)

    def notify_local_peer_gone(self, channel: str) -> None:
        """Wakes up a `recv_from_channel(channel, timeout=None)` call
        that's blocked waiting for the next message, once this channel's
        LOCAL peer (the real process on this machine using this channel)
        has disconnected - see `_handle_local_client`'s docstring for the
        real bug this fixes (a channel whose local peer left otherwise
        left `recv_from_channel` blocked forever, permanently occupying
        an executor thread). Goes through the SAME FIFO queue
        `recv_from_channel` reads from, so it's delivered strictly after
        every real message already queued for this channel - nothing
        already in flight is skipped or raced."""
        with self._channel_queues_lock:
            q = self._channel_queues.get(channel)
            if q is None:
                import queue as _queue

                q = self._channel_queues[channel] = _queue.Queue()
            q.put_nowait(_LOCAL_PEER_GONE)

    def recv_from_channel(self, channel: str, timeout: float | None = None) -> bytes:
        q = self._queue_for(channel)
        try:
            data = q.get(timeout=timeout)
        except Exception as exc:  # queue.Empty
            raise TimeoutError(f"recv_from_channel({channel!r}) timed out after {timeout}s") from exc
        if isinstance(data, Exception):
            q.put_nowait(data)  # see quic_transport.py's recv() for why this is put back
            raise data
        return data

    def close(self) -> None:
        if self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self._close_async(), self._loop)
            with contextlib.suppress(Exception):
                fut.result(timeout=11.0)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    async def _close_async(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
        if self._quic is not None and self._protocol is not None and self._protocol.closed_exc is None:
            # Drain every open channel stream's send buffer before
            # closing. Real, qlog-confirmed bug hit building this:
            # `stream.sender.buffer_is_empty` (what quic_transport.py's
            # own single-stream close() waits for) is True the moment
            # every byte has been handed to a packet *at least once* -
            # aioquic only flips it back to False later, if and when its
            # OWN loss-detection actually notices that data went missing
            # and needs retransmitting (see QuicStreamSender.on_data_delivery
            # in aioquic's stream.py). Checking it immediately after a
            # burst of sends - exactly what closing right after finishing
            # a batch of work does - reads "empty" for data that has only
            # ever been sent ONCE and never actually confirmed, because no
            # loss has been *detected* yet (detection itself takes time:
            # an ACK gap or an RTO). Closed the connection at that point
            # and a qlog trace showed the real failure mode directly: the
            # peer's CONNECTION_CLOSE frame arrives right after its last
            # good STREAM frame, with zero `recovery:packet_lost` events
            # ever recorded - not because nothing was lost, but because
            # the connection ended before aioquic's own loss timer got a
            # chance to fire and retransmit. `sender._buffer_start`, by
            # contrast, only advances when `on_data_delivery` observes a
            # real ACK for that range (see the ACKED branch) - comparing
            # it to `highest_offset` is genuine "every byte I ever sent on
            # this stream has been acknowledged by the peer", not just
            # "nothing is currently queued to retransmit".
            deadline = self._loop.time() + 10.0
            while self._loop.time() < deadline:
                pending = [
                    sid for sid in self._protocol._out_streams.values()
                    if (s := self._quic._streams.get(sid)) is not None
                    and s.sender._buffer_start < s.sender.highest_offset
                ]
                if not pending:
                    break
                await asyncio.sleep(0.02)
            self._quic.close()
            self._protocol._transmit()
        if self._transport_obj is not None:
            self._transport_obj.close()


# ---- local Unix-domain-socket IPC hop: the real per-process client side
# lives in quic_multiplexed_transport.py; this is the broker-side listener
# it connects to. ----

_LOCAL_LEN_PREFIX = struct.Struct("!Q")  # 8-byte length prefix, local-socket messages


def broker_socket_path(self_id: str, peer_id: str) -> str:
    """Deterministic path so a client process (which only knows its own
    self_id/peer_id, not the broker's internal state) can find the right
    socket. Keyed by BOTH names, not just self_id, so a middle-of-chain
    PP stage (two distinct peers - the previous stage and the next one,
    each needing its own separate QuicBroker connection) gets two
    non-colliding sockets rather than one broker's channels silently
    overwriting the other's. Under /tmp (not /kaggle/working - this is
    ephemeral IPC plumbing, not output - see the project's own placement
    convention)."""
    def _safe(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

    return f"/tmp/vllm_quic_broker_{_safe(self_id)}__{_safe(peer_id)}.sock"


async def _handle_local_client(
    broker: QuicBroker, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
) -> None:
    channel = "?"
    try:
        name_line = await reader.readline()
        channel = name_line.decode("utf-8").strip()
        if not channel:
            writer.close()
            return

        # Real bug hit building this: `_pump_from_quic` originally blocked
        # *forever* on `recv_from_channel(channel, None)` (no timeout)
        # waiting for a message that would never arrive once the local
        # peer had gone away, since nothing tied its lifetime to
        # `_pump_to_quic` hitting EOF. With several channels sharing this
        # loop's default executor (`run_in_executor(None, ...)`, a fixed-
        # size thread pool), each channel whose local peer disconnected
        # first left its `_pump_from_quic` permanently camped on one pool
        # thread - observed directly stealing worker threads from a
        # DIFFERENT, still-active channel's `recv_from_channel` submission
        # once the pool was exhausted, so that channel's `recv()` timed
        # out despite the QUIC connection itself having delivered every
        # byte correctly (confirmed by direct inspection of the
        # QuicConnection stream state at the moment of failure:
        # `stream.sender.buffer_is_empty` was True - no data was ever
        # actually lost at the QUIC layer, only stuck downstream of it).
        #
        # Two fix attempts that were themselves real bugs, in order:
        #  1. Cancel `_pump_from_quic` via `asyncio.wait(FIRST_COMPLETED)`
        #     the moment `_pump_to_quic` hit EOF -
        #     `loop.run_in_executor()`'s cancellation does NOT stop the
        #     underlying thread-pool worker once it's already running the
        #     blocking call (a well-known asyncio/concurrent.futures
        #     limitation) - it only detaches the awaiting Task, so the
        #     in-flight `recv_from_channel` kept running in the
        #     background and could still pull a REAL, not-yet-delivered
        #     message out of the queue with nothing left listening for
        #     the result - silent message loss, worse than the leak.
        #  2. Poll `recv_from_channel` with a short timeout instead of
        #     blocking indefinitely, checking a "peer gone" flag between
        #     polls - avoided the cancellation bug, but resubmitting a
        #     fresh `run_in_executor` call every ~0.5s for every channel
        #     turned out to add enough scheduling/thread-pool churn under
        #     3 concurrent channels to make throughput WORSE across the
        #     board (measured: previously-passing channels started
        #     failing too).
        # The actual fix: never poll AND never force-cancel. `_pump_to_quic`,
        # on EOF, pushes a sentinel through the SAME FIFO queue
        # `recv_from_channel` already reads from (`notify_local_peer_gone`) -
        # `_pump_from_quic`'s blocking `q.get()` wakes up naturally and
        # immediately, in queue order (so every real message already
        # queued ahead of the sentinel is still delivered first), with no
        # polling overhead and no possibility of racing a still-running
        # blocking call.
        async def _pump_to_quic() -> None:
            try:
                while True:
                    header = await reader.readexactly(_LOCAL_LEN_PREFIX.size)
                    (n,) = _LOCAL_LEN_PREFIX.unpack(header)
                    payload = await reader.readexactly(n)
                    await asyncio.get_running_loop().run_in_executor(
                        None, broker.send_on_channel, channel, payload
                    )
            finally:
                await asyncio.get_running_loop().run_in_executor(
                    None, broker.notify_local_peer_gone, channel
                )

        async def _pump_from_quic() -> None:
            loop = asyncio.get_running_loop()
            while True:
                data = await loop.run_in_executor(
                    None, broker.recv_from_channel, channel, None
                )
                if data is _LOCAL_PEER_GONE:
                    return
                writer.write(_LOCAL_LEN_PREFIX.pack(len(data)) + data)
                await writer.drain()

        results = await asyncio.gather(_pump_to_quic(), _pump_from_quic(), return_exceptions=True)
        for r in results:
            if isinstance(r, (asyncio.IncompleteReadError, ConnectionError)):
                continue  # expected: local peer closed its side, or the QUIC link died
            if isinstance(r, BaseException):
                logger.warning(
                    "QuicBroker local channel %r: pump task failed unexpectedly: %r", channel, r
                )
    finally:
        with contextlib.suppress(Exception):
            writer.close()


def serve_local_broker(broker: QuicBroker, socket_path: str) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Starts the local Unix-socket listener that fans channels out to
    real client processes, on its own event loop/thread (separate from
    the broker's own QUIC event loop - keeps a slow/blocked local client
    from ever delaying QUIC protocol processing, and vice versa)."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(socket_path)

    ready = threading.Event()
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Real bug hit building this: the default `run_in_executor(None, ...)`
        # pool size is `min(32, cpu_count + 4)` - on a 4-CPU box, 8 threads.
        # `_pump_from_quic` occupies one thread for as long as it's
        # genuinely waiting on the next message (not indefinitely - see
        # that function's docstring for the sentinel-based shutdown fix -
        # but for as long as a channel is legitimately idle between
        # messages, which under bursty concurrent traffic across several
        # channels adds up). With 3 channels each needing up to 2 threads
        # at once (one `_pump_from_quic` wait + one transient
        # `_pump_to_quic`/`ensure_channel_open` call), demand can reach 6
        # concurrent threads - close enough to an 8-thread default pool,
        # SHARED with whatever else this process's default executor
        # serves, that real submissions measurably queued waiting for a
        # free worker instead of running immediately, one channel then
        # missing its recv() deadline even though the peer had already
        # delivered its data (confirmed via directly reading
        # `_channel_queues[channel].qsize()` at the moment of failure -
        # items were sitting in the queue, undelivered, with no exception
        # anywhere - a thread-starvation stall, not a crash or a protocol
        # bug). A dedicated, generously-sized executor for this loop only
        # (never shared with unrelated default-executor consumers
        # elsewhere in the process) removes the contention at its source.
        loop.set_default_executor(ThreadPoolExecutor(max_workers=256, thread_name_prefix="quic-broker-local"))
        loop_holder["loop"] = loop

        async def _main() -> None:
            server = await asyncio.start_unix_server(
                lambda r, w: _handle_local_client(broker, r, w), path=socket_path
            )
            ready.set()
            async with server:
                await server.serve_forever()

        loop.run_until_complete(_main())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    ready.wait(timeout=10.0)
    return loop_holder["loop"], thread
