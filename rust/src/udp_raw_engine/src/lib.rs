// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Raw UDP transport core, built after profiling this project's existing
//! `udp_transport.py` and `quic_rs_transport.py` and finding both
//! dominated by per-packet userspace/Python dispatch overhead - see
//! `vllm/transport/udp_rs_raw_bench.py`'s module docstring for the full
//! story and real throughput numbers. Started as a deliberately
//! *unreliable* experiment (find the real throughput ceiling first,
//! decide whether reliability on top is worth its cost second) - the
//! `send_reliable`/`recv_reliable` pair is now genuinely reliable (real
//! retransmission on loss, see `send_reliable`'s docstring), having
//! answered that question: yes, worth it, the retransmit logic added
//! negligible measured overhead (see the `project_raw_udp_rs_bench`
//! memory entry / commit history for the before/after numbers).
//! `send_reliable_gso`/`recv_reliable_gro` (the GSO/GRO-batched
//! alternative - measured slower than the plain `sendmmsg`-based pair,
//! see the same memory entry) have NOT had the same retransmission logic
//! added and remain unreliable - `send_batch`/`recv_batch` themselves
//! are also still raw, ordering/retry-free primitives, same as always.
//!
//! Uses `sendmmsg(2)`/`recvmmsg(2)` (Linux batched datagram syscalls -
//! send/receive MANY datagrams in ONE syscall, the direct UDP analogue of
//! the GSO/GRO batching `quic_rs_transport.py` already uses for QUIC) to
//! minimize syscalls-per-byte, and lets the caller pick a large chunk size
//! (loopback's MTU is ~65536 bytes, dramatically bigger than a real
//! network path's ~1200-1500 - see `chunk_size`'s docstring) instead of
//! being fixed to a real-world-safe MTU.
//!
//! No socket creation/binding logic lives here - `RawUdpEngine::new` takes
//! an already-bound, already-`connect()`-ed `std::net::UdpSocket` (matching
//! this project's other engines' "no socket ownership" convention) so the
//! Python side keeps doing hole-punch/binding exactly like it already does
//! for every other backend.

pub mod error;

use std::net::UdpSocket;
use std::os::unix::io::AsRawFd;

pub use error::{EngineError, Result};

/// One raw UDP engine bound to a single already-connected socket (see
/// module docstring for why `connect()` happens on the Python side, not
/// here). `connect()`-ing means every `sendmmsg`/`recvmmsg` call here can
/// omit the per-message peer address entirely (`msg_name = NULL`) - the
/// kernel already filters inbound datagrams to the connected peer and
/// fills in the right destination for outbound ones.
pub struct RawUdpEngine {
    socket: UdpSocket,
    /// Reused across `recv_batch`/`recv_gro` calls instead of freshly
    /// allocating+zero-initializing every time - real, measured bug found
    /// by re-reading a Rust performance discussion the user pointed at
    /// (users.rust-lang.org "Rust vs. C vs. Go runtime speed comparison"):
    /// its core lesson was that the ORIGINAL 16s-vs-1.5s gap there was not
    /// a language/compiler problem at all, but an accidentally quadratic
    /// algorithm hiding underneath - which prompted re-checking THIS
    /// crate for the same class of self-inflicted waste, not just tuning
    /// the syscall choice (sendmmsg vs GSO) further. Found: the ACK-
    /// polling loop inside `send_reliable`/`send_reliable_gso` calls
    /// `recv_batch`/`recv_gro` once per round (~90-260 times for a 16MB
    /// transfer) purely to check for a 5-byte ACK that is usually not
    /// even there yet - and each of those calls was allocating AND
    /// zero-filling up to `batch * max_msg_len` bytes (~184KB) of fresh
    /// memory only to discard it immediately. Over a whole transfer that
    /// is tens of MB of pure allocator/zeroing churn that has nothing to
    /// do with the actual data being moved. Not algorithmically quadratic
    /// in the strict sense the linked thread's bug was, but the same
    /// underlying mistake: optimizing the syscall itself while missing
    /// real, avoidable work happening around it every single round.
    scratch_recv: std::sync::Mutex<Vec<Vec<u8>>>,
    /// Same reuse-not-reallocate reasoning as `scratch_recv`, for
    /// `recv_gro`'s single buffer instead of `recv_batch`'s array of them.
    scratch_gro: std::sync::Mutex<Vec<u8>>,
    /// Real bug found testing `send_message`/`recv_message` back-to-back
    /// (a ping-pong pattern - send a message, then immediately receive
    /// the peer's reply) - see `poll_acks`/`poll_data`'s docstring for
    /// the full mechanism. Holds packets read off the wire by ONE poll
    /// call (e.g. `send_message`'s ack-scan) that turned out to belong
    /// to the OTHER kind (a `TYPE_DATA` packet arriving in the same
    /// `recvmmsg` batch as the ack it was actually waiting for) - without
    /// this, such a packet was silently discarded (recv_batch drains it
    /// off the kernel queue either way), forcing the sender to fall back
    /// on a real ~20ms retransmit-timeout cycle to recover instead of
    /// just using the copy it already has in hand.
    pending_acks: std::sync::Mutex<Vec<Vec<u8>>>,
    /// Same mechanism as `pending_acks`, for stray `TYPE_ACK` packets
    /// encountered while polling for data.
    pending_data: std::sync::Mutex<Vec<Vec<u8>>>,
    /// Auto-incrementing id `send_message` tags every chunk of ONE
    /// message with - see `InboundMsgState`'s docstring for why this
    /// exists.
    next_msg_id: std::sync::atomic::AtomicU32,
    /// Real bug found testing many `send_message`/`recv_message` calls
    /// back-to-back with NO pause between them (a streaming pattern -
    /// this project's own `test8_tensor_streaming.py`-style workload):
    /// `send_message` for message N can return (its final ack seen) and
    /// `send_message` for message N+1 can start emitting chunks BEFORE
    /// the RECEIVER's `recv_message` call for message N has actually
    /// returned control to its Python caller and been called again for
    /// message N+1 - the kernel socket buffer happily queues N+1's early
    /// chunks in the meantime. Without a `msg_id`, those early chunks
    /// reuse the SAME sequence numbers (0, 1, 2, ...) message N used, so
    /// if one coincides with a sequence number message N hadn't received
    /// yet, `recv_message`'s old call-scoped logic would silently treat
    /// it as message N's data and write message N+1's bytes into message
    /// N's buffer - confirmed directly: a 300-message randomized-size
    /// streaming test produced a byte-length mismatch on message 11
    /// before this fix (received length landed between two of the
    /// surrounding messages' real sizes, exactly the signature of this
    /// cross-message splice). Every `send_message`-built chunk now
    /// carries a `msg_id` (see `MSG_DATA_HEADER_BYTES`/
    /// `MSG_ACK_HEADER_BYTES`), and `inbound`/`completed_order` give
    /// `recv_message` PERSISTENT (not call-scoped) per-message state, so
    /// out-of-turn early chunks land in their own message's bucket
    /// instead of corrupting whichever message happens to be
    /// call-scoped-"current" at that instant.
    inbound: std::sync::Mutex<std::collections::HashMap<u32, InboundMsgState>>,
    /// `msg_id`s whose message has fully arrived but not yet been
    /// returned to the Python caller (can happen if, e.g., message N+1's
    /// last chunk happens to complete it while a `recv_message` call is
    /// still draining the same batch that also finished off message N) -
    /// `recv_message` always drains this FIFO before doing any new
    /// polling, so completed-but-unclaimed messages are never dropped or
    /// returned out of order.
    completed_order: std::sync::Mutex<std::collections::VecDeque<u32>>,
}

/// Per-message receive state for `recv_message`, keyed by `msg_id` in
/// `RawUdpEngine::inbound` - see that field's docstring for why this
/// needs to be persistent across `recv_message` calls, not local to one.
struct InboundMsgState {
    out: Vec<u8>,
    received: Vec<bool>,
    received_bytes: usize,
    next_expected: usize,
    highest_seq: i64,
    max_gap: i64,
    chunks_received: usize,
    chunks_since_ack: usize,
    last_ack_resend_at: std::time::Instant,
}

impl RawUdpEngine {
    pub fn new(socket: UdpSocket) -> Result<Self> {
        socket
            .set_nonblocking(true)
            .map_err(|error| error::setup!("failed to set socket non-blocking: {error}"))?;
        Ok(Self {
            socket,
            scratch_recv: std::sync::Mutex::new(Vec::new()),
            scratch_gro: std::sync::Mutex::new(Vec::new()),
            pending_acks: std::sync::Mutex::new(Vec::new()),
            pending_data: std::sync::Mutex::new(Vec::new()),
            next_msg_id: std::sync::atomic::AtomicU32::new(0),
            inbound: std::sync::Mutex::new(std::collections::HashMap::new()),
            completed_order: std::sync::Mutex::new(std::collections::VecDeque::new()),
        })
    }

    /// Polls for `TYPE_ACK` packets: first returns anything already
    /// stashed by a PRIOR `poll_data` call that turned out to be an ack
    /// (see `pending_acks`'s docstring), only falling through to a real
    /// `recv_batch` syscall if that stash is empty. Any `TYPE_DATA`
    /// packet seen along the way is stashed in `pending_data` for a
    /// future `poll_data` call to pick up, instead of being discarded.
    fn poll_acks(&self, batch: usize, max_msg_len: usize) -> Result<Vec<Vec<u8>>> {
        {
            let mut stashed = self.pending_acks.lock().unwrap();
            if !stashed.is_empty() {
                return Ok(std::mem::take(&mut stashed));
            }
        }
        let raw = self.recv_batch(batch, max_msg_len)?;
        let mut acks = Vec::with_capacity(raw.len());
        let mut stray_data = Vec::new();
        for pkt in raw {
            match pkt.first() {
                Some(&TYPE_ACK) => acks.push(pkt),
                Some(&TYPE_DATA) => stray_data.push(pkt),
                _ => {} // unrecognized tag - genuinely safe to discard
            }
        }
        if !stray_data.is_empty() {
            self.pending_data.lock().unwrap().extend(stray_data);
        }
        Ok(acks)
    }

    /// Symmetric counterpart to `poll_acks` - see its docstring, same
    /// mechanism with data/ack swapped.
    fn poll_data(&self, batch: usize, max_msg_len: usize) -> Result<Vec<Vec<u8>>> {
        {
            let mut stashed = self.pending_data.lock().unwrap();
            if !stashed.is_empty() {
                return Ok(std::mem::take(&mut stashed));
            }
        }
        let raw = self.recv_batch(batch, max_msg_len)?;
        let mut data = Vec::with_capacity(raw.len());
        let mut stray_acks = Vec::new();
        for pkt in raw {
            match pkt.first() {
                Some(&TYPE_DATA) => data.push(pkt),
                Some(&TYPE_ACK) => stray_acks.push(pkt),
                _ => {}
            }
        }
        if !stray_acks.is_empty() {
            self.pending_acks.lock().unwrap().extend(stray_acks);
        }
        Ok(data)
    }

    /// Blocks (efficiently - a real `poll(2)` wait, not a busy loop) until
    /// the socket has something to read or `timeout` elapses, whichever
    /// first. Returns `true` if genuinely readable, `false` on timeout.
    ///
    /// Real latency bug found and fixed by this: every "nothing to do
    /// yet, must wait" branch in `send_reliable`/`recv_reliable`/
    /// `send_message`/`recv_message` used to call
    /// `std::thread::sleep(Duration::from_micros(200))` unconditionally
    /// before re-checking the (non-blocking) socket - a fixed polling
    /// granularity that adds up to 200us of pure sleep latency PER wait,
    /// regardless of how quickly the peer's reply actually arrives. A
    /// tiny ping-pong round trip touches several of these waits in
    /// sequence (sender's own ACK-wait, receiver's data-wait, the
    /// reply's ACK-wait, the original sender's reply-wait) - confirmed
    /// directly: this made average round-trip latency ~560-620us,
    /// noticeably worse than this project's own `quic-rs` backend
    /// (~290-420us) despite QUIC having MORE per-packet protocol
    /// overhead (crypto, ACK frames) than this engine's much simpler
    /// scheme - the gap traced entirely to `quic`'s driver using a real
    /// blocking `recv_from()` with a computed timeout (OS wakes the
    /// thread up the INSTANT data arrives) versus this engine's fixed
    /// 200us polling floor. `wait_readable` closes that gap the same
    /// way: block on `poll(2)` for up to `timeout`, but return
    /// IMMEDIATELY the moment the kernel reports data waiting, instead
    /// of on some fixed cadence. Deliberately only used in genuine
    /// "must wait" branches (window full, waiting for the final ack,
    /// receiver has nothing new) - the fast "window open, keep sending"
    /// throughput path never calls this and is unaffected by it, so
    /// this closes the latency gap without touching the sendmmsg/
    /// recvmmsg batching throughput numbers at all.
    fn wait_readable(&self, timeout: std::time::Duration) -> bool {
        let mut pfd = libc::pollfd {
            fd: self.socket.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let timeout_ms = timeout.as_millis().min(libc::c_int::MAX as u128) as libc::c_int;
        // SAFETY: `pfd` is a single, valid, stack-local `pollfd` with a
        // real open fd (`self.socket`, guaranteed alive for at least as
        // long as `&self` here) - a standard, well-defined libc syscall
        // wrapper with no invariants beyond that.
        let ret = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        ret > 0 && (pfd.revents & libc::POLLIN) != 0
    }

    /// Sends every buffer in `chunks` as one `sendmmsg(2)` call (a single
    /// syscall for the WHOLE batch, not one per chunk) - each `Vec<u8>` in
    /// `chunks` becomes exactly one UDP datagram. Returns how many of them
    /// the kernel actually accepted (`sendmmsg` can do a genuine partial
    /// send, e.g. if the socket send buffer fills up mid-batch - the
    /// caller is responsible for retrying the remainder, mirroring
    /// `quic_rs_transport.py`'s `send_message` retry-loop contract for the
    /// same reason: never silently drop what the caller asked to send).
    pub fn send_batch(&self, chunks: &[Vec<u8>]) -> Result<usize> {
        if chunks.is_empty() {
            return Ok(0);
        }
        let mut iovecs: Vec<libc::iovec> = chunks
            .iter()
            .map(|chunk| libc::iovec {
                iov_base: chunk.as_ptr() as *mut libc::c_void,
                iov_len: chunk.len(),
            })
            .collect();
        let mut msgs: Vec<libc::mmsghdr> = iovecs
            .iter_mut()
            .map(|iov| libc::mmsghdr {
                msg_hdr: libc::msghdr {
                    msg_name: std::ptr::null_mut(),
                    msg_namelen: 0,
                    msg_iov: iov as *mut libc::iovec,
                    msg_iovlen: 1,
                    msg_control: std::ptr::null_mut(),
                    msg_controllen: 0,
                    msg_flags: 0,
                },
                msg_len: 0,
            })
            .collect();

        // SAFETY: `msgs`/`iovecs` are valid for the duration of this one
        // call (both live until this function returns), each `iovec`
        // points at a `chunks[i]` buffer that also outlives the call
        // (borrowed via `&[Vec<u8>]`, not moved), and `msgs.len()` fits in
        // `libc::c_uint` for any batch size this crate's callers use (the
        // Python side bounds batch size well below u32::MAX - see
        // `udp_rs_transport.py`).
        let sent = unsafe {
            libc::sendmmsg(
                self.socket.as_raw_fd(),
                msgs.as_mut_ptr(),
                msgs.len() as libc::c_uint,
                0,
            )
        };
        if sent < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(0);
            }
            return Err(error::io!("sendmmsg failed: {err}"));
        }
        Ok(sent as usize)
    }

    /// Receives up to `max_batch` datagrams in ONE `recvmmsg(2)` call, each
    /// truncated to `max_msg_len` bytes if larger (matches `recv(2)`'s own
    /// truncation behavior for an oversized UDP datagram - silently
    /// dropping the excess, not an error, since UDP has no way to deliver
    /// "the rest" of a datagram later anyway). Returns an empty `Vec` if
    /// nothing is available right now (non-blocking) - not an error.
    pub fn recv_batch(&self, max_batch: usize, max_msg_len: usize) -> Result<Vec<Vec<u8>>> {
        if max_batch == 0 {
            return Ok(Vec::new());
        }
        // Reused, not freshly allocated - see `scratch_recv`'s field
        // docstring for the real, measured waste this replaces. `resize`
        // is a no-op realloc once each buffer has already grown to
        // `max_msg_len` on a prior call (this crate's callers always pass
        // the same `max_msg_len` for a given engine's whole lifetime), and
        // `[0u8; N]`-filling on resize-growth only actually costs anything
        // the very first time a given slot grows past its previous size.
        let mut bufs_guard = self.scratch_recv.lock().unwrap();
        while bufs_guard.len() < max_batch {
            bufs_guard.push(Vec::new());
        }
        for buf in bufs_guard.iter_mut().take(max_batch) {
            buf.clear();
            buf.resize(max_msg_len, 0);
        }
        let bufs = &mut bufs_guard[..max_batch];
        let mut iovecs: Vec<libc::iovec> = bufs
            .iter_mut()
            .map(|buf| libc::iovec {
                iov_base: buf.as_mut_ptr() as *mut libc::c_void,
                iov_len: buf.len(),
            })
            .collect();
        let mut msgs: Vec<libc::mmsghdr> = iovecs
            .iter_mut()
            .map(|iov| libc::mmsghdr {
                msg_hdr: libc::msghdr {
                    msg_name: std::ptr::null_mut(),
                    msg_namelen: 0,
                    msg_iov: iov as *mut libc::iovec,
                    msg_iovlen: 1,
                    msg_control: std::ptr::null_mut(),
                    msg_controllen: 0,
                    msg_flags: 0,
                },
                msg_len: 0,
            })
            .collect();

        // SAFETY: same reasoning as `send_batch` - `msgs`/`iovecs`/`bufs`
        // all outlive this call, and each `iovec`'s `iov_len` matches the
        // real allocated size of its backing `Vec<u8>`, so the kernel
        // never writes past what each buffer actually owns. `timeout =
        // NULL` with the socket already `O_NONBLOCK` (set in `new`) means
        // this returns immediately if nothing is queued, rather than the
        // kernel needing a real timeout value to avoid blocking.
        let received = unsafe {
            libc::recvmmsg(
                self.socket.as_raw_fd(),
                msgs.as_mut_ptr(),
                msgs.len() as libc::c_uint,
                0,
                std::ptr::null_mut::<libc::timespec>(),
            )
        };
        if received < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(Vec::new());
            }
            return Err(error::io!("recvmmsg failed: {err}"));
        }
        let received = received as usize;
        // Copies only the bytes actually received out of the reused
        // scratch buffers into fresh, right-sized owned `Vec<u8>`s for the
        // caller - `bufs` itself stays borrowed from `scratch_recv` and
        // gets reused as-is (already the right capacity) on the next call.
        let mut out = Vec::with_capacity(received);
        for (i, buf) in bufs.iter().take(received).enumerate() {
            let len = msgs[i].msg_len as usize;
            out.push(buf[..len].to_vec());
        }
        Ok(out)
    }

    /// Sends the WHOLE of `data`, chunked/batched/ACK-paced, in one call -
    /// the entire loop that used to live in `udp_rs_raw_bench.py`'s
    /// `RawUdpBench.send()`, moved into Rust. Real, measured reason this
    /// exists: that Python-level loop needed ~90 `send_batch`/`recv_batch`
    /// PyO3 round trips for a 16MB transfer (one per buffer-sized window
    /// round, itself bounded by the confirmed ~208KB kernel UDP buffer
    /// ceiling - see `project_os_udp_buffer_ceiling` in this project's own
    /// session memory) - each round paid real Python interpreter + FFI
    /// marshaling overhead this internal loop skips entirely by staying in
    /// Rust for the whole transfer. Same wire format/pacing algorithm as
    /// the Python version it replaces (1-byte type tag + 4-byte
    /// big-endian sequence number header, `window_chunks`-deep ACK
    /// window, unconditional small sleep between batches - see that
    /// module's docstring for why each of those exists, not repeated
    /// here). Blocks until every chunk is at least within the peer's most
    /// recently reported window - still no retransmission, a genuinely
    /// lost chunk is gone forever (see module docstring: this crate is a
    /// throughput-ceiling experiment, not a reliable transport).
    /// Retransmit-on-stall timeout: if the receiver's cumulative ack
    /// (see `recv_reliable`'s docstring for why it's cumulative, not
    /// "highest seen") hasn't advanced in this long while chunks remain
    /// outstanding, assume something in the current window was lost and
    /// resend the WHOLE outstanding range (Go-Back-N, not selective
    /// retransmit - see below for why that's safe here). 20ms is well
    /// above loopback/LAN RTT (sub-ms) but short enough not to waste much
    /// time on a real loss before recovering.
    const RETRANSMIT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(20);
    /// Give up and return an error after this many consecutive stalls
    /// with zero ack progress - matches the old Python `UDPTransport`'s
    /// `_MAX_SEND_ATTEMPTS` role (distinguish "peer is slow" from "peer
    /// is gone"), just expressed as a retransmit-round budget instead of
    /// a batch-attempt budget since this loop resends continuously
    /// rather than in discrete batches.
    const MAX_STALLED_ROUNDS: u32 = 500; // 500 * 20ms = ~10s of zero progress
    /// How often `recv_reliable`/`recv_message` re-send their current
    /// cumulative ack while otherwise idle, REGARDLESS of whether its
    /// value has changed since the last send. A real deadlock was found
    /// and fixed here, not just a missed optimization: the original
    /// version only re-sent when the ack VALUE changed, tracked via a
    /// local "last value sent" variable that gets updated the instant
    /// `send_batch` is CALLED - with no delivery confirmation for acks
    /// themselves (by design, to avoid an infinite ack-of-ack regress).
    /// If that one ack packet was itself lost in transit (a real,
    /// frequent event under real loss, not a hypothetical), the receiver
    /// believed it had already informed the sender of its latest state
    /// and would never spontaneously repeat it - while the sender's own
    /// periodic resends of already-fully-received data produce ZERO new
    /// state changes on the receiver (every chunk in them is now a
    /// duplicate), so nothing ever prompted a fresh ack attempt either.
    /// Both sides then wait on each other indefinitely, only broken by
    /// `MAX_STALLED_ROUNDS` giving up. Confirmed directly: induced 5-10%
    /// packet loss (via a lossy relay proxy - genuine loopback traffic
    /// essentially never drops on its own) reproduced this exact stall
    /// before the fix, resolved after switching to unconditional
    /// time-based ack resends. 2ms is well under `RETRANSMIT_TIMEOUT`
    /// (20ms) so a lost ack recovers well before the sender even
    /// considers the data itself lost.
    const ACK_RESEND_INTERVAL: std::time::Duration = std::time::Duration::from_millis(2);

    /// Sends `data`, chunked and ACK-windowed exactly like the unreliable
    /// version this replaces, but now genuinely reliable: a lost chunk
    /// gets resent, not silently dropped forever. Real fix, not just a
    /// pacing tweak - the old version advanced its send window using
    /// `highest_acked = highest sequence number the receiver has EVER
    /// SEEN`, which is wrong for detecting loss (a gap in the middle
    /// still lets a later chunk's ack "cover" it, so the sender never
    /// finds out and never resends it - confirmed as the actual
    /// mechanism behind this crate's long-standing "no retransmission,
    /// gone forever" limitation). `recv_reliable` now reports the
    /// cumulative highest CONTIGUOUS sequence received instead (like
    /// TCP's ack semantics), so a real gap keeps the ack pinned below it
    /// until the missing chunk actually arrives.
    ///
    /// Retransmission itself is Go-Back-N (resend the ENTIRE unacked
    /// range on a stall), not selective/SACK-based - deliberately simple:
    /// `recv_reliable` already de-duplicates by sequence number (a
    /// `received[seq]` bitmap - re-arriving chunks are simply ignored,
    /// not double-counted or double-written), so re-sending chunks the
    /// receiver already has is wasteful but never incorrect. Selective
    /// retransmission (resend only the actual missing seq) would use
    /// less bandwidth on a partial-window loss but needs the receiver to
    /// report which specific seqs are missing, not just a cumulative
    /// ack - left as a possible future improvement, not needed to reach
    /// "reliable" in the first place.
    pub fn send_reliable(
        &self,
        data: &[u8],
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> Result<()> {
        self.send_reliable_impl(data, chunk_payload, batch, window_chunks, timeout_ms)
    }

    /// Receives until `expected_bytes` worth of payload has arrived or
    /// `timeout_ms` elapses, sending periodic ACKs back so the peer's
    /// `send_reliable` can pace AND retransmit correctly - see that
    /// method's docstring and module docstring for the full rationale,
    /// not repeated here.
    ///
    /// **Ack semantics changed to make retransmission possible**: reports
    /// the cumulative highest CONTIGUOUS sequence received (like TCP),
    /// not the highest sequence ever seen. The old "highest seen" scheme
    /// let a later chunk's ack silently "cover" an earlier gap, so the
    /// sender never found out a chunk was lost - the actual mechanism
    /// behind this crate's old "no retransmission, gone forever"
    /// limitation. A `received: Vec<bool>` bitmap (one bit per expected
    /// chunk) tracks exactly which sequence numbers have actually
    /// arrived, both to compute the real contiguous frontier and to
    /// safely de-duplicate resent chunks (see `send_reliable`'s Go-Back-N
    /// docstring - a duplicate chunk arriving here is just a harmless
    /// no-op re-write, never double-counted in `received_bytes`).
    ///
    /// Reassembles chunks directly into their correct byte offset
    /// (derived from each chunk's sequence number and `chunk_payload`),
    /// not by concatenation in arrival order, so reordered chunks still
    /// land correctly. Returns `(bytes, chunks_received,
    /// max_seq_gap_observed)` - `max_seq_gap_observed` is still a raw
    /// "furthest a chunk arrived ahead of the contiguous frontier"
    /// diagnostic, not a correctness signal (correctness is now
    /// guaranteed by the retransmit loop, not just reported).
    pub fn recv_reliable(
        &self,
        expected_bytes: usize,
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> Result<(Vec<u8>, usize, i64)> {
        let ack_interval = (window_chunks / 2).max(1);
        let n_chunks = expected_bytes.div_ceil(chunk_payload.max(1));
        let mut out = vec![0u8; expected_bytes];
        let mut received: Vec<bool> = vec![false; n_chunks];
        let mut received_bytes = 0usize;
        let mut chunks_received = 0usize;
        let mut next_expected: usize = 0; // cumulative contiguous frontier
        let mut highest_seq: i64 = -1; // diagnostic only, see docstring
        let mut max_gap: i64 = 0;
        let mut chunks_since_ack = 0usize;
        let mut last_ack_resend_at = std::time::Instant::now();
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

        while received_bytes < expected_bytes && std::time::Instant::now() < deadline {
            let batch_pkts = self.poll_data(batch, chunk_payload + HEADER_BYTES)?;
            if batch_pkts.is_empty() {
                // Nothing arrived this poll - still worth re-sending the
                // current cumulative ack periodically (unconditionally,
                // NOT gated on "did the value change since last time" -
                // see `Self::ACK_RESEND_INTERVAL`'s docstring for why
                // that gating was a real deadlock bug, not just a missed
                // optimization) in case the sender's own copy of THIS
                // exact ack was itself lost in transit, not just data -
                // otherwise both sides can stall waiting on each other
                // with no further packets ever exchanged.
                if next_expected > 0 && last_ack_resend_at.elapsed() >= Self::ACK_RESEND_INTERVAL {
                    let mut ack = Vec::with_capacity(HEADER_BYTES);
                    ack.push(TYPE_ACK);
                    ack.extend_from_slice(&((next_expected - 1) as u32).to_be_bytes());
                    self.send_batch(std::slice::from_ref(&ack))?;
                    last_ack_resend_at = std::time::Instant::now();
                }
                self.wait_readable(std::time::Duration::from_micros(200));
                continue;
            }
            for pkt in batch_pkts {
                if pkt.len() < HEADER_BYTES || pkt[0] != TYPE_DATA {
                    continue;
                }
                let seq = u32::from_be_bytes(pkt[1..5].try_into().unwrap()) as i64;
                if highest_seq >= 0 && seq != highest_seq + 1 {
                    max_gap = max_gap.max((seq - highest_seq - 1).abs());
                }
                highest_seq = highest_seq.max(seq);

                let seq_idx = seq as usize;
                if seq_idx >= n_chunks || received[seq_idx] {
                    continue; // out of range or a harmless duplicate (see docstring)
                }
                received[seq_idx] = true;
                let payload = &pkt[HEADER_BYTES..];
                let offset = seq_idx * chunk_payload;
                let end = (offset + payload.len()).min(out.len());
                if offset < out.len() {
                    out[offset..end].copy_from_slice(&payload[..end - offset]);
                }
                received_bytes += end - offset;
                chunks_received += 1;
                while next_expected < n_chunks && received[next_expected] {
                    next_expected += 1;
                }
                chunks_since_ack += 1;
                if chunks_since_ack >= ack_interval && next_expected > 0 {
                    let mut ack = Vec::with_capacity(HEADER_BYTES);
                    ack.push(TYPE_ACK);
                    ack.extend_from_slice(&((next_expected - 1) as u32).to_be_bytes());
                    self.send_batch(std::slice::from_ref(&ack))?;
                    last_ack_resend_at = std::time::Instant::now();
                    chunks_since_ack = 0;
                }
            }
        }
        if next_expected > 0 {
            let mut ack = Vec::with_capacity(HEADER_BYTES);
            ack.push(TYPE_ACK);
            ack.extend_from_slice(&((next_expected - 1) as u32).to_be_bytes());
            self.send_batch(std::slice::from_ref(&ack))?;
        }
        out.truncate(received_bytes.min(expected_bytes));
        Ok((out, chunks_received, max_gap))
    }

    /// `send_reliable`'s counterpart for real `Transport` use, where the
    /// caller passing `data` obviously knows its own length - just calls
    /// straight through. Exists so callers have a `send_message`/
    /// `recv_message` pair with matching names; the interesting
    /// difference is entirely on the receive side (see `recv_message`'s
    /// docstring for why `expected_bytes` can't be a parameter there).
    /// Note the wire format differs from `send_reliable`'s (see
    /// `MSG_HEADER_BYTES`), so a `send_message` call must be paired with
    /// `recv_message` on the peer, never with `recv_reliable`.
    pub fn send_message(
        &self,
        data: &[u8],
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> Result<()> {
        let msg_id = self.next_msg_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if data.is_empty() {
            return Ok(()); // nothing to disambiguate - matches send_reliable_impl's own early-out
        }
        let n_chunks = data.len().div_ceil(chunk_payload);
        let total_len = data.len() as u32;
        let chunks: Vec<Vec<u8>> = (0..n_chunks)
            .map(|i| {
                let start = i * chunk_payload;
                let end = (start + chunk_payload).min(data.len());
                let mut chunk = Vec::with_capacity(MSG_DATA_HEADER_BYTES + (end - start));
                chunk.push(TYPE_DATA);
                chunk.extend_from_slice(&(i as u32).to_be_bytes());
                chunk.extend_from_slice(&total_len.to_be_bytes());
                chunk.extend_from_slice(&msg_id.to_be_bytes());
                chunk.extend_from_slice(&data[start..end]);
                chunk
            })
            .collect();
        self.send_chunks_reliable(
            chunks,
            chunk_payload + MSG_DATA_HEADER_BYTES,
            batch,
            window_chunks,
            timeout_ms,
            Some(msg_id),
        )
    }

    /// Receives one complete message without knowing its size in advance
    /// - unlike `recv_reliable`, which needs `expected_bytes` as a
    /// parameter (fine for a benchmark that already knows the payload
    /// size it asked to be sent, wrong for a real `Transport.recv()`
    /// caller that has no way to know how big the next message is before
    /// it arrives). Every chunk `send_message` sends carries `total_len`
    /// AND `msg_id` in its header (`MSG_DATA_HEADER_BYTES` = 13 -
    /// `[type:1][seq:4][total_len:4][msg_id:4]`), so `recv_message` can
    /// both learn the real size from WHICHEVER chunk happens to arrive
    /// first (regardless of reordering) and route each chunk to the
    /// right in-progress message (see `InboundMsgState`'s docstring for
    /// why that's needed even under this project's own
    /// send()-fully-returns-before-next-send() usage pattern). Otherwise
    /// identical Go-Back-N reliability to `recv_reliable`/`send_message`
    /// (cumulative contiguous ack per message, resend-on-stall).
    ///
    /// Safe to call repeatedly back-to-back with a peer doing the
    /// matching `send_message` sequence, including with NO pause between
    /// calls (streaming many small messages) - `msg_id` plus the
    /// persistent `inbound`/`completed_order` state on the engine is
    /// exactly what makes that safe now (see their docstrings).
    pub fn recv_message(
        &self,
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> Result<(Vec<u8>, usize, i64)> {
        if let Some(result) = self.take_completed_message()? {
            return Ok(result);
        }
        let ack_interval = (window_chunks / 2).max(1);
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

        loop {
            if std::time::Instant::now() >= deadline {
                return Err(error::io!("recv_message timed out after {timeout_ms}ms with no complete message"));
            }
            let batch_pkts = self.poll_data(batch, chunk_payload + MSG_DATA_HEADER_BYTES)?;
            if batch_pkts.is_empty() {
                // Re-send every currently-open message's cumulative ack
                // periodically, unconditionally - see `recv_reliable`'s
                // identical branch for why (a real deadlock bug fix, not
                // a micro-optimization), just applied per-message here.
                let mut inbound = self.inbound.lock().unwrap();
                for (&msg_id, state) in inbound.iter_mut() {
                    if state.next_expected > 0 && state.last_ack_resend_at.elapsed() >= Self::ACK_RESEND_INTERVAL {
                        self.send_msg_ack(msg_id, state.next_expected - 1)?;
                        state.last_ack_resend_at = std::time::Instant::now();
                    }
                }
                drop(inbound);
                self.wait_readable(std::time::Duration::from_micros(200));
                continue;
            }

            let mut inbound = self.inbound.lock().unwrap();
            let mut completed = self.completed_order.lock().unwrap();
            for pkt in batch_pkts {
                if pkt.len() < MSG_DATA_HEADER_BYTES || pkt[0] != TYPE_DATA {
                    continue;
                }
                let seq = u32::from_be_bytes(pkt[1..5].try_into().unwrap()) as i64;
                let total_len = u32::from_be_bytes(pkt[5..9].try_into().unwrap()) as usize;
                let msg_id = u32::from_be_bytes(pkt[9..13].try_into().unwrap());

                if completed.contains(&msg_id) {
                    continue; // already fully received, awaiting pickup - a harmless duplicate
                }
                let state = inbound.entry(msg_id).or_insert_with(|| {
                    let n_chunks = total_len.div_ceil(chunk_payload.max(1)).max(1);
                    InboundMsgState {
                        out: vec![0u8; total_len],
                        received: vec![false; n_chunks],
                        received_bytes: 0,
                        next_expected: 0,
                        highest_seq: -1,
                        max_gap: 0,
                        chunks_received: 0,
                        chunks_since_ack: 0,
                        last_ack_resend_at: std::time::Instant::now(),
                    }
                });
                let n_chunks = state.received.len();

                if state.highest_seq >= 0 && seq != state.highest_seq + 1 {
                    state.max_gap = state.max_gap.max((seq - state.highest_seq - 1).abs());
                }
                state.highest_seq = state.highest_seq.max(seq);

                let seq_idx = seq as usize;
                if seq_idx >= n_chunks || state.received[seq_idx] {
                    continue;
                }
                state.received[seq_idx] = true;
                let payload = &pkt[MSG_DATA_HEADER_BYTES..];
                let offset = seq_idx * chunk_payload;
                let end = (offset + payload.len()).min(state.out.len());
                if offset < state.out.len() {
                    state.out[offset..end].copy_from_slice(&payload[..end - offset]);
                }
                state.received_bytes += end - offset;
                state.chunks_received += 1;
                while state.next_expected < n_chunks && state.received[state.next_expected] {
                    state.next_expected += 1;
                }
                state.chunks_since_ack += 1;
                let should_ack = state.chunks_since_ack >= ack_interval && state.next_expected > 0;
                let ack_target = if should_ack { Some(state.next_expected - 1) } else { None };
                if should_ack {
                    state.chunks_since_ack = 0;
                    state.last_ack_resend_at = std::time::Instant::now();
                }

                if state.received_bytes >= total_len {
                    completed.push_back(msg_id);
                }
                if let Some(ack_seq) = ack_target {
                    self.send_msg_ack(msg_id, ack_seq)?;
                }
                if completed.back() == Some(&msg_id) {
                    // Just completed - send the final ack now (covers the
                    // last chunk even if it didn't happen to cross
                    // `ack_interval`), matching `recv_reliable`'s own
                    // unconditional final-ack send.
                    if let Some(state) = inbound.get(&msg_id) {
                        if state.next_expected > 0 {
                            self.send_msg_ack(msg_id, state.next_expected - 1)?;
                        }
                    }
                }
            }
            drop(completed);
            drop(inbound);
            if let Some(result) = self.take_completed_message()? {
                return Ok(result);
            }
        }
    }

    /// Pops the oldest fully-received-but-unclaimed message (see
    /// `completed_order`'s docstring), converting its `InboundMsgState`
    /// into `recv_message`'s public return shape and removing both
    /// entries. `Ok(None)` means nothing is ready yet - not an error.
    fn take_completed_message(&self) -> Result<Option<(Vec<u8>, usize, i64)>> {
        let msg_id = {
            let mut completed = self.completed_order.lock().unwrap();
            let Some(id) = completed.pop_front() else {
                return Ok(None);
            };
            id
        };
        let mut state = self
            .inbound
            .lock()
            .unwrap()
            .remove(&msg_id)
            .expect("completed_order and inbound must stay in sync - see recv_message");
        state.out.truncate(state.received_bytes.min(state.out.len()));
        Ok(Some((state.out, state.chunks_received, state.max_gap)))
    }

    /// Builds and sends one `MSG_ACK_HEADER_BYTES`-shaped ack packet for
    /// `msg_id` - shared by every ack-send site in `recv_message`.
    fn send_msg_ack(&self, msg_id: u32, cumulative_seq: usize) -> Result<()> {
        let mut ack = Vec::with_capacity(MSG_ACK_HEADER_BYTES);
        ack.push(TYPE_ACK);
        ack.extend_from_slice(&(cumulative_seq as u32).to_be_bytes());
        ack.extend_from_slice(&msg_id.to_be_bytes());
        self.send_batch(std::slice::from_ref(&ack))?;
        Ok(())
    }

    /// Builds `send_reliable`'s plain (no `total_len`/`msg_id`) chunks
    /// and hands them to `send_chunks_reliable` - see `send_reliable`'s
    /// docstring for the Go-Back-N retransmission scheme.
    fn send_reliable_impl(
        &self,
        data: &[u8],
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let n_chunks = data.len().div_ceil(chunk_payload);
        let chunks: Vec<Vec<u8>> = (0..n_chunks)
            .map(|i| {
                let start = i * chunk_payload;
                let end = (start + chunk_payload).min(data.len());
                let mut chunk = Vec::with_capacity(HEADER_BYTES + (end - start));
                chunk.push(TYPE_DATA);
                chunk.extend_from_slice(&(i as u32).to_be_bytes());
                chunk.extend_from_slice(&data[start..end]);
                chunk
            })
            .collect();
        self.send_chunks_reliable(chunks, chunk_payload + HEADER_BYTES, batch, window_chunks, timeout_ms, None)
    }

    /// The actual Go-Back-N ack-windowed send loop, shared by
    /// `send_reliable`/`send_message` (both just differ in how `chunks`
    /// got built, and whether `msg_id` disambiguation is needed) - see
    /// `send_reliable`'s docstring for the full rationale, kept there
    /// since that's the method most callers/docs reference by name.
    ///
    /// `msg_id`: `None` for `send_reliable`'s plain wire format (5-byte
    /// ack: `[type][seq]`, matches ANY ack it sees, since a single-shot
    /// benchmark call never has another message's ack to confuse it
    /// with). `Some(id)` for `send_message`'s format (9-byte ack:
    /// `[type][seq][msg_id]`) - acks for a DIFFERENT `msg_id` are
    /// ignored here (a stray/delayed ack for an earlier message this
    /// engine sent), not treated as progress on the current one.
    fn send_chunks_reliable(
        &self,
        chunks: Vec<Vec<u8>>,
        recv_probe_len: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
        msg_id: Option<u32>,
    ) -> Result<()> {
        let matches_ack = |pkt: &[u8]| -> Option<i64> {
            match msg_id {
                None => {
                    if pkt.first() == Some(&TYPE_ACK) && pkt.len() >= HEADER_BYTES {
                        Some(u32::from_be_bytes(pkt[1..5].try_into().unwrap()) as i64)
                    } else {
                        None
                    }
                }
                Some(want_id) => {
                    if pkt.first() == Some(&TYPE_ACK) && pkt.len() >= MSG_ACK_HEADER_BYTES {
                        let ack_msg_id = u32::from_be_bytes(pkt[5..9].try_into().unwrap());
                        (ack_msg_id == want_id)
                            .then(|| u32::from_be_bytes(pkt[1..5].try_into().unwrap()) as i64)
                    } else {
                        None
                    }
                }
            }
        };

        let overall_deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let mut highest_acked: i64 = -1;
        let mut idx = 0usize;
        let mut last_progress = std::time::Instant::now();
        let mut stalled_rounds: u32 = 0;
        while idx < chunks.len() {
            if std::time::Instant::now() > overall_deadline {
                return Err(error::io!(
                    "send timed out after {timeout_ms}ms with {}/{} chunks unacked",
                    chunks.len() - (highest_acked + 1) as usize,
                    chunks.len()
                ));
            }
            let mut acked_this_round = false;
            for pkt in self.poll_acks(batch, recv_probe_len)? {
                if let Some(seq) = matches_ack(&pkt) {
                    if seq > highest_acked {
                        highest_acked = seq;
                        acked_this_round = true;
                    }
                }
            }
            if acked_this_round {
                last_progress = std::time::Instant::now();
                stalled_rounds = 0;
            }

            let outstanding = idx as i64 - (highest_acked + 1);
            if outstanding > 0 && last_progress.elapsed() >= Self::RETRANSMIT_TIMEOUT {
                let resend_start = (highest_acked + 1) as usize;
                self.send_batch(&chunks[resend_start..idx])?;
                last_progress = std::time::Instant::now();
                stalled_rounds += 1;
                if stalled_rounds >= Self::MAX_STALLED_ROUNDS {
                    return Err(error::io!(
                        "send: peer stopped acking - {} consecutive stalls with chunk {} \
                         still unacked (peer likely gone)",
                        stalled_rounds,
                        resend_start
                    ));
                }
                continue;
            }

            let window_end = (highest_acked + 1) as usize + window_chunks;
            let allowed = window_end.saturating_sub(idx);
            if allowed == 0 {
                // Window genuinely full - nothing to do but wait for an
                // ack to open it up. `wait_readable` (real blocking
                // `poll(2)`), not a fixed sleep - see its own docstring
                // for the real latency bug this fixes.
                self.wait_readable(std::time::Duration::from_micros(500));
                continue;
            }

            let this_batch_len = allowed.min(batch).min(chunks.len() - idx);
            let sent = self.send_batch(&chunks[idx..idx + this_batch_len])?;
            idx += sent;
            if sent == 0 {
                // Kernel send buffer momentarily full (not a "waiting for
                // incoming data" case - `wait_readable` doesn't apply
                // here) - just yield.
                std::thread::sleep(std::time::Duration::from_micros(0));
            } else if idx < chunks.len() {
                // Deliberate send-side pacing, not idle-waiting - see
                // this method's docstring for why an unconditional small
                // gap after every batch is needed regardless of window
                // state. Intentionally still a fixed sleep, not
                // `wait_readable` - throttling our OWN send rate, not
                // waiting for the peer.
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
        let final_deadline = std::time::Instant::now() + Self::RETRANSMIT_TIMEOUT * 4;
        while highest_acked + 1 < chunks.len() as i64 {
            if std::time::Instant::now() > overall_deadline {
                return Err(error::io!(
                    "send timed out after {timeout_ms}ms waiting for final ack ({}/{} chunks acked)",
                    highest_acked + 1,
                    chunks.len()
                ));
            }
            for pkt in self.poll_acks(batch, recv_probe_len)? {
                if let Some(seq) = matches_ack(&pkt) {
                    if seq > highest_acked {
                        highest_acked = seq;
                        last_progress = std::time::Instant::now();
                    }
                }
            }
            if highest_acked + 1 >= chunks.len() as i64 {
                break;
            }
            if last_progress.elapsed() >= Self::RETRANSMIT_TIMEOUT || std::time::Instant::now() > final_deadline {
                let resend_start = (highest_acked + 1) as usize;
                self.send_batch(&chunks[resend_start..])?;
                last_progress = std::time::Instant::now();
            } else {
                // Genuinely waiting for the peer's final ack - see
                // `wait_readable`'s docstring for the real ping-pong
                // latency bug this fixes (this exact branch was a
                // dominant contributor: EVERY `send_reliable`/
                // `send_message` call passes through here at least
                // once, even for a single-chunk message).
                self.wait_readable(std::time::Duration::from_micros(200));
            }
        }
        Ok(())
    }

    /// GSO-based counterpart to `send_reliable` - same wire format and
    /// ACK-windowed pacing algorithm, but each round's worth of chunks is
    /// sent as ONE `send_gso` call (kernel-side segmentation) instead of
    /// several `sendmmsg`-batched discrete messages. `round_bytes` bounds
    /// how many chunks go into one GSO buffer (still needs to respect the
    /// same kernel receive-buffer ceiling `send_reliable`'s `batch` does -
    /// GSO changes how cheaply the SEND side can emit a burst, not how
    /// much the RECEIVE side can buffer, so the same buffer-derived
    /// sizing still applies here).
    pub fn send_reliable_gso(
        &self,
        data: &[u8],
        chunk_payload: usize,
        round_bytes: usize,
        window_chunks: usize,
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let segment_size = chunk_payload + HEADER_BYTES;
        let n_chunks = data.len().div_ceil(chunk_payload);
        // One `sendmsg(GSO)` call's aggregate buffer is bounded by the
        // classic UDP max-payload size (~65507 bytes - the 16-bit length
        // field a UDP datagram itself is limited to, which the whole GSO
        // "super-buffer" still has to fit inside before the kernel
        // internally splits it into real wire segments). Confirmed
        // directly: `send_gso` with a 64400-byte buffer succeeds, a
        // 65800-byte one fails with `EMSGSIZE` (os error 90) - this is
        // NOT the same as `round_bytes`/the kernel socket buffer ceiling
        // (`project_os_udp_buffer_ceiling`), it is a hard, separate
        // per-call cap regardless of buffer size. Real bug found running
        // this with `chunk_payload` sized for loopback's large MTU
        // (~61KB, matching the sendmmsg-based methods) - a "round" of
        // even 2 such chunks already exceeds 65535, so `send_gso` failed
        // outright. GSO's actual sweet spot (confirmed against quinn-udp/
        // Firefox's own production choice - see `udp_rs_raw_bench.py`'s
        // module docstring) is many SMALL, real-network-MTU-sized
        // segments packed into that one ~65KB budget, not a few large
        // loopback-sized ones - capped here defensively regardless of
        // what `chunk_payload` the caller passes.
        const GSO_MAX_AGGREGATE_BYTES: usize = 65500;
        let chunks_per_round = (round_bytes / segment_size)
            .max(1)
            .min(GSO_MAX_AGGREGATE_BYTES / segment_size.max(1))
            .max(1);

        let mut highest_acked: i64 = -1;
        let mut idx = 0usize;
        while idx < n_chunks {
            loop {
                let (buf, _seg) = self.recv_gro(segment_size)?;
                if buf.is_empty() {
                    break;
                }
                if buf[0] == TYPE_ACK && buf.len() >= HEADER_BYTES {
                    let seq = u32::from_be_bytes(buf[1..5].try_into().unwrap()) as i64;
                    highest_acked = highest_acked.max(seq);
                }
            }

            let window_end = (highest_acked + 1) as usize + window_chunks;
            let allowed = window_end.saturating_sub(idx);
            if allowed == 0 {
                std::thread::sleep(std::time::Duration::from_micros(500));
                continue;
            }

            let this_round = allowed.min(chunks_per_round).min(n_chunks - idx);
            let mut round_buf = Vec::with_capacity(this_round * segment_size);
            for j in 0..this_round {
                let seq = idx + j;
                let start = seq * chunk_payload;
                let end = (start + chunk_payload).min(data.len());
                round_buf.push(TYPE_DATA);
                round_buf.extend_from_slice(&(seq as u32).to_be_bytes());
                round_buf.extend_from_slice(&data[start..end]);
            }
            let sent_bytes = self.send_gso(&round_buf, segment_size as u16)?;
            if sent_bytes >= round_buf.len() {
                idx += this_round;
                if idx < n_chunks {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            } else {
                // Partial/failed GSO send (rare - the kernel send path was
                // momentarily unable to accept the whole batch) - retry
                // the SAME round next iteration rather than guessing which
                // prefix of segments actually made it out; safe because
                // this engine has no reliability guarantee to violate
                // either way (see module docstring).
                std::thread::sleep(std::time::Duration::from_micros(0));
            }
        }
        Ok(())
    }

    /// GRO-based counterpart to `recv_reliable` - same reassembly/ack
    /// algorithm, but each `recvmsg` may return several coalesced
    /// segments at once (see `recv_gro`) instead of one datagram at a
    /// time, and there is no artificial `max_batch` - the kernel decides
    /// how much it was able to coalesce.
    pub fn recv_reliable_gro(
        &self,
        expected_bytes: usize,
        chunk_payload: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> Result<(Vec<u8>, usize, i64)> {
        let ack_interval = (window_chunks / 2).max(1);
        let segment_size = chunk_payload + HEADER_BYTES;
        // Big enough for a fully GRO-coalesced burst up to one window's
        // worth - the kernel truncates to whatever it actually coalesced,
        // never writes past this.
        let recv_buf_len = segment_size * window_chunks.max(1);
        let mut out = vec![0u8; expected_bytes];
        let mut received_bytes = 0usize;
        let mut chunks_received = 0usize;
        let mut highest_seq: i64 = -1;
        let mut max_gap: i64 = 0;
        let mut chunks_since_ack = 0usize;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

        while received_bytes < expected_bytes && std::time::Instant::now() < deadline {
            let (buf, seg) = self.recv_gro(recv_buf_len)?;
            if buf.is_empty() {
                std::thread::sleep(std::time::Duration::from_micros(0));
                continue;
            }
            let seg = seg.max(1);
            for piece in buf.chunks(seg) {
                if piece.len() < HEADER_BYTES || piece[0] != TYPE_DATA {
                    continue;
                }
                let seqno = u32::from_be_bytes(piece[1..5].try_into().unwrap()) as i64;
                if highest_seq >= 0 && seqno != highest_seq + 1 {
                    max_gap = max_gap.max((seqno - highest_seq - 1).abs());
                }
                highest_seq = highest_seq.max(seqno);
                let payload = &piece[HEADER_BYTES..];
                let offset = seqno as usize * chunk_payload;
                if offset < out.len() {
                    let end = (offset + payload.len()).min(out.len());
                    out[offset..end].copy_from_slice(&payload[..end - offset]);
                }
                received_bytes += payload.len();
                chunks_received += 1;
                chunks_since_ack += 1;
                if chunks_since_ack >= ack_interval {
                    let mut ack = Vec::with_capacity(HEADER_BYTES);
                    ack.push(TYPE_ACK);
                    ack.extend_from_slice(&(highest_seq as u32).to_be_bytes());
                    self.send_batch(std::slice::from_ref(&ack))?;
                    chunks_since_ack = 0;
                }
            }
        }
        if highest_seq >= 0 {
            let mut ack = Vec::with_capacity(HEADER_BYTES);
            ack.push(TYPE_ACK);
            ack.extend_from_slice(&(highest_seq as u32).to_be_bytes());
            self.send_batch(std::slice::from_ref(&ack))?;
        }
        out.truncate(received_bytes.min(expected_bytes));
        Ok((out, chunks_received, max_gap))
    }
}

const HEADER_BYTES: usize = 5; // 1 byte type tag + 4 byte seq/ack number
/// `send_message`/`recv_message`'s data header: `HEADER_BYTES` plus a
/// 4-byte `total_len` field (so the receiver can discover message size
/// dynamically - see `recv_message`'s docstring) plus a 4-byte `msg_id`
/// (so chunks from consecutive messages on the same engine can never be
/// confused with each other - see `InboundMsgState`'s docstring for the
/// real cross-message corruption bug this fixes).
const MSG_DATA_HEADER_BYTES: usize = HEADER_BYTES + 4 + 4;
/// `send_message`/`recv_message`'s ack header: `HEADER_BYTES` plus the
/// same 4-byte `msg_id` (so a sender can tell an ack for ITS current
/// message apart from a stray/delayed ack belonging to an earlier one).
const MSG_ACK_HEADER_BYTES: usize = HEADER_BYTES + 4;
const TYPE_DATA: u8 = 0;
const TYPE_ACK: u8 = 1;

/// `linux/udp.h`: `#define UDP_SEGMENT 103` / `#define UDP_GRO 104` - not
/// exposed as named constants by the `libc` crate (same situation as
/// `quic_rs_transport.py`'s own `_UDP_SEGMENT`, see that module's
/// docstring), so the raw kernel values are used directly here too.
const UDP_SEGMENT: libc::c_int = 103;
const UDP_GRO: libc::c_int = 104;

/// `CMSG_SPACE(len)`/`CMSG_LEN(len)` - the `libc` crate exposes
/// `CMSG_FIRSTHDR`/`CMSG_DATA` as real functions but not these two (they're
/// header macros in C, awkward to bind directly), so they're reimplemented
/// here matching glibc's own definition exactly: align `sizeof(cmsghdr)`
/// and `len` each to `size_t`, then sum.
fn cmsg_align(len: usize) -> usize {
    let align = std::mem::size_of::<usize>();
    (len + align - 1) & !(align - 1)
}

fn cmsg_space(len: usize) -> usize {
    cmsg_align(std::mem::size_of::<libc::cmsghdr>()) + cmsg_align(len)
}

fn cmsg_len(len: usize) -> usize {
    cmsg_align(std::mem::size_of::<libc::cmsghdr>()) + len
}

impl RawUdpEngine {
    /// Opts this socket in to Linux UDP GRO (Generic Receive Offload) -
    /// the kernel coalesces several incoming same-flow datagrams into one
    /// larger buffer BEFORE this process ever wakes up for them, delivered
    /// via a single `recvmsg()` (see `recv_gro`) instead of one syscall
    /// (or even one batched `recvmmsg`) per original datagram. This is the
    /// same GSO/GRO pairing quinn-udp (Firefox's own QUIC UDP layer) uses
    /// in production, deliberately preferred over `sendmmsg`/`recvmmsg` -
    /// see `udp_rs_raw_bench.py`'s module docstring for the citation.
    /// Must be called once per socket before `recv_gro` is useful (a
    /// socket without this set just gets ordinary non-coalesced datagrams
    /// back, one per `recv_gro` call, correctly but without the batching
    /// benefit).
    pub fn enable_gro(&self) -> Result<()> {
        let value: libc::c_int = 1;
        // SAFETY: `value` is a valid, live `c_int` for the duration of
        // this call; `setsockopt` reads exactly `size_of::<c_int>()`
        // bytes from the pointer given a matching `optlen`, which is what
        // is passed.
        let ret = unsafe {
            libc::setsockopt(
                self.socket.as_raw_fd(),
                libc::SOL_UDP,
                UDP_GRO,
                &value as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        if ret < 0 {
            return Err(error::setup!(
                "setsockopt(UDP_GRO) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }

    /// Sends the whole of `data` as ONE real Linux GSO batch (`sendmsg()`
    /// with a `UDP_SEGMENT` control message) instead of several
    /// `sendmmsg()`-batched discrete messages - the kernel itself splits
    /// `data` into `segment_size`-byte datagrams (the last one may be
    /// shorter) and transmits all of them from this ONE syscall. Returns
    /// how many bytes the kernel accepted (a genuine partial send is
    /// possible, same contract as `send_batch` - the caller retries the
    /// remainder). `data.len()` should be a multiple of `segment_size`
    /// except possibly for the final short segment, matching GSO's own
    /// requirement (checked directly against `quic_rs_transport.py`'s
    /// existing GSO send path, which this mirrors).
    pub fn send_gso(&self, data: &[u8], segment_size: u16) -> Result<usize> {
        if data.is_empty() {
            return Ok(0);
        }
        let mut iov = libc::iovec {
            iov_base: data.as_ptr() as *mut libc::c_void,
            iov_len: data.len(),
        };
        let mut cmsg_buf = vec![0u8; cmsg_space(std::mem::size_of::<u16>())];

        // SAFETY: `msg` is fully zeroed first (every field explicitly
        // valid, no uninitialized padding read), then only the fields
        // this call actually needs are set; `msg_iov`/`msg_control` point
        // at `iov`/`cmsg_buf`, both of which outlive the `sendmsg` call
        // below (declared in this same stack frame, not moved).
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_buf.len();

        // SAFETY: `CMSG_FIRSTHDR` returns a pointer into `cmsg_buf`
        // (valid, non-null, since `msg_controllen` is exactly
        // `cmsg_space(size_of::<u16>())` - large enough for one cmsg
        // header + a `u16` payload). Writing the header fields and then
        // the `u16` segment size via `CMSG_DATA`'s returned pointer stays
        // within that allocation.
        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_UDP;
            (*cmsg).cmsg_type = UDP_SEGMENT;
            (*cmsg).cmsg_len = cmsg_len(std::mem::size_of::<u16>()) as _;
            (libc::CMSG_DATA(cmsg) as *mut u16).write_unaligned(segment_size);
        }

        // SAFETY: `msg` and everything it points to (`iov`, `cmsg_buf`)
        // are valid and alive for the duration of this call.
        let sent = unsafe { libc::sendmsg(self.socket.as_raw_fd(), &msg, 0) };
        if sent < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Ok(0);
            }
            return Err(error::io!("sendmsg (GSO) failed: {err}"));
        }
        Ok(sent as usize)
    }

    /// Receives one (possibly GRO-coalesced) message via `recvmsg()`.
    /// Returns `(bytes, segment_size)` - if the kernel actually coalesced
    /// several incoming datagrams (requires `enable_gro` to have been
    /// called first, AND the peer's datagrams to have arrived close
    /// enough together for the kernel to merge them - not guaranteed
    /// every time), `segment_size < bytes.len()` and the caller should
    /// split `bytes` into `segment_size`-byte pieces (the last one may be
    /// shorter). If nothing was coalesced (the common case for a single
    /// isolated datagram, or GRO not enabled), `segment_size ==
    /// bytes.len()` - safe to treat as one plain datagram either way by
    /// always doing the split, since a 1-segment "split" is a no-op.
    pub fn recv_gro(&self, max_len: usize) -> Result<(Vec<u8>, usize)> {
        // Reused, not freshly allocated - see `scratch_gro`'s field
        // docstring.
        let mut buf_guard = self.scratch_gro.lock().unwrap();
        buf_guard.clear();
        buf_guard.resize(max_len, 0);
        let buf = &mut *buf_guard;
        let mut iov = libc::iovec {
            iov_base: buf.as_mut_ptr() as *mut libc::c_void,
            iov_len: buf.len(),
        };
        let mut cmsg_buf = vec![0u8; cmsg_space(std::mem::size_of::<u16>())];

        // SAFETY: same reasoning as `send_gso` - `msg` fully zeroed then
        // only valid fields set, `msg_iov`/`msg_control` point at `iov`/
        // `cmsg_buf`, both alive for the duration of this call.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_buf.len();

        // SAFETY: `msg` and everything it points to are valid and alive
        // for the duration of this call; the kernel never writes past
        // `iov.iov_len`/`msg_controllen`, which match the real allocated
        // sizes of `buf`/`cmsg_buf`.
        let received = unsafe { libc::recvmsg(self.socket.as_raw_fd(), &mut msg, 0) };
        if received < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Ok((Vec::new(), 0));
            }
            return Err(error::io!("recvmsg (GRO) failed: {err}"));
        }
        let received = received as usize;

        let mut segment_size = received;
        // SAFETY: `CMSG_FIRSTHDR` either returns null (no cmsg present -
        // handled below) or a valid pointer into `cmsg_buf`, which is
        // still alive here. Only read if non-null AND it's actually the
        // `UDP_GRO` cmsg this code is looking for.
        unsafe {
            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            if !cmsg.is_null() && (*cmsg).cmsg_level == libc::SOL_UDP && (*cmsg).cmsg_type == UDP_GRO {
                segment_size = (libc::CMSG_DATA(cmsg) as *const u16).read_unaligned() as usize;
            }
        }
        // Copies out only the bytes actually received into a fresh owned
        // `Vec<u8>` for the caller - `scratch_gro` itself stays borrowed
        // from `self` and gets reused (already the right capacity) on the
        // next call, same reasoning as `recv_batch`.
        Ok((buf[..received].to_vec(), segment_size))
    }
}
