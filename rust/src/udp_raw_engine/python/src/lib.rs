// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Thin PyO3 bindings for `vllm_udp_raw_engine::RawUdpEngine`.
//!
//! Takes ownership of a DUPLICATED copy of the Python-side socket's file
//! descriptor (`libc::dup`), never the original - the Python `socket.
//! socket` object keeps managing/closing its own fd exactly as it always
//! has (e.g. via `sock.close()` in `udp_rs_transport.py`'s `close()`), and
//! this engine's independent copy is closed separately when `PyRawUdpEngine`
//! is dropped. Constructing a `std::net::UdpSocket` directly from the
//! ORIGINAL fd (`from_raw_fd` without `dup` first) would make Rust believe
//! it uniquely owns that fd and close it on drop - a real double-close/
//! use-after-close hazard against the Python socket object still holding
//! the same fd number.

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use std::net::UdpSocket;
use std::os::unix::io::FromRawFd;

use vllm_udp_raw_engine::RawUdpEngine;

fn to_py_err<E: std::fmt::Display>(error: E) -> PyErr {
    PyOSError::new_err(error.to_string())
}

#[pyclass]
struct PyRawUdpEngine {
    inner: RawUdpEngine,
}

#[pymethods]
impl PyRawUdpEngine {
    /// `fd` is the Python socket's `sock.fileno()` - see module docstring
    /// for why this duplicates it rather than taking the original.
    #[staticmethod]
    fn from_fd(fd: i32) -> PyResult<Self> {
        // SAFETY: `libc::dup` on a valid, open fd (guaranteed by the
        // Python caller passing a real `socket.socket().fileno()`) either
        // returns a new valid fd referring to the same underlying open
        // file description, or -1 on error (checked immediately below,
        // before the result is trusted).
        let dup_fd = unsafe { libc::dup(fd) };
        if dup_fd < 0 {
            return Err(to_py_err(std::io::Error::last_os_error()));
        }
        // SAFETY: `dup_fd` was just returned by a successful `dup()`
        // above - a fresh, valid, open, non-negative fd that nothing else
        // holds a `UdpSocket`/`OwnedFd` wrapper around yet, so this is the
        // unique Rust-side owner of exactly this fd number from here on.
        let socket = unsafe { UdpSocket::from_raw_fd(dup_fd) };
        let inner = RawUdpEngine::new(socket).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Sends `chunks` (a list of `bytes`) as one `sendmmsg()` batch.
    /// Returns how many were actually accepted - see `RawUdpEngine::
    /// send_batch`'s docstring for why this can be less than
    /// `len(chunks)` and what the caller must do about it.
    fn send_batch(&self, chunks: Vec<Vec<u8>>) -> PyResult<usize> {
        self.inner.send_batch(&chunks).map_err(to_py_err)
    }

    /// Receives up to `max_batch` datagrams (each truncated to
    /// `max_msg_len` bytes) in one `recvmmsg()` call. Returns an empty
    /// list if nothing is available right now (non-blocking).
    fn recv_batch<'py>(
        &self,
        py: Python<'py>,
        max_batch: usize,
        max_msg_len: usize,
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyBytes>>> {
        let batches = self.inner.recv_batch(max_batch, max_msg_len).map_err(to_py_err)?;
        Ok(batches
            .into_iter()
            .map(|buf| pyo3::types::PyBytes::new(py, &buf))
            .collect())
    }

    /// Sends the whole of `data` in one call - see `RawUdpEngine::
    /// send_reliable`'s docstring for the full rationale (this exists to
    /// move the entire chunk/batch/ack-window loop out of Python, which
    /// needed ~90 separate `send_batch`/`recv_batch` PyO3 calls for a
    /// 16MB transfer otherwise). Releases the GIL for the duration (`py.
    /// detach`) since this can legitimately run for a while and
    /// touches no Python state internally - matches this workspace's
    /// convention of only holding the GIL for as long as Python objects
    /// are actually being read/constructed.
    #[pyo3(signature = (data, chunk_payload, batch, window_chunks, timeout_ms))]
    fn send_reliable(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .send_reliable(&data, chunk_payload, batch, window_chunks, timeout_ms)
        })
        .map_err(to_py_err)
    }

    /// Receives `expected_bytes` worth of payload (or until `timeout_ms`
    /// elapses) - see `RawUdpEngine::recv_reliable`'s docstring. Returns
    /// `(bytes, chunks_received, max_seq_gap)`. Also releases the GIL for
    /// the duration.
    #[pyo3(signature = (expected_bytes, chunk_payload, batch, window_chunks, timeout_ms))]
    fn recv_reliable<'py>(
        &self,
        py: Python<'py>,
        expected_bytes: usize,
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, usize, i64)> {
        let (data, chunks, gap) = py
            .detach(|| {
                self.inner
                    .recv_reliable(expected_bytes, chunk_payload, batch, window_chunks, timeout_ms)
            })
            .map_err(to_py_err)?;
        Ok((pyo3::types::PyBytes::new(py, &data), chunks, gap))
    }

    /// `send_reliable`'s counterpart for real `Transport.send()` use -
    /// see `RawUdpEngine::send_message`'s docstring. Must be paired with
    /// `recv_message` on the peer, never with `recv_reliable`.
    #[pyo3(signature = (data, chunk_payload, batch, window_chunks, timeout_ms))]
    fn send_message(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .send_message(&data, chunk_payload, batch, window_chunks, timeout_ms)
        })
        .map_err(to_py_err)
    }

    /// Receives one complete message without knowing its size in advance
    /// - see `RawUdpEngine::recv_message`'s docstring for the discovery
    /// mechanism and the single-message-in-flight caveat. Returns
    /// `(bytes, chunks_received, max_seq_gap)`.
    #[pyo3(signature = (chunk_payload, batch, window_chunks, timeout_ms))]
    fn recv_message<'py>(
        &self,
        py: Python<'py>,
        chunk_payload: usize,
        batch: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, usize, i64)> {
        let (data, chunks, gap) = py
            .detach(|| self.inner.recv_message(chunk_payload, batch, window_chunks, timeout_ms))
            .map_err(to_py_err)?;
        Ok((pyo3::types::PyBytes::new(py, &data), chunks, gap))
    }

    /// Opts this socket in to Linux UDP GRO - see `RawUdpEngine::
    /// enable_gro`'s docstring. Call once before `send_reliable_gso`/
    /// `recv_reliable_gro`.
    fn enable_gro(&self) -> PyResult<()> {
        self.inner.enable_gro().map_err(to_py_err)
    }

    /// Low-level single GSO send - see `RawUdpEngine::send_gso`'s
    /// docstring. Exposed mainly for direct testing of the GSO/GRO
    /// primitives independent of the higher-level windowed methods.
    fn send_gso(&self, data: Vec<u8>, segment_size: u16) -> PyResult<usize> {
        self.inner.send_gso(&data, segment_size).map_err(to_py_err)
    }

    /// Low-level single GRO receive - see `RawUdpEngine::recv_gro`'s
    /// docstring. Returns `(bytes, segment_size)`.
    fn recv_gro<'py>(
        &self,
        py: Python<'py>,
        max_len: usize,
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, usize)> {
        let (buf, seg) = self.inner.recv_gro(max_len).map_err(to_py_err)?;
        Ok((pyo3::types::PyBytes::new(py, &buf), seg))
    }

    /// GSO-based counterpart to `send_reliable` - see `RawUdpEngine::
    /// send_reliable_gso`'s docstring.
    #[pyo3(signature = (data, chunk_payload, round_bytes, window_chunks))]
    fn send_reliable_gso(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        chunk_payload: usize,
        round_bytes: usize,
        window_chunks: usize,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .send_reliable_gso(&data, chunk_payload, round_bytes, window_chunks)
        })
        .map_err(to_py_err)
    }

    /// GRO-based counterpart to `recv_reliable` - see `RawUdpEngine::
    /// recv_reliable_gro`'s docstring. Returns `(bytes, chunks_received,
    /// max_seq_gap)`.
    #[pyo3(signature = (expected_bytes, chunk_payload, window_chunks, timeout_ms))]
    fn recv_reliable_gro<'py>(
        &self,
        py: Python<'py>,
        expected_bytes: usize,
        chunk_payload: usize,
        window_chunks: usize,
        timeout_ms: u64,
    ) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, usize, i64)> {
        let (data, chunks, gap) = py
            .detach(|| {
                self.inner
                    .recv_reliable_gro(expected_bytes, chunk_payload, window_chunks, timeout_ms)
            })
            .map_err(to_py_err)?;
        Ok((pyo3::types::PyBytes::new(py, &data), chunks, gap))
    }
}

#[pymodule]
fn _rust_udp_raw_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRawUdpEngine>()?;
    Ok(())
}
