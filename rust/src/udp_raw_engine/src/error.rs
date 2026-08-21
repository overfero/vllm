// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Error type for `vllm_udp_raw_engine` - same `panic = "abort"` rationale
//! as `vllm_quic_engine::error` (see that module's docstring), applies
//! identically here.

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, EngineError>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
pub enum EngineError {
    #[error("UDP socket setup failed: {message}")]
    Setup { message: String },
    #[error("UDP I/O error: {message}")]
    Io { message: String },
}
