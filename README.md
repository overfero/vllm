# vLLM over UDP hole-punching: cross-NAT pipeline-parallel serving

Serve a large model **pipeline-parallel across machines that don't have
public IPs or a shared VPC** - no Ray, no gRPC, no port-forwarding, no VPN.
Each machine punches a direct UDP path to its neighbors through NAT using
only a small public signaling server for rendezvous, and vLLM's own
pipeline-parallel scheduling runs on top of that transport instead of NCCL
send/recv.

Currently running Qwen3.5-122B-A10B-GPTQ-Int4 across 4 machines (2x Tesla
T4 each, TP=2/PP=4, 12 layers/stage) with MTP speculative decoding enabled
to cut down the number of network round-trips per generated token. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the current status, measured
numbers, and the real bugs that had to be fixed to get MTP working across a
synthetic (non-native) pipeline-parallel group.

This is a fork of [vLLM](https://github.com/vllm-project/vllm) - upstream's
own README is at [`docs/UPSTREAM_VLLM_README.md`](docs/UPSTREAM_VLLM_README.md).

## Why

vLLM's real pipeline-parallel support assumes every rank can reach every
other rank directly (NCCL over TCP/IB, typically same datacenter or a VPC).
That's unavailable when the machines are, say, free-tier GPU notebook
instances on different cloud accounts with no public IP and no shared
network - exactly the case this project targets. The fix is a drop-in
transport swap: vLLM's PP send/recv path is redirected to a custom
transport, and one of the transport backends is a UDP hole-punch
implementation that only needs outbound internet + a small stateless
signaling server, not any inbound port or VPN.

## Architecture

```
udp_holepunch/          standalone UDP hole-punch library (signaling
                         client/server, NAT traversal, reliable delivery
                         on top of UDP) - no vLLM or torch dependency

vllm/transport/          the vLLM-specific adapter: imports udp_holepunch/
                         (and implements a plain TCP backend) behind a
                         small Transport interface, then replaces the
                         pipeline-parallel dimension of vLLM's real
                         GroupCoordinator with a transport-backed one
                         right after local TP bootstrap (pp_worker.py,
                         pipeline_bootstrap.py), plus a TransportExecutor
                         for the driver rank (rpc_executor.py) - see
                         vllm/transport/README.md

humming_fix/             SM75 (Tesla T4) runtime bugfixes for the
                         humming-kernels MoE package this model needs,
                         plus the checkpoint download/extraction tooling
                         (per-stage selective shard download, memory-safe
                         batched extraction)

scripts/                 stage_server.py (non-driver stage launcher) and
                         launch_pp_stage.py (driver/vllm-serve launcher)

ops/                     orchestrator-side remote machine bring-up
                         (torch/vllm/humming-kernels install, checkpoint
                         extraction) over SSH

pp_tests/                launch scripts for the current cluster
                         (pp_tests/launch/), diagnostics, and validation
                         scripts written against real hardware
```

Two-layer split (`udp_holepunch` -> `vllm/transport`), not three: an
earlier design (`docs/ARCHITECTURE_DECISION.md`) planned a
framework-agnostic `transport_runtime` package in between, generic enough
to sit under other inference frameworks too - but that migration was
never actually wired into the live code path (`vllm/transport/*.py`
always imported `udp_holepunch/peer.py` directly and still does), so it's
been removed from this repo rather than kept as misleading dead code. See
`docs/ARCHITECTURE_DECISION.md`'s path note for the full history if
reviving that extraction is ever worth it.

## Quick start

Only 2 remote machines available (no MTP)? `cluster/qwen35_122ba10b_3machine.sh`
is a one-shot script for exactly that: 3 machines total (this sandbox +
2 remote), 48 layers split 16/16/16, no MTP.

```bash
git clone <this repo> && cd vllm
cp .env.example .env   # fill in current session's MACHINE_B/C SSH port+password
./cluster/qwen35_122ba10b_3machine.sh
```

For the full 4-machine/MTP cluster this repo is currently validated
against, there's no single wrapper script yet - see
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the manual per-machine
bring-up (`ops/setup_machine.sh` for B/C/D, then the individual
`pp_tests/launch/launch_machine{A,B,C,D}.sh` scripts).

The pieces are reusable independently for a different topology/model too:
`ops/setup_machine.sh` for the per-machine environment bring-up (torch
pin, precompiled-kernel workarounds, humming-kernels), a signaling server
for hole-punch rendezvous (`python3 -m uvicorn
udp_holepunch.signaling_server:app --host 0.0.0.0 --port 8765` - any
public HTTP tunnel works, doesn't need a real public IP), and
`pp_tests/launch/launch_machine{A,B,C,D}.sh` as worked examples of the
actual `scripts/stage_server.py`/`scripts/launch_pp_stage.py` invocations
(model, quantization, KV cache sizing, MTP config).

## Status

Real, running, measured - not a design doc. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for current topology, the MTP
bring-up (three separate real bugs found and fixed to get speculative
decoding working across a synthetic multi-machine PP group), and measured
throughput/latency numbers. [`docs/history/`](docs/history/) has the
debugging history from earlier phases (GPT-OSS-120B, pre-MTP Qwen3.5).

## Known limitations

- `--enable-cudagraph` and MTP speculative decoding don't currently work
  together (a real `torch.compile`/Dynamo limitation tracing the MTP
  drafter) - the current MTP deployment runs the whole pipeline in eager
  mode.
- The UDP transport assumes real internet-routable NAT traversal is
  possible between machines; some NAT configurations (e.g. symmetric NAT
  on both sides) can still fail to punch through - a plain TCP backend
  (`vllm/transport/tcp_transport.py`) is available as a fallback for
  machines on a shared network.
