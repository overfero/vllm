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

transport_runtime/       framework-agnostic communication runtime that
                         wraps udp_holepunch/ (and a plain TCP backend)
                         behind one Backend interface - no vLLM
                         dependency either; see transport_runtime/README.md
                         and docs/ARCHITECTURE_DECISION.md for why this
                         layer was extracted out of the vLLM adapter

vllm/transport/          the vLLM-specific adapter: consumes
                         transport_runtime, replaces the pipeline-parallel
                         dimension of vLLM's real GroupCoordinator with a
                         transport-backed one right after local TP
                         bootstrap (pp_worker.py, pipeline_bootstrap.py),
                         plus a TransportExecutor for the driver rank
                         (rpc_executor.py) - see vllm/transport/README.md

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

Three-layer split (`udp_holepunch` -> `transport_runtime` -> `vllm/transport`)
so the hole-punch transport and its generic runtime wrapper stay reusable
outside vLLM entirely - `transport_runtime` has no vLLM or torch import in
it. See `docs/ARCHITECTURE_DECISION.md` for the full reasoning (why extract
at all, what was debated, what got reverted) and `transport_runtime/README.md`
for what actually migrated vs. what's still vendored directly in
`vllm/transport/`.

## Quick start

```bash
# On each machine: install this fork's vllm (needs a CUDA GPU)
pip install -e . --no-build-isolation   # see ops/setup_machine.sh for the
                                         # full real bring-up (torch pin,
                                         # precompiled-kernel workarounds,
                                         # humming-kernels)

# One machine runs the signaling server (small, CPU-only, needs a public
# HTTP endpoint - a tunnel like zrok/ngrok/cloudflared works fine, doesn't
# need a real public IP itself):
python3 -m uvicorn udp_holepunch.signaling_server:app --host 0.0.0.0 --port 8765

# Each machine launches its pipeline stage against that signaling URL -
# see pp_tests/launch/launch_machine{A,B,C,D}.sh for real, current
# example commands (model, quantization, KV cache sizing, MTP config) and
# docs/DEPLOYMENT.md for the full walkthrough.
```

For the specific Qwen3.5/4-machine cluster this repo is currently
validated against, `setup_cluster.sh` automates the whole bring-up
end-to-end (torch/vllm/humming-kernels install + checkpoint extraction on
every machine) from a `.env` with each machine's current SSH
port/password - see `.env.example`.

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
