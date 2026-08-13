# docs/history/

Point-in-time reports from earlier phases of this project, kept for the
debugging history (root causes, dead ends, what was actually tried) but
**superseded by [`docs/DEPLOYMENT.md`](../DEPLOYMENT.md)** for current
status. Don't follow these as setup instructions - topology, model, and
scripts have all changed since.

Rough chronology:

1. `README_GPTOSS_120B_UDP.md`, `README_GPTOSS_120B_CLUSTER.md`,
   `BLOCKER_REPORT.md`, `STATUS_AND_REPRODUCE.md` - GPT-OSS-120B on a
   3-machine cluster, first end-to-end UDP hole-punch PP success.
2. `README_PIPELINE_BUGFIX_VALIDATION.md` - root-cause and fix for a
   decode-degradation bug in the custom transport-backed pipeline.
3. `README_RUN_GPTOSS_CLUSTER.md` - operational run guide for the
   GPT-OSS-120B 3-machine cluster.
4. `QWEN3.5_DEPLOYMENT_REPORT.md` - model switched to
   Qwen3.5-122B-A10B-GPTQ-Int4, still 3-machine/16-layers-per-stage, no
   MTP yet.

Current state (4-machine/12-layers-per-stage, MTP speculative decoding,
`transport_runtime`-backed transport): `docs/DEPLOYMENT.md`.
