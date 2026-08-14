---
name: feedback-kaggle-working-placement
description: "Where to place project files vs large downloads in this Kaggle sandbox; why 'df shows space free' is not safe evidence of real quota; and why large files must NEVER move machine-to-machine over SSH (5GB/day total bandwidth budget) — corrected 2026-08-03 after real crashes and a direct user correction"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 13e4e60f-dd1c-4271-9591-133f415e9ab8
  modified: 2026-08-03T08:25:02.465Z
---

Put all working repos/code/project artifacts in `/kaggle/working`. Put
genuinely large downloads (multi-GB+ model checkpoints, e.g. GPT-OSS
120B's weights) under `/` (root filesystem, outside `/kaggle/working`),
never under `/kaggle/working` itself. Symlink into `/kaggle/working` if
the user needs convenient access from there.

**Why, corrected — the user stated this directly on 2026-08-03**:
`/kaggle/working` is a small, dedicated volume (`/dev/loop1`, ~20GB).
**Downloading a large model directly into `/kaggle/working` fills this
volume and kills the session outright** — this is what caused a prior
session to fail and lose its work before it could be backed up (the
user's own words: "jangan download model besar di kaggle/working karna
itu bikin sessionnya mati, karna itu penyebab session sebelumnya failed
dan aku ga bisa back up datanya"). This is a session-crashing failure
mode, not just a slow/inconvenient one — treat it as a hard rule, not a
style preference.

**Persistence is more complicated than "one side persists, one doesn't"
— confirmed by direct observation in the 2026-08-03 session**:
- `/kaggle/working` (code, e.g. the `vllm/` checkout with all transport
  work) *did* survive a session reset that session.
- `/data/models/...` (a 61GB checkpoint placed under `/`, outside
  `/kaggle/working`) did *not* survive that same reset — it was gone
  afterward and had to be re-sourced from other machines.
- So: `/kaggle/working` surviving a reset is not guaranteed either — it
  has survived at least once, but don't treat it as a durable backup
  target. `/` is not reliably persistent across resets. **Neither
  location is a safe permanent store for anything irreplaceable** —
  large downloads on `/` should be treated as disposable/re-fetchable
  (e.g. re-downloadable from Hugging Face), not as the only copy of
  something that took real effort to produce.

**How to apply**: default project/code work to `/kaggle/working/...`.
For any download or artifact that could plausibly exceed a few GB,
route it to `/...` (outside `/kaggle/working`) without exception — even
under time pressure, even if `/kaggle/working` looks like it has
headroom at the moment. Before starting a large download, sanity-check
the target path is not under `/kaggle/working`. If a large asset needs
regenerating after a reset, prefer re-fetching from its original source
(HF Hub, etc.) over trying to recover a lost local copy, unless another
live machine still has it (as happened this session — two sibling
machines still had the checkpoint and the real `humming_fix`/
`transport_runtime` code, reachable via SSH, which is how that session's
work was actually restored).

**SECOND, DISTINCT CRASH — same session, later — `df -h /` free space is
NOT a safe proxy for real quota, even when writing to `/` correctly**:
On two remote Kaggle machines (akun5, akun6, accessed over SSH), `/` was
used correctly per the rule above — never `/kaggle/working` — yet both
crashed with a real Kaggle-side error ("Your notebook tried to use more
disk space than is available") after the *cumulative* writes on each
machine reached roughly 100GB+: the full 61GB checkpoint plus 2-3
extracted ~20-23GB per-stage shards, kept simultaneously instead of
deleting the full checkpoint after each shard was extracted from it.
`df -h /` had reported ~1TB free the entire time and was checked before
starting — it was actively misleading. On these container-based
notebooks, `df` reports the underlying shared host's overlay filesystem
capacity, not this specific notebook's actual enforced disk quota;
Kaggle enforces its own (much smaller, exact number unconfirmed) cap
per-notebook regardless of what `df` shows, and it can be blown through
while `df` still claims hundreds of GB free. One machine's last write
(a safetensors save) failed with a raw I/O error ("Bad address (os error
14)") immediately before it went dark — a real disk-full symptom, not a
network fluke; both machines' SSH tunnels dropped within the same
couple of minutes.

**How to apply (in addition to the rule above)**: never trust `df -h`
alone as evidence a large write is safe on these Kaggle-hosted machines.
Keep *cumulative* large-file footprint per machine as small as possible
at every point in time — after extracting what you need from a large
downloaded checkpoint (e.g. per-stage shards), delete or move aside the
original full download rather than letting both coexist "just in case."
Do not stack multiple independent multi-GB+ artifacts (full checkpoint +
several extracted shards + backups) on one machine without deleting
intermediates first. If a machine goes silent (SSH reset/timeout) right
after or during a large write, suspect disk exhaustion first, not just
network/tunnel flakiness — check the operation's own error output (not
just connectivity) before concluding it's [[blocker-2-connectivity-loss]]-style infra flakiness with no local cause.

**THIRD crash, same session, later still — quantified the real ceiling,
and "keep the full checkpoint to extract a second shard from it" is
itself unsafe**: built `ops/setup_machine.sh` with a `--keep-full-checkpoint`
flag specifically so ONE machine could extract two different layer
ranges from a single 61GB download (avoiding a second 61GB re-download).
Used it on akun5: downloaded the 61GB checkpoint, extracted a 23GB shard
while keeping the 61GB original (peak ~84GB - this level worked fine,
repeatedly, across multiple machines), then immediately tried extracting
a second ~21GB shard from the same still-present 61GB original - peak
disk demand ~105GB (61+23+21) - and the machine crashed again with the
exact same `SafetensorError: ... Bad address (os error 14)` mid-write,
SSH dying seconds later. Two data points now bracket the real quota:
**~84GB peak has succeeded multiple times; ~105GB peak has failed twice.**
The actual cap is very likely a round number near 100GB, though the exact
value is still unconfirmed.

**How to apply**: treat ~85GB as the practical safe ceiling for
cumulative large-file footprint on one of these machines until a tighter
number is confirmed. Never keep a full checkpoint around to extract a
*second* shard from it if a first shard is already sitting on disk too -
transfer/consume the first shard and delete it before extracting the
second (or just re-download the full checkpoint fresh for the second
extraction - a 61GB re-download takes ~5min and is far cheaper than a
crash). `--keep-full-checkpoint` is only safe when nothing else large is
already on disk. Extracting two shards from one machine as an
optimization to save bandwidth is not worth the crash risk - prefer the
simpler, always-below-~85GB sequence: download, extract one shard,
delete the original, repeat if a second shard is needed.

**CRITICAL, stated directly by the user 2026-08-03 — never move large
files machine-to-machine over SSH; the user's total bandwidth budget is
5GB/day across ALL these machines combined**: mid-session, tried to
`rsync` a 22GB extracted checkpoint shard from one remote machine to
another over the SSH/zrok tunnel (to give the local machine a shard it
needed but didn't have the source checkpoint for). It was extremely slow
(~1.5-2MB/s, projecting to 3-4 hours for 22GB) and the connection kept
dropping mid-transfer (got 6.4GB through once, then reset). The user then
stated explicitly: "kamu jangan transfer file besar lewat ssh jg... aku
cuman punya 5gb bandwidth tiap hari" (don't transfer large files via ssh
either, I only have 5GB bandwidth per day). A single one of these
transfer attempts could have consumed 4-5x the user's entire daily
budget. This is a hard rule, not a performance tuning question - the
slowness observed was likely the bandwidth constraint itself manifesting,
not just tunnel flakiness.

**How to apply**: NEVER `scp`/`rsync`/transfer a multi-GB file between
two of these remote machines (or between a remote machine and this local
sandbox) over their SSH/zrok tunnels, even once, even "just this one
file." If a machine needs a large asset (e.g. a checkpoint shard) that
another machine already has, do NOT move it between them - instead have
the machine that NEEDS it fetch/derive it independently from the
original external source (e.g. re-download the full 61GB checkpoint from
Hugging Face directly on that machine, then extract its own shard
locally) even though this costs extra HF bandwidth - HF downloads in
this environment have consistently been fast (~200MB/s, ~5min for 61GB)
and apparently don't count against the same constrained 5GB/day budget
the SSH tunnels do. Small files (code via `rsync` of the `vllm`/
`humming_fix`/etc. source trees, config, .git) are fine - the rule is
specifically about multi-GB checkpoint/model-weight data. If ever unsure
whether a transfer counts as "large," treat anything over ~100MB crossing
an SSH tunnel between machines as suspect and find an HF-direct or
recompute-locally alternative instead.
