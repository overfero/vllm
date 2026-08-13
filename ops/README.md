# ops/ - orchestrator-side machine setup

These scripts run FROM this sandbox (the orchestrator, Machine A) and set
up the remote machines over SSH: sync this whole repo, install torch/vllm/
humming-kernels, selectively download + extract each machine's checkpoint
stage. `setup_cluster.sh` (repo root) drives this for the current 4-machine
cluster end-to-end; use the scripts here directly for anything more
ad hoc (a single machine, a different stage split, re-running just one
stage).

Exists because every machine reset meant redoing the same ~8 manual steps
(torch upgrade, .so copy, vllm build, humming-kernels, checkpoint
download+extraction) by hand over SSH. Now it's one command, and it's
idempotent - safe to re-run any time, every stage checks current state
first (torch version, `pip show vllm`, checkpoint file presence) and skips
what's already done.

## Single machine

```bash
ops/setup_machine.sh --port 9194 --password 'xxx' --name machineB
```

Add checkpoint + stage extraction (auto-deletes the full raw download
afterward - see the disk-safety note in both this script and
`docs/DEPLOYMENT.md`):

```bash
ops/setup_machine.sh --port 9194 --password 'xxx' --name machineB \
    --extract-stage 12:24:/data/stage1-checkpoint
```

## All machines at once

```bash
cp ops/machines.conf.example ops/machines.conf
# edit ops/machines.conf with current port/password per machine
ops/setup_all.sh
```

Runs every machine's setup in parallel; logs land in `ops/logs/<name>.log`
(gitignored - regenerate, don't commit).

## Local machine (this sandbox, Machine A)

Not scripted the same way since it's the orchestrator itself and doesn't
need SSH/rsync to reach - `setup_cluster.sh` handles Machine A's own
torch/vllm/humming-kernels install and checkpoint extraction directly
(no SSH). If it ever needs redoing from scratch, follow the same steps
`setup_machine.sh`'s stages encode, run locally instead of over `rssh`.

## What actually gets synced to a remote machine

The entire repo (this checkout's root, i.e. everything `git ls-files`
would show minus `__pycache__`/`.pytest_cache`) - the vllm package itself,
`humming_fix/`, `transport_runtime/`, `udp_holepunch/`, `ops/`, `pp_tests/`,
`scripts/`, `_pysitecustomize/` all travel together in one `rsync`, landing
at the same absolute path (`/kaggle/working/vllm` by default, override via
`REMOTE_PROJECT_ROOT`) on the remote as this checkout occupies locally -
every path assumption in the launch scripts depends on that symmetry.

## Stage/topology reference

Current cluster: 4 machines, 48 layers, TP=2 per machine, PP=4 across
machines, 12 layers/stage. See `docs/DEPLOYMENT.md` for the full table and
`pp_tests/launch/` for the actual launch commands.
