#!/usr/bin/env bash
# Runs setup_machine.sh against every machine listed in a config file, in
# parallel, so a full re-setup after a multi-machine reset is one command
# instead of re-typing the same steps per machine by hand.
#
# Config file format (default: ops/machines.conf), one machine per line:
#   NAME PORT PASSWORD [EXTRA_SETUP_MACHINE_ARGS...]
# Blank lines and lines starting with # are ignored.
#
# Example line for a stage-1 machine that should also fetch+extract its
# own checkpoint shard:
#   akun5 9195 ZNWae2bcyqGKuovVAViw8GKz --extract-stage 12:24:/data/stage1-checkpoint
#
# Usage:
#   ops/setup_all.sh [config_file] [-- extra args passed to every machine]
#
# Logs: ops/logs/<name>.log (one per machine, tailed live to the terminal
# with a [name] prefix via a background `tail -f` while jobs run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/machines.conf}"
shift || true
COMMON_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  COMMON_ARGS=("$@")
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "config file not found: $CONFIG_FILE" >&2
  echo "create it (see this script's header for the format) or pass a path as \$1" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"
PIDS=()
NAMES=()

while read -r name port password rest; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  logfile="$SCRIPT_DIR/logs/$name.log"
  echo "[setup_all] launching setup for $name (port $port) -> $logfile"
  # shellcheck disable=SC2086
  "$SCRIPT_DIR/setup_machine.sh" --port "$port" --password "$password" --name "$name" \
    $rest "${COMMON_ARGS[@]}" > "$logfile" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
done < "$CONFIG_FILE"

echo "[setup_all] ${#PIDS[@]} machine(s) running in parallel, waiting..."
FAILED=()
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[setup_all] ${NAMES[$i]}: DONE"
  else
    echo "[setup_all] ${NAMES[$i]}: FAILED - see $SCRIPT_DIR/logs/${NAMES[$i]}.log"
    FAILED+=("${NAMES[$i]}")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[setup_all] FAILED: ${FAILED[*]}" >&2
  exit 1
fi
echo "[setup_all] all machines set up successfully"
