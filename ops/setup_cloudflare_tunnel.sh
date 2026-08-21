#!/usr/bin/env bash
# Restore/(re)start the Cloudflare Tunnel (llm.overfero.dev -> local vLLM
# :8080) on a remote machine, using this project's backed-up tunnel
# credentials so `cloudflared tunnel login` never has to be redone after a
# Kaggle machine reset wipes the remote's /root/.cloudflared.
#
# Exists because Akun2 has reset mid-session multiple times, and manually
# reinstalling cloudflared + re-copying cert.pem/credentials.json + rewriting
# config.yml + restarting the tunnel by hand each time was repetitive and
# error-prone (see the earlier incident where a backgrounded `tunnel login`
# got killed when wrapped in `timeout` without `disown` - this script always
# backgrounds with nohup+disown so it survives the SSH session tearing down).
#
# Usage:
#   ./ops/setup_cloudflare_tunnel.sh --port 9194 --password '...'
#   ./ops/setup_cloudflare_tunnel.sh --port 9194 --password '...' \
#       --local-port 8080 --hostname llm.overfero.dev
#
# Prereqs: a local backup directory (default /kaggle/working/.cloudflared_backup,
# deliberately OUTSIDE this git repo) containing cert.pem and <tunnel-id>.json,
# produced once via `cloudflared tunnel login` + `cloudflared tunnel create`.

set -euo pipefail

PORT=""
PASSWORD=""
LOCAL_PORT="8080"
HOSTNAME_="llm.overfero.dev"
BACKUP_DIR="/kaggle/working/.cloudflared_backup"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --password) PASSWORD="$2"; shift 2 ;;
    --local-port) LOCAL_PORT="$2"; shift 2 ;;
    --hostname) HOSTNAME_="$2"; shift 2 ;;
    --backup-dir) BACKUP_DIR="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$PORT" && -n "$PASSWORD" ]] || { echo "need --port and --password (see --help)" >&2; exit 1; }

CRED_JSON="$(ls "$BACKUP_DIR"/*.json 2>/dev/null | head -1)"
CERT_PEM="$BACKUP_DIR/cert.pem"
[[ -f "$CRED_JSON" && -f "$CERT_PEM" ]] || {
  echo "backup files missing in $BACKUP_DIR (expected cert.pem and <tunnel-id>.json)" >&2
  exit 1
}
TUNNEL_ID="$(basename "$CRED_JSON" .json)"

ssh_cmd() { sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" root@127.0.0.1 "$@"; }
scp_file() { sshpass -p "$PASSWORD" scp -o StrictHostKeyChecking=no -P "$PORT" "$1" "root@127.0.0.1:$2"; }

echo "[cf-tunnel] target tunnel id: $TUNNEL_ID  ->  https://${HOSTNAME_} -> 127.0.0.1:${LOCAL_PORT}"

echo "[cf-tunnel] ensuring cloudflared is installed on remote..."
ssh_cmd '
  set -e
  if ! command -v cloudflared >/dev/null 2>&1; then
    curl -fsSL -o /usr/local/bin/cloudflared \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /usr/local/bin/cloudflared
  fi
  cloudflared --version
'

echo "[cf-tunnel] restoring credentials..."
ssh_cmd "mkdir -p /root/.cloudflared"
scp_file "$CERT_PEM" "/root/.cloudflared/cert.pem"
scp_file "$CRED_JSON" "/root/.cloudflared/${TUNNEL_ID}.json"

echo "[cf-tunnel] writing config.yml..."
ssh_cmd "cat > /root/.cloudflared/config.yml <<CFEOF
tunnel: ${TUNNEL_ID}
credentials-file: /root/.cloudflared/${TUNNEL_ID}.json
ingress:
  - hostname: ${HOSTNAME_}
    service: http://127.0.0.1:${LOCAL_PORT}
  - service: http_status:404
CFEOF"

echo "[cf-tunnel] (re)starting tunnel (nohup+disown so it survives this SSH session closing)..."
ssh_cmd "
  pkill -f 'cloudflared tunnel run' 2>/dev/null || true
  sleep 1
  cd /root && setsid nohup cloudflared tunnel run ${TUNNEL_ID} > /var/log/cloudflared.log 2>&1 < /dev/null &
  sleep 3
  echo '--- last 15 lines of /var/log/cloudflared.log ---'
  tail -n 15 /var/log/cloudflared.log
"

echo "[cf-tunnel] done. Verify with: curl -s https://${HOSTNAME_}/health"
