"""Simple local chatbot server: serves a static chat UI and proxies
/v1/chat/completions to the vLLM API running on Machine C (akun3), reached
through the SSH tunnel already forwarded to 127.0.0.1:8081. Same-origin
(UI + API proxy both on this one port) so no CORS setup is needed - just
forward THIS port via VSCode's port forwarding to use it from a local browser.

Usage:
    python3 chatbot_server.py [--port 7860] [--upstream http://127.0.0.1:8081]
"""
import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = "http://127.0.0.1:8081"
INDEX_HTML = (Path(__file__).resolve().parent / "chatbot_ui.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        req = urllib.request.Request(
            f"{UPSTREAM}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            payload = e.read()
            status = e.code
        except Exception as e:
            payload = json.dumps({"error": str(e)}).encode()
            status = 502
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--upstream", default=UPSTREAM)
    args = ap.parse_args()
    UPSTREAM = args.upstream

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"chatbot server on http://0.0.0.0:{args.port} -> proxying to {UPSTREAM}")
    server.serve_forever()


if __name__ == "__main__":
    main()
