#!/usr/bin/env python3
"""Local proxy for the chat dashboard: serves chat_dashboard.html at / and
forwards /v1/* + /health to the live vLLM cluster - keeps the browser on a
single origin (avoids CORS entirely) and lets a plain static HTML page act
as a real streaming chat client against the deployed cluster.

Usage: python3 chat_proxy.py [--upstream http://127.0.0.1:8081] [--port 8090]
"""
import argparse
import http.server
import os
import socketserver
import sys
import urllib.error
import urllib.request

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_dashboard.html")


def make_handler(upstream: str):
    class ProxyHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._serve_html()
            elif self.path.startswith("/v1/") or self.path == "/health":
                self._proxy("GET")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.startswith("/v1/"):
                self._proxy("POST")
            else:
                self.send_error(404)

        def _serve_html(self):
            with open(HTML_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self, method):
            url = upstream + self.path
            content_length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(content_length) if content_length else None
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while True:
                        chunk = resp.read(256)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(e.read())
            except Exception as e:  # noqa: BLE001
                try:
                    self.send_error(502, str(e))
                except Exception:
                    pass
            self.close_connection = True

        def log_message(self, fmt, *args):
            sys.stderr.write("[chat_proxy] " + (fmt % args) + "\n")

    return ProxyHandler


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8081",
                     help="vLLM API base URL reachable from this machine (default: the existing SSH tunnel to the driver)")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    handler = make_handler(args.upstream.rstrip("/"))
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"[chat_proxy] serving chat dashboard on :{args.port}, proxying /v1/* + /health to {args.upstream}",
          file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
