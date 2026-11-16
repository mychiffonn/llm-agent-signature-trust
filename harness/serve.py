"""Serve summary.html (and nothing else) on localhost:1306.

  nohup .venv/bin/python -m harness.serve > serve.log 2>&1 &

The file is read on every request, so a regenerated summary shows up on reload.
To view it from another machine: ssh -L 1306:localhost:1306 <server>
"""
from __future__ import annotations

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(__file__), "..", "summary.html")
HOST, PORT = os.environ.get("EPI_HOST", "127.0.0.1"), int(os.environ.get("EPI_PORT", "1306"))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/summary.html"):
            self.send_error(404)
            return
        try:
            body = open(PAGE, "rb").read()
        except FileNotFoundError:
            body = b"summary.html not generated yet: run python -m harness.report"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"serving summary.html on http://{HOST}:{PORT}/")
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
