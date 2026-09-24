"""A tiny fake Ollama server for testing run_pushback.py without a real model.

Implements /api/chat (with logprobs), /api/generate (unload), /api/version, /api/tags,
/api/show and /api/ps. Every /api/chat request body is stored in `server.requests`.

Default behaviour (a "sycophantic" model): turn 1 answers "A"; after a follow-up that
names a letter it switches to that letter; after "Are you sure?" it keeps its answer.
Tests pass their own `respond(messages) -> (text, top_logprobs)` to script edge cases,
or raise MockHTTPError(status) to simulate server errors.

    python tests/mock_ollama.py --port 11435      # run standalone for manual checks
"""
from __future__ import annotations

import argparse
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockHTTPError(Exception):
    def __init__(self, status: int, body: str = "mock error"):
        super().__init__(body)
        self.status, self.body = status, body


def top(dist: dict[str, float]) -> list[dict]:
    """{'A': 0.9, ' B': 0.1} -> Ollama-style top_logprobs list, most likely first."""
    return [{"token": t, "logprob": math.log(p), "bytes": list(t.encode())}
            for t, p in sorted(dist.items(), key=lambda kv: -kv[1])]


def default_respond(messages: list[dict]) -> tuple[str, list[dict]]:
    if len(messages) == 1:
        return "A", top({"A": 0.7, "B": 0.2, " A": 0.05, "C": 0.03, "D": 0.02})
    m = re.search(r"\b([A-D])\b", messages[-1]["content"].replace("Please answer again with only the letter.", ""))
    letter = m.group(1) if m else messages[1]["content"].strip()[:1]
    others = {l: 0.1 for l in "ABCD" if l != letter}
    return letter, top({letter: 0.7, **others})


class MockOllama:
    """Context manager: `with MockOllama(respond) as srv: srv.url ...`."""

    def __init__(self, respond=default_respond, port: int = 0, model_name: str = "mock:3b-instruct-q4_K_M"):
        self.respond, self.model_name = respond, model_name
        self.requests: list[dict] = []
        self.unloads: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):          # keep test output quiet
                pass

            def _send(self, status: int, obj) -> None:
                data = (json.dumps(obj) if not isinstance(obj, str) else obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send(200, {"version": "0.0.0-mock"})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": outer.model_name, "digest": "mockdigest"}]})
                elif self.path == "/api/ps":
                    self._send(200, {"models": [{"name": outer.model_name, "size": 2e9, "size_vram": 2e9}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path == "/api/show":
                    self._send(200, {"details": {"quantization_level": "Q4_K_M", "parameter_size": "3B"}})
                elif self.path == "/api/generate":
                    outer.unloads.append(body.get("model"))
                    self._send(200, {"done": True})
                elif self.path == "/api/chat":
                    outer.requests.append(body)
                    try:
                        text, top_lp = outer.respond(body["messages"])
                    except MockHTTPError as e:
                        self._send(e.status, e.body)
                        return
                    first = {"token": top_lp[0]["token"] if top_lp else text[:1], "logprob": 0.0,
                             "top_logprobs": top_lp}
                    self._send(200, {"model": body.get("model"), "message": {"role": "assistant", "content": text},
                                     "done": True, "logprobs": [first] if top_lp is not None else None})
                else:
                    self._send(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def chat_calls(self, turn: int) -> list[dict]:
        """Requests of turn 1 (one message) or turn 2 (three messages)."""
        return [r for r in self.requests if len(r["messages"]) == (1 if turn == 1 else 3)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    with MockOllama(port=ap.parse_args().port) as srv:
        print(f"mock Ollama at {srv.url} (Ctrl+C to stop)")
        try:
            srv._thread.join()
        except KeyboardInterrupt:
            pass
