"""A small fake Ollama server for testing run_pushback.py without a real model.

By default the fake model answers "A" in turn 1 and then switches to whatever letter the
follow-up names. Tests can pass their own respond(messages) function.

    python tests/mock_ollama.py --port 11435
"""
import argparse
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockHTTPError(Exception):
    def __init__(self, status, body="mock error"):
        super().__init__(body)
        self.status = status
        self.body = body


def top(dist):
    """{'A': 0.9, ' B': 0.1} -> Ollama-style top_logprobs list, most likely token first."""
    result = []
    for token, prob in sorted(dist.items(), key=lambda pair: -pair[1]):
        result.append({"token": token, "logprob": math.log(prob), "bytes": list(token.encode())})
    return result


def default_respond(messages):
    if len(messages) == 1:
        return "A", top({"A": 0.7, "B": 0.2, " A": 0.05, "C": 0.03, "D": 0.02})
    followup = messages[-1]["content"].replace("Please answer again with only the letter.", "")
    match = re.search(r"\b([A-D])\b", followup)
    if match:
        letter = match.group(1)
    else:
        letter = messages[1]["content"].strip()[:1]
    dist = {letter: 0.7}
    for other in "ABCD":
        if other != letter:
            dist[other] = 0.1
    return letter, top(dist)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, status, data):
        if isinstance(data, str):
            body = data.encode()
        else:
            body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        mock = self.server.mock
        if self.path == "/api/version":
            self.send_json(200, {"version": "0.0.0-mock"})
        elif self.path == "/api/tags":
            self.send_json(200, {"models": [{"name": mock.model_name, "digest": "mockdigest"}]})
        elif self.path == "/api/ps":
            self.send_json(200, {"models": [{"name": mock.model_name, "size": 2e9, "size_vram": 2e9}]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        mock = self.server.mock
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/show":
            self.send_json(200, {"details": {"quantization_level": "Q4_K_M", "parameter_size": "3B"}})
        elif self.path == "/api/generate":
            mock.unloads.append(body.get("model"))
            self.send_json(200, {"done": True})
        elif self.path == "/api/chat":
            mock.requests.append(body)
            try:
                text, top_logprobs = mock.respond(body["messages"])
            except MockHTTPError as e:
                self.send_json(e.status, e.body)
                return
            if top_logprobs is None:
                logprobs = None
            else:
                first_token = top_logprobs[0]["token"] if top_logprobs else text[:1]
                logprobs = [{"token": first_token, "logprob": 0.0, "top_logprobs": top_logprobs}]
            self.send_json(200, {"model": body.get("model"), "message": {"role": "assistant", "content": text},
                                 "done": True, "logprobs": logprobs})
        else:
            self.send_json(404, {"error": "not found"})


class MockOllama:
    """Use it as: with MockOllama(respond) as server: ... server.url ..."""

    def __init__(self, respond=default_respond, port=0, model_name="mock:3b-instruct-q4_K_M"):
        self.respond = respond
        self.model_name = model_name
        self.requests = []
        self.unloads = []
        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.mock = self
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def chat_calls(self, turn):
        """Turn-1 requests have one message, turn-2 requests have three."""
        n_messages = 1 if turn == 1 else 3
        return [r for r in self.requests if len(r["messages"]) == n_messages]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11435)
    with MockOllama(port=parser.parse_args().port) as server:
        print(f"mock Ollama at {server.url} (Ctrl+C to stop)")
        try:
            server.thread.join()
        except KeyboardInterrupt:
            pass
