"""Send chat requests to Ollama (or the OpenAI API) with the fixed decoding settings."""
import os
import re
import time

import requests

from cavein.config import NUM_PREDICT, SEED, TOP_LOGPROBS

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
TIMEOUT = 900
MAX_TRIES = 5


def redact(text):
    """Hide anything that looks like an API key."""
    text = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***REDACTED***", str(text))
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        text = text.replace(key, "***REDACTED***")
    return text


class Backend:
    def __init__(self, kind, model, host):
        self.kind = kind
        self.model = model
        self.host = host.rstrip("/")
        self.session = requests.Session()
        if kind == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit("OPENAI_API_KEY is not set.")
            self.session.headers["Authorization"] = "Bearer " + key

    def payload(self, messages):
        if self.kind == "ollama":
            return {"model": self.model, "messages": messages, "stream": False, "think": False,
                    "options": {"temperature": 0, "seed": SEED, "num_predict": NUM_PREDICT},
                    "logprobs": True, "top_logprobs": TOP_LOGPROBS}
        return {"model": self.model, "messages": messages, "temperature": 0, "seed": SEED,
                "logprobs": True, "top_logprobs": TOP_LOGPROBS, "max_completion_tokens": NUM_PREDICT}

    def ask(self, messages):
        """Send one request. Retries on connection errors, 429 and 5xx; raises RuntimeError otherwise."""
        if self.kind == "ollama":
            url = self.host + "/api/chat"
        else:
            url = OPENAI_URL
        body = self.payload(messages)
        last_error = ""
        for attempt in range(MAX_TRIES):
            start = time.perf_counter()
            try:
                response = self.session.post(url, json=body, timeout=TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code == 200:
                return self.parse(response.json(), time.perf_counter() - start)
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code != 429 and response.status_code < 500:
                break
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(redact(last_error))

    def parse(self, data, latency):
        if self.kind == "ollama":
            text = data.get("message", {}).get("content", "")
            positions = data.get("logprobs") or []
        else:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            logprobs = choice.get("logprobs") or {}
            positions = logprobs.get("content") or []
        top = positions[0].get("top_logprobs") if positions else None
        return {"text": text, "top": top, "raw_json": data, "latency_s": round(latency, 3)}

    def unload(self):
        # free the GPU memory after a run
        if self.kind == "ollama":
            try:
                self.session.post(self.host + "/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=60)
            except requests.RequestException:
                pass
