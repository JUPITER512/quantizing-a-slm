"""Write the details of the model and the machine to env_info.txt."""
import datetime
import json
import platform
import subprocess

import requests

from cavein.backend import redact


def record_env(backend, meta, out_path, env_path):
    """Add Python, Ollama, model digest and GPU details to env_info.txt."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [f"\n=== {now}  {meta['model']}  -> {out_path.name}",
             f"python {platform.python_version()} | {platform.platform()}"]
    if backend.kind == "ollama":
        try:
            version = backend.session.get(backend.host + "/api/version", timeout=10).json().get("version")
            lines.append(f"ollama {version}")
            models = backend.session.get(backend.host + "/api/tags", timeout=10).json().get("models", [])
            digest = None
            for m in models:
                if m.get("name") == meta["model"]:
                    digest = m.get("digest")
                    break
            lines.append(f"digest {digest}")
            show = backend.session.post(backend.host + "/api/show", json={"model": meta["model"]}, timeout=30).json()
            lines.append("details " + json.dumps(show.get("details", {}), ensure_ascii=False))
        except requests.RequestException as e:
            lines.append(f"ollama info unavailable: {redact(e)}")
    else:
        lines.append(f"openai model {meta['model']} (API; seed is best-effort)")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10)
        lines.append("gpu " + result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        lines.append("gpu unknown")
    with open(env_path, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def record_split(backend, env_path):
    """Add how much of the loaded model sits on the GPU to env_info.txt."""
    if backend.kind != "ollama":
        return
    try:
        models = backend.session.get(backend.host + "/api/ps", timeout=10).json().get("models", [])
    except requests.RequestException:
        return
    for m in models:
        if m.get("name") == backend.model:
            size = m.get("size", 0)
            vram = m.get("size_vram", 0)
            share = 100 * vram / size if size else 0
            with open(env_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"loaded size {size / 1e9:.2f} GB, on GPU {vram / 1e9:.2f} GB ({share:.0f}%)\n")
