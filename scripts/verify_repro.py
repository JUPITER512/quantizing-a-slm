"""Offline reproducibility check (no model, no GPU, no network).

1. frozen inputs: data/items.jsonl and data/followups.json have the SHA-256 of the pre-registration commit
2. results: every non-pilot file is valid JSONL with the expected line count, no errors, no duplicates
3. results/ is unchanged since the git tag data-frozen (skipped if git or the tag is not available)
4. analysis: scripts/analyze.py rebuilt into a temporary folder is byte-identical to analysis/*.csv
5. figures: scripts/make_figures.py rebuilt from the rebuilt CSVs is byte-identical to figures/*.svg

    python scripts/verify_repro.py
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so that `import cavein` works

from cavein.config import ANALYSIS_DIR, CONDITIONS, FIGURES_DIR, FOLLOWUPS_FILE, ITEMS_FILE, RESULTS_DIR, ROOT

FROZEN = {
    ITEMS_FILE: "d712731f89f0344b4c50f1bf3c573ac1bf8e926fdadea4a35ce3bc02f6aa4c25",
    FOLLOWUPS_FILE: "2f7c58d1a472ef11c054c8247d15eb691383cda2c14d9f474aa5fd1f9355bbad",
}
LINES = {"main": 1200, "reversed": 400, "para1": 400, "para2": 400, "det1": 200, "det2": 200}
N_FILES = {"main": 18, "reversed": 6, "para1": 6, "para2": 6, "det1": 1, "det2": 1}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_name(path):
    parts = path.stem.split("__")
    return parts[2] if len(parts) > 2 else parts[1]


def check_frozen():
    problems = []
    for path, expected in FROZEN.items():
        if sha256(path) != expected:
            problems.append(f"{path.name}: SHA-256 differs from the pre-registration commit")
    return problems


def check_results(results_dir):
    problems = []
    files = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        if "__pilot" in path.name:
            continue
        run = run_name(path)
        files[run] = files.get(run, 0) + 1
        keys = set()
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"{path.name} line {number}: not valid JSON")
                continue
            if record.get("error"):
                problems.append(f"{path.name} line {number}: error record")
            if record.get("condition") not in CONDITIONS:
                problems.append(f"{path.name} line {number}: unknown condition {record.get('condition')!r}")
            key = (record.get("model"), record.get("item_id"), record.get("condition"))
            if key in keys:
                problems.append(f"{path.name} line {number}: duplicate {key}")
            keys.add(key)
        if len(lines) != LINES.get(run, -1):
            problems.append(f"{path.name}: {len(lines)} lines, expected {LINES.get(run)}")
    if files != N_FILES:
        problems.append(f"result files per run {files}, expected {N_FILES}")
    return problems, sum(files.values())


def check_tag(results_dir):
    try:
        out = subprocess.run(["git", "diff", "--name-only", "data-frozen", "--", str(results_dir)], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    changed = [name for name in out.stdout.splitlines() if name.endswith(".jsonl") and "__pilot" not in name]
    return [f"changed since data-frozen: {name}" for name in changed]


def compare(folder_a, folder_b, pattern):
    problems = []
    names_a = sorted(p.name for p in folder_a.glob(pattern))
    names_b = sorted(p.name for p in folder_b.glob(pattern))
    if names_a != names_b:
        problems.append(f"different file lists: {sorted(set(names_a) ^ set(names_b))}")
    for name in sorted(set(names_a) & set(names_b)):
        if sha256(folder_a / name) != sha256(folder_b / name):
            problems.append(f"{name}: rebuilt file differs")
    return problems, len(names_a)


def main(argv=None):
    parser = argparse.ArgumentParser(description="offline reproducibility check")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--skip-rebuild", action="store_true", help="only check data and results (fast)")
    args = parser.parse_args(argv)
    failed = False

    def report(title, problems, detail=""):
        nonlocal failed
        if problems is None:
            print(f"SKIP  {title}{detail}")
            return
        print(f"{'PASS' if not problems else 'FAIL'}  {title}{detail}")
        for problem in problems[:20]:
            print("        " + problem)
        failed = failed or bool(problems)

    report("frozen inputs match the pre-registration commit", check_frozen())
    problems, n_files = check_results(args.results_dir)
    report("result files complete, valid, error-free, duplicate-free", problems, f" ({n_files} files)")
    tag_problems = check_tag(args.results_dir)
    report("results/ unchanged since tag data-frozen", tag_problems,
           "" if tag_problems is not None else " (git or tag not available)")

    if not args.skip_rebuild:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scripts = Path(__file__).resolve().parent
            subprocess.run([sys.executable, str(scripts / "analyze.py"), "--results-dir", str(args.results_dir),
                            "--out-dir", str(tmp / "analysis")], check=True, capture_output=True)
            problems, n = compare(ANALYSIS_DIR, tmp / "analysis", "*.csv")
            report("analysis/*.csv rebuilt byte-identical", problems, f" ({n} tables)")
            subprocess.run([sys.executable, str(scripts / "make_figures.py"), "--analysis-dir", str(tmp / "analysis"),
                            "--out-dir", str(tmp / "figures")], check=True, capture_output=True)
            problems, n = compare(FIGURES_DIR, tmp / "figures", "*.svg")
            report("figures/*.svg rebuilt byte-identical", problems, f" ({n} figures)")

    print("\nRESULT: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
