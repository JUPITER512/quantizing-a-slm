"""Ask every item (turn 1), then push back with the four follow-ups (turn 2).

Writes one JSON line per item and follow-up to results/<model>__<variant>.jsonl and
continues where it stopped if the run is interrupted.

    python scripts/run_pushback.py --model llama3.2:3b-instruct-q4_K_M --debug
    python scripts/run_pushback.py --model llama3.2:3b-instruct-q4_K_M --n-items 20
    python scripts/run_pushback.py --model llama3.2:3b-instruct-q4_K_M
    python scripts/run_pushback.py --model <tag> --variant reversed --control-only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so that `import cavein` works

from cavein.backend import Backend
from cavein.config import ENV_FILE, FOLLOWUPS_FILE, ITEMS_FILE, RESULTS_DIR, VARIANTS
from cavein.items import load_followups, load_items, select_items
from cavein.records import model_meta, out_file
from cavein.runner import debug, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask each item, then push back with four follow-ups.")
    parser.add_argument("--model", required=True, help="Ollama tag or OpenAI model name")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--variant", choices=VARIANTS, default="main")
    parser.add_argument("--control-only", action="store_true", help="only the 100 control items")
    parser.add_argument("--n-items", type=int, help="random sample of N items (writes a __pilotN file)")
    parser.add_argument("--suffix", help="extra file-name part, e.g. det1 or det2")
    parser.add_argument("--debug", action="store_true", help="one item, print everything, write nothing")
    parser.add_argument("--precision", help="set the precision instead of reading it from the tag")
    parser.add_argument("--family", help="set the family instead of reading it from the tag")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--items", type=Path, default=ITEMS_FILE)
    parser.add_argument("--followups", type=Path, default=FOLLOWUPS_FILE)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    args = parser.parse_args(argv)

    items = load_items(args.items, args.variant)
    followups = load_followups(args.followups, args.variant)
    backend = Backend(args.backend, args.model, args.host)
    try:
        if args.debug:
            item = select_items(items, 1, args.control_only)[0]
            debug(backend, item, followups)
            return 0
        selected = select_items(items, args.n_items, args.control_only)
        suffix = args.suffix
        if not suffix and args.n_items:
            suffix = f"pilot{args.n_items}"
        path = out_file(args.out_dir, args.model, args.variant, args.control_only, suffix)
        meta = model_meta(args.model, args.backend, args.precision, args.family)
        stats = run(backend, selected, followups, meta, args.variant, path, args.env_file)
        print(f"done: {stats['records']} records, {stats['calls']} calls, {stats['errors']} errors -> {path}")
        if stats["errors"]:
            return 1
        return 0
    finally:
        backend.unload()


if __name__ == "__main__":
    sys.exit(main())
