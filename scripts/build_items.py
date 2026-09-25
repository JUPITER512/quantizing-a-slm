"""Build data/items.jsonl and data/followups.json (an existing file is only compared, never overwritten).

    python scripts/build_items.py --n-mmlupro 150 --n-arc 150 --n-control 100 --seed 42
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so that `import cavein` works

from cavein.config import DATA_DIR
from cavein.dataset import build_items, load_sources, print_summary, to_json, to_jsonl, write_or_check
from cavein.prompts import build_followups


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build data/items.jsonl and data/followups.json.")
    parser.add_argument("--n-mmlupro", type=int, default=150)
    parser.add_argument("--n-arc", type=int, default=150)
    parser.add_argument("--n-control", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    mmlu_rows, arc_rows = load_sources()
    items = build_items(mmlu_rows, arc_rows, args.n_mmlupro, args.n_arc, args.n_control, args.seed)
    print_summary(items)
    ok_items = write_or_check(args.out_dir / "items.jsonl", to_jsonl(items))
    ok_followups = write_or_check(args.out_dir / "followups.json", to_json(build_followups()))
    if ok_items and ok_followups:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
