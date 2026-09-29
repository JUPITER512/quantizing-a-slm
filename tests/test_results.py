"""Tests for cavein/results.py."""
import json
from pathlib import Path

import pandas as pd

from cavein import results


def test_run_of_file_names():
    assert results.run_of(Path("m_x__main.jsonl")) == "main"
    assert results.run_of(Path("m_x__reversed.jsonl")) == "reversed"
    assert results.run_of(Path("m_x__main__det2.jsonl")) == "det2"
    assert results.run_of(Path("m_x__main__control.jsonl")) == "control"


def record(family, precision, item, condition, correct0, harmful, c0=0.8):
    return {"family": family, "precision": precision, "item_id": item, "condition": condition,
            "correct_0": correct0, "harmful": harmful, "c0": c0}


def test_paired_keeps_only_items_correct_at_both_precisions():
    rows = [record("f", "q4_K_M", "i1", "user", True, True), record("f", "fp16", "i1", "user", True, False),
            record("f", "q4_K_M", "i2", "user", True, True), record("f", "fp16", "i2", "user", False, None),
            record("f", "q4_K_M", "i3", "user", True, None), record("f", "fp16", "i3", "user", True, True)]
    df = pd.DataFrame(rows).astype({"correct_0": "boolean", "harmful": "boolean"})
    pairs = results.paired(df, "user", "q4_K_M", "fp16")
    assert list(pairs["item_id"]) == ["i1"]
    assert bool(pairs["harmful_a"][0]) and not bool(pairs["harmful_b"][0])


def test_load_results_skips_pilots_errors_and_keeps_last(tmp_path):
    base = {"model": "m:q4_K_M", "variant": "main", "item_id": "i1", "condition": "user", "error": None,
            "correct_0": True, "harmful": True, "went_to_x": True, "beneficial": False, "flipped": True}
    lines = [dict(base, harmful=None, error="turn2: boom"), dict(base, harmful=False)]
    (tmp_path / "m_q4__main.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    (tmp_path / "m_q4__main__pilot20.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
    df = results.load_results(tmp_path)
    assert len(df) == 1 and df["run"].iloc[0] == "main" and bool(df["harmful"].iloc[0]) is False
