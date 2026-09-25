"""Unit tests for scripts/analyze.py on small synthetic data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze as A


# ---------------------------------------------------------------- helpers

def test_run_of_file_names():
    assert A.run_of(Path("m_x__main.jsonl")) == "main"
    assert A.run_of(Path("m_x__reversed.jsonl")) == "reversed"
    assert A.run_of(Path("m_x__main__det2.jsonl")) == "det2"
    assert A.run_of(Path("m_x__main__control.jsonl")) == "control"


def test_cohens_h():
    assert A.cohens_h(0.5, 0.5) == 0
    assert A.cohens_h(0.6, 0.4) == pytest.approx(0.4027, abs=1e-4)


def test_ece_perfect_and_known_value():
    assert A.ece([1.0, 1.0], [1, 1]) == 0
    # all at confidence 0.9, accuracy 0.5 -> |0.5 - 0.9| = 0.4
    assert A.ece([0.9] * 4, [1, 1, 0, 0]) == pytest.approx(0.4)


def test_mcnemar_exact_uses_discordant_pairs():
    assert A.mcnemar_p(0, 0) == 1.0
    assert A.mcnemar_p(10, 0) == pytest.approx(2 * 0.5 ** 10)   # exact binomial, two-sided


def test_wilson_ci_known_values():
    lo, hi = A.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = A.wilson_ci(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.2


def test_bootstrap_is_deterministic_and_covers_point():
    num = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0], float)
    lo, hi = A.boot_ratio_ci(num, np.ones_like(num))
    assert (lo, hi) == A.boot_ratio_ci(num, np.ones_like(num))
    assert lo <= num.mean() <= hi


def test_boot_diff_ci_zero_when_identical():
    x = np.array([1, 0, 2, 1], float)
    lo, hi = A.boot_diff_ci(x, x, np.array([1, 1, 2, 1], float))
    assert lo == 0 and hi == 0


# ---------------------------------------------------------------- pairing and comparison

def rec(family, precision, item, condition, correct0, harmful, c0=0.8):
    return {"family": family, "precision": precision, "item_id": item, "condition": condition,
            "correct_0": correct0, "harmful": harmful, "c0": c0}


def test_paired_keeps_only_items_correct_at_both_precisions():
    rows = [rec("f", "q4_K_M", "i1", "user", True, True), rec("f", "fp16", "i1", "user", True, False),
            rec("f", "q4_K_M", "i2", "user", True, True), rec("f", "fp16", "i2", "user", False, None),
            rec("f", "q4_K_M", "i3", "user", True, None), rec("f", "fp16", "i3", "user", True, True)]
    df = pd.DataFrame(rows).astype({"correct_0": "boolean", "harmful": "boolean"})
    p = A.paired(df, "user", "q4_K_M", "fp16")
    assert list(p["item_id"]) == ["i1"] and bool(p["harmful_a"][0]) and not bool(p["harmful_b"][0])


def test_compare_counts_and_direction():
    pairs = pd.DataFrame({"family": "f", "item_id": [f"i{k}" for k in range(6)],
                          "harmful_a": [True, True, True, False, False, True],
                          "harmful_b": [True, False, False, False, True, True],
                          "c0_a": 0.9, "c0_b": 0.9})
    r = A.compare(pairs)
    assert (r["both_harmful"], r["only_a"], r["only_b"], r["neither"]) == (2, 2, 1, 1)
    assert r["hfr_a"] == pytest.approx(4 / 6) and r["diff_a_minus_b"] == pytest.approx(1 / 6)
    assert r["p_mcnemar_exact"] == pytest.approx(1.0)


# ---------------------------------------------------------------- loading

def test_load_results_skips_pilots_errors_and_keeps_last(tmp_path):
    base = {"model": "m:q4_K_M", "variant": "main", "item_id": "i1", "condition": "user", "error": None,
            "correct_0": True, "harmful": True, "went_to_x": True, "beneficial": False, "flipped": True}
    lines = [dict(base, harmful=None, error="turn2: boom"), dict(base, harmful=False)]
    (tmp_path / "m_q4__main.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    (tmp_path / "m_q4__main__pilot20.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
    df = A.load_results(tmp_path)
    assert len(df) == 1 and df["run"].iloc[0] == "main" and bool(df["harmful"].iloc[0]) is False
