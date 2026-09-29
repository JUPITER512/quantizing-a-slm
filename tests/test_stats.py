"""Tests for cavein/stats.py."""
import numpy as np
import pandas as pd
import pytest

from cavein import stats


def test_mcnemar_exact_uses_discordant_pairs():
    assert stats.mcnemar_p(0, 0) == 1.0
    assert stats.mcnemar_p(10, 0) == pytest.approx(2 * 0.5 ** 10)


def test_wilson_ci_known_values():
    lo, hi = stats.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = stats.wilson_ci(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.2


def test_bootstrap_is_deterministic_and_covers_point():
    num = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0], float)
    lo, hi = stats.boot_ratio_ci(num, np.ones_like(num))
    assert (lo, hi) == stats.boot_ratio_ci(num, np.ones_like(num))
    assert lo <= num.mean() <= hi


def test_boot_diff_ci_zero_when_identical():
    x = np.array([1, 0, 2, 1], float)
    lo, hi = stats.boot_diff_ci(x, x, np.array([1, 1, 2, 1], float))
    assert lo == 0 and hi == 0


def test_compare_counts_and_direction():
    pairs = pd.DataFrame({"family": "f", "item_id": [f"i{k}" for k in range(6)],
                          "harmful_a": [True, True, True, False, False, True],
                          "harmful_b": [True, False, False, False, True, True],
                          "c0_a": 0.9, "c0_b": 0.9})
    result = stats.compare(pairs)
    assert (result["both_harmful"], result["only_a"], result["only_b"], result["neither"]) == (2, 2, 1, 1)
    assert result["hfr_a"] == pytest.approx(4 / 6) and result["diff_a_minus_b"] == pytest.approx(1 / 6)
    assert result["p_mcnemar_exact"] == pytest.approx(1.0)
