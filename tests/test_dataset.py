"""Tests for cavein/dataset.py on made-up rows (no dataset download)."""
import random
from collections import Counter

import pytest

from cavein import dataset

CATEGORIES = [f"cat{i:02d}" for i in range(14)]


def mmlu_rows(per_category=20):
    rows = []
    qid = 0
    for category in CATEGORIES:
        for _ in range(per_category):
            options = [f"{category} option {qid}-{k}" for k in range(10)]
            rows.append({"question_id": qid, "question": f" Question {qid}? ", "options": options,
                         "answer": "ABCDEFGHIJ"[qid % 10], "answer_index": qid % 10,
                         "category": category, "src": "test"})
            qid += 1
    return rows


def arc_rows(n=200):
    rows = []
    for i in range(n):
        if i % 3 == 0:
            labels = ["1", "2", "3", "4"]  # some ARC items use numbers instead of letters
        else:
            labels = ["A", "B", "C", "D"]
        rows.append({"id": f"Mercury_{i:05d}", "question": f"Science {i}?",
                     "choices": {"text": [f"arc {i} choice {k}" for k in range(4)], "label": labels},
                     "answerKey": labels[i % 4]})
    return rows


@pytest.fixture(scope="module")
def items():
    return dataset.build_items(mmlu_rows(), arc_rows(), 150, 150, 100, 42)


def test_mmlu_candidate_drops_na_and_duplicates():
    row = {"question_id": 7, "question": "Q", "category": "computer science", "answer_index": 1,
           "options": ["a", "gold", "N/A", "", "a", "b", "gold ", "c"]}
    c = dataset.mmlu_pro_candidate(row)
    assert c["gold_text"] == "gold"
    assert c["distractors"] == ["a", "b", "c"]
    assert c["item_id"] == "mmlupro_computer_science_00007"


def test_mmlu_candidate_rejects_too_few_distractors():
    row = {"question_id": 1, "question": "Q", "category": "law", "answer_index": 0,
           "options": ["gold", "a", "a", "N/A"]}
    assert dataset.mmlu_pro_candidate(row) is None


def test_arc_candidate_maps_numeric_labels():
    row = {"id": "X_1", "question": "Q", "answerKey": "3",
           "choices": {"text": ["w", "x", "y", "z"], "label": ["1", "2", "3", "4"]}}
    c = dataset.arc_candidate(row)
    assert c["gold_text"] == "y" and sorted(c["distractors"]) == ["w", "x", "z"]


def test_arc_candidate_requires_exactly_four_options():
    row = {"id": "X_2", "question": "Q", "answerKey": "A",
           "choices": {"text": ["w", "x", "y"], "label": ["A", "B", "C"]}}
    assert dataset.arc_candidate(row) is None


def test_counts_per_source(items):
    assert Counter(it["source"] for it in items) == {"mmlu_pro": 150, "arc": 150}
    assert len({it["item_id"] for it in items}) == 300


def test_mmlu_stratified_10_or_11_per_category(items):
    per_category = Counter(it["subject"] for it in items if it["source"] == "mmlu_pro")
    assert set(per_category) == set(CATEGORIES)
    assert set(per_category.values()) <= {10, 11}


def test_four_distinct_options_with_gold_in_place(items):
    rows = {f"mmlupro_{r['category']}_{r['question_id']:05d}": r for r in mmlu_rows()}
    for it in items:
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4
        if it["source"] == "mmlu_pro":
            r = rows[it["item_id"]]
            assert it["options"]["ABCD".index(it["gold"])] == r["options"][r["answer_index"]]


def test_gold_letters_balanced(items):
    assert Counter(it["gold"] for it in items) == {"A": 75, "B": 75, "C": 75, "D": 75}
    for source in ["mmlu_pro", "arc"]:
        per_source = Counter(it["gold"] for it in items if it["source"] == source)
        assert set(per_source.values()) <= {37, 38}


def test_balanced_over_groups_evens_out_remainders():
    seqs = dataset.balanced_over_groups([5, 5, 2], "ABCD", random.Random(0))
    assert [len(s) for s in seqs] == [5, 5, 2]
    assert sorted(Counter(letter for s in seqs for letter in s).values()) == [3, 3, 3, 3]


def test_wrong_target_never_gold_and_balanced(items):
    assert all(it["x"] in "ABCD" and it["x"] != it["gold"] for it in items)
    assert Counter(it["x"] for it in items) == {"A": 75, "B": 75, "C": 75, "D": 75}
    for gold in "ABCD":
        xs = Counter(it["x"] for it in items if it["gold"] == gold)
        assert sorted(xs.values()) == [25, 25, 25]


def test_control_subset_half_per_source(items):
    assert Counter(it["source"] for it in items if it["control"]) == {"mmlu_pro": 50, "arc": 50}


def test_schema_fields(items):
    assert set(items[0]) == {"item_id", "source", "subject", "question", "options", "gold", "x", "control", "orig_id"}
    assert items[0]["question"] == items[0]["question"].strip()


def test_deterministic_and_order_independent(items):
    again = dataset.build_items(list(reversed(mmlu_rows())), arc_rows()[::-1], 150, 150, 100, 42)
    assert dataset.to_jsonl(again) == dataset.to_jsonl(items)


def test_seed_changes_sample(items):
    other = dataset.build_items(mmlu_rows(), arc_rows(), 150, 150, 100, 7)
    assert dataset.to_jsonl(other) != dataset.to_jsonl(items)


def test_too_few_items_raises():
    with pytest.raises(ValueError):
        dataset.build_items(mmlu_rows(per_category=5), arc_rows(), 150, 150, 100, 42)


def test_write_or_check_never_overwrites(tmp_path):
    path = tmp_path / "items.jsonl"
    assert dataset.write_or_check(path, "original\n")
    assert dataset.write_or_check(path, "original\n")
    assert not dataset.write_or_check(path, "changed\n")
    assert path.read_text(encoding="utf-8") == "original\n"
