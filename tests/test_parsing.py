"""Tests for cavein/parsing.py: the answer letter and the letter probabilities."""
import pytest

from cavein import parsing
from mock_ollama import top


@pytest.mark.parametrize("text, letter, status", [
    ("B", "B", "ok"), (" B\n", "B", "ok"), ("B)", "B", "ok"), ("(B)", "B", "ok"), ("**B**", "B", "ok"),
    ("B.", "B", "ok"), ("b", "B", "ok"),
    ("Answer: B", "B", "extracted"), ("The answer is (C).", "C", "extracted"),
    ("The correct answer is **D**", "D", "extracted"), ("answer: a", None, "not_a_letter"),
    ("B) carbon dioxide", "B", "extracted"), ("C. Nitrogen", "C", "extracted"),
    ("I would go with (A) because", "A", "extracted"),
    ("D\n\nThe pedestrian entered the club without permission", "D", "extracted"),
    ("**C**\nBecause plants need it.", "C", "extracted"),
    ("A\nB", None, "ambiguous"),
    ("B) his intelligence  \nD) his height  \n\nTraits like", None, "ambiguous"),
    ("B  \nD  \n\nTraits like intelligence", None, "ambiguous"),
    ("D) 1.0/8\n\nFirst cousins share, on average,", "D", "extracted"),
    ("C\n\nA monopsony, being the sole buyer", "C", "extracted"),
    ("<think>maybe A or B</think>\nC", "C", "ok"),
    ("<think>still thinking about A", None, "not_a_letter"),
    ("<think>only thinking, no answer</think>", None, "not_a_letter"),
    ("A plant takes in carbon dioxide.", None, "not_a_letter"),
    ("The answer is a bit tricky", None, "not_a_letter"),
    ("Either (B) or (C)", None, "ambiguous"),
    ("", None, "not_a_letter"), (None, None, "not_a_letter"),
    ("   ", None, "not_a_letter"), ("\n", None, "not_a_letter"),
])
def test_text_letter(text, letter, status):
    assert parsing.text_letter(text) == (letter, status)


def test_letter_probs_sums_variants_and_normalises():
    lp = parsing.letter_probs(top({"B": 0.5, " B": 0.1, "B)": 0.1, "(A": 0.2, "hello": 0.1}))
    assert lp["ft_letter"] == "B"
    assert lp["answer_mass"] == pytest.approx(0.9)
    assert lp["p"]["B"] == pytest.approx(0.7 / 0.9)
    assert lp["p"]["A"] == pytest.approx(0.2 / 0.9)


def test_letter_outside_top_k_is_missing_not_zero():
    lp = parsing.letter_probs(top({"A": 0.6, "B": 0.3}))
    assert lp["p"]["C"] is None and lp["p"]["D"] is None
    assert sum(v for v in lp["p"].values() if v) == pytest.approx(1.0)


def test_no_letter_in_top_k():
    lp = parsing.letter_probs(top({"**": 0.8, "The": 0.2}))
    assert lp == {"p": {"A": None, "B": None, "C": None, "D": None}, "ft_letter": None, "answer_mass": 0.0}


def test_no_logprobs_at_all():
    assert parsing.letter_probs(None)["answer_mass"] is None


def test_lowercase_token_is_not_a_letter_variant():
    assert parsing.letter_probs(top({"a": 0.9, "B": 0.1}))["p"]["A"] is None


def test_confidence_is_prob_of_text_letter_not_argmax():
    s = parsing.scored({"text": "B", "top": top({"A": 0.6, "B": 0.4}), "latency_s": 0})
    assert s["a"] == "B" and s["ft_letter"] == "A" and s["c"] == pytest.approx(0.4)
