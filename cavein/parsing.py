"""Read the answer letter from a reply and the letter probabilities from the log-probs."""
import math
import re

from cavein.config import LETTERS

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
ONLY_LETTER = re.compile(r"^[\W_]*([A-Da-d])[\W_]*$")
ANSWER_IS = re.compile(r"(?i:answer)\s*(?:(?i:is)|:)?\s*[:\-]?\s*[\(\[\*]*\s*([A-D])\b(?![\w'])")
LEADING = re.compile(r"^[\s\*\(\[]*([A-D])[\)\]\.:](?!\w)")
MARKED = re.compile(r"\(([A-D])\)|(?<![\w(])([A-D])\)")
LINE_START = re.compile(r"^[\s\*\(\[]*([A-D])(?:[\)\]\.:](?!\w)|[\s\*]*$)", re.M)


def letter_of_token(token):
    # 'A', ' A', 'A)', '(A' and '**A' all count as A; lowercase 'a' does not (it is usually the article)
    text = token.strip().strip("()[]{}*.:,;\"'`_ ")
    if len(text) == 1 and text in LETTERS:
        return text
    return None


def letter_probs(top):
    """Letter probabilities at the first generated position. A letter missing from the top-20 is None, not 0."""
    if not top:
        return {"p": {"A": None, "B": None, "C": None, "D": None}, "ft_letter": None, "answer_mass": None}
    sums = {}
    for entry in top:
        letter = letter_of_token(entry["token"])
        if letter:
            sums[letter] = sums.get(letter, 0.0) + math.exp(entry["logprob"])
    mass = sum(sums.values())
    p = {}
    for letter in LETTERS:
        if letter in sums and mass > 0:
            p[letter] = sums[letter] / mass
        else:
            p[letter] = None
    ft_letter = max(sums, key=sums.get) if sums else None
    return {"p": p, "ft_letter": ft_letter, "answer_mass": mass}


def text_letter(text):
    """Letter written in the reply -> (letter, status): ok, extracted, ambiguous or not_a_letter."""
    if not text:
        return None, "not_a_letter"
    text = THINK_BLOCK.sub("", text)
    if re.search("<think>", text, re.IGNORECASE):
        return None, "not_a_letter"
    text = text.strip()
    if text == "":
        return None, "not_a_letter"

    match = ONLY_LETTER.match(text)
    if match:
        return match.group(1).upper(), "ok"
    if len(set(LINE_START.findall(text))) > 1:
        return None, "ambiguous"
    match = ONLY_LETTER.match(text.splitlines()[0])
    if match:
        return match.group(1).upper(), "extracted"
    match = ANSWER_IS.search(text)
    if match:
        return match.group(1), "extracted"
    match = LEADING.match(text)
    if match:
        return match.group(1), "extracted"

    marked = set()
    for a, b in MARKED.findall(text):
        marked.add(a or b)
    if len(marked) == 1:
        return marked.pop(), "extracted"
    if len(marked) > 1:
        return None, "ambiguous"
    return None, "not_a_letter"


def scored(response):
    """A model response -> answer letter, its confidence, first-token letter and probabilities."""
    letter, status = text_letter(response["text"])
    probs = letter_probs(response["top"])
    confidence = probs["p"][letter] if letter else None
    return {"a": letter, "c": confidence, "ft_letter": probs["ft_letter"], "p": probs["p"],
            "answer_mass": probs["answer_mass"], "raw": response["text"], "parse_status": status}
