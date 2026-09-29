"""Read the answer letter from a reply, and the letter probabilities from the log-probs."""
import math
import re

from cavein.config import LETTERS

# The patterns below find the answer letter in the text of a reply.
# <think> ... </think> blocks are removed first.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
# only one letter, maybe with punctuation around it: "B", "B.", "(B)", "**B**"
ONLY_LETTER = re.compile(r"^[\W_]*([A-Da-d])[\W_]*$")
# "Answer: B", "The answer is B", "answer is (B)"
ANSWER_IS = re.compile(r"(?i:answer)\s*(?:(?i:is)|:)?\s*[:\-]?\s*[\(\[\*]*\s*([A-D])\b(?![\w'])")
# the reply starts with a letter and a mark: "B) Paris", "B. Paris"
LEADING = re.compile(r"^[\s\*\(\[]*([A-D])[\)\]\.:](?!\w)")
# a letter in brackets somewhere in the text: "(B)" or "B)"
MARKED = re.compile(r"\(([A-D])\)|(?<![\w(])([A-D])\)")
# a line that starts with a letter (used to see if the reply lists several options)
LINE_START = re.compile(r"^[\s\*\(\[]*([A-D])(?:[\)\]\.:](?!\w)|[\s\*]*$)", re.M)


def letter_of_token(token):
    """'A', ' A', 'A)', '(A' and '**A' all count as A. A small 'a' does not count (it is usually the word 'a')."""
    text = token.strip().strip("()[]{}*.:,;\"'`_ ")
    if len(text) == 1 and text in LETTERS:
        return text
    return None


def letter_probs(top):
    """Letter probabilities at the first generated position.

    A letter that is not in the top-20 tokens gets None (missing), not 0.
    """
    if not top:
        return {"p": {"A": None, "B": None, "C": None, "D": None}, "ft_letter": None, "answer_mass": None}

    # add up the probability of all tokens that mean the same letter
    sums = {}
    for entry in top:
        letter = letter_of_token(entry["token"])
        if letter:
            if letter not in sums:
                sums[letter] = 0.0
            sums[letter] = sums[letter] + math.exp(entry["logprob"])

    # answer_mass = how much probability went to A-D at all
    mass = sum(sums.values())

    # normalise so that A-D add up to 1
    p = {}
    for letter in LETTERS:
        if letter in sums and mass > 0:
            p[letter] = sums[letter] / mass
        else:
            p[letter] = None

    # the letter with the highest probability (the first one if two are equal)
    ft_letter = None
    for letter in sums:
        if ft_letter is None or sums[letter] > sums[ft_letter]:
            ft_letter = letter

    return {"p": p, "ft_letter": ft_letter, "answer_mass": mass}


def text_letter(text):
    """The letter written in the reply -> (letter, status). Status is ok, extracted, ambiguous or not_a_letter."""
    if not text:
        return None, "not_a_letter"
    text = THINK_BLOCK.sub("", text)
    if re.search("<think>", text, re.IGNORECASE):
        return None, "not_a_letter"  # a <think> block that was never closed
    text = text.strip()
    if text == "":
        return None, "not_a_letter"

    # 1. the whole reply is just one letter
    match = ONLY_LETTER.match(text)
    if match:
        return match.group(1).upper(), "ok"

    # 2. several lines that start with different letters: we cannot tell which one is the answer
    line_letters = set(LINE_START.findall(text))
    if len(line_letters) > 1:
        return None, "ambiguous"

    # 3. the first line is just one letter
    first_line = text.splitlines()[0]
    match = ONLY_LETTER.match(first_line)
    if match:
        return match.group(1).upper(), "extracted"

    # 4. "the answer is B"
    match = ANSWER_IS.search(text)
    if match:
        return match.group(1), "extracted"

    # 5. the reply starts with "B)" or "B."
    match = LEADING.match(text)
    if match:
        return match.group(1), "extracted"

    # 6. a letter in brackets somewhere: fine if it is always the same letter
    marked = set()
    for first, second in MARKED.findall(text):
        if first:
            marked.add(first)
        else:
            marked.add(second)
    if len(marked) == 1:
        return marked.pop(), "extracted"
    if len(marked) > 1:
        return None, "ambiguous"

    return None, "not_a_letter"


def scored(response):
    """A model response -> answer letter, its confidence, first-token letter and probabilities."""
    letter, status = text_letter(response["text"])
    probs = letter_probs(response["top"])
    if letter:
        confidence = probs["p"][letter]
    else:
        confidence = None
    return {"a": letter, "c": confidence, "ft_letter": probs["ft_letter"], "p": probs["p"],
            "answer_mass": probs["answer_mass"], "raw": response["text"], "parse_status": status}
