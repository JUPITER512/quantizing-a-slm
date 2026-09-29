"""One result line per item and follow-up: building it, naming the file, resuming a run."""
import datetime
import json
import re

from cavein.config import PRECISIONS

# the turn-1 fields of a record (turn 1 is the same for all four follow-ups)
TURN1_FIELDS = ["a0", "c0", "ft_letter_0", "p0", "answer_mass_0", "raw_0", "parse_status_0"]


def model_meta(model, backend, precision=None, family=None):
    """Model name -> precision and family. Example: 'llama3.2:3b-instruct-q8_0' -> 'q8_0', 'llama3.2-3b'."""
    # the precision is the end of the tag
    if precision is None:
        for p in PRECISIONS:
            if model.lower().endswith(p.lower()):
                precision = p
                break
    if precision is None:
        if backend == "openai":
            precision = "api"
        else:
            precision = "default"

    # the family is the name plus the size, e.g. 'llama3.2' + '3b'
    if family is None:
        name, _, tag = model.partition(":")
        size = re.match(r"(\d+(?:\.\d+)?b)", tag)
        if size:
            family = f"{name}-{size.group(1)}"
        else:
            family = name
    return {"model": model, "precision": precision, "family": family, "backend": backend}


def safe_name(text):
    # characters that Windows does not allow in file names become '_'
    return re.sub(r'[:/\\<>"|?*]', "_", text)


def out_file(out_dir, model, variant, control_only, suffix):
    """The result file name, e.g. results/llama3.2_3b-instruct-q4_K_M__main.jsonl"""
    name = f"{safe_name(model)}__{variant}"
    if control_only and variant == "main":
        name = name + "__control"
    if suffix:
        name = name + f"__{safe_name(suffix)}"
    return out_dir / f"{name}.jsonl"


def load_done(path):
    """Read an existing result file: the finished (item_id, condition) pairs and the saved turn-1 answer of every item."""
    done = set()
    turn1 = {}
    if not path.exists():
        return done, turn1

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error"):
            continue  # lines with an error are asked again
        done.add((record["item_id"], record["condition"]))
        if record["item_id"] not in turn1:
            saved = {}
            for field in TURN1_FIELDS:
                saved[field] = record[field]
            turn1[record["item_id"]] = saved
    return done, turn1


def turn1_fields(s):
    # a scored turn-1 reply -> the turn-1 fields of a record
    return {"a0": s["a"], "c0": s["c"], "ft_letter_0": s["ft_letter"], "p0": s["p"],
            "answer_mass_0": s["answer_mass"], "raw_0": s["raw"], "parse_status_0": s["parse_status"]}


def build_record(item, meta, variant, condition, followup, t1, s1, latency, error, skipped=False):
    """One line of the result file. The order of the fields is the order in the file."""
    if not t1:
        t1 = {}
        for field in TURN1_FIELDS:
            t1[field] = None

    a0 = t1["a0"]
    if s1:
        a1 = s1["a"]
    else:
        a1 = None
    gold = item["gold"]
    x = item["x"]
    both_letters = a0 is not None and a1 is not None

    record = {}
    # the item
    record["item_id"] = item["item_id"]
    record["source"] = item["source"]
    record["subject"] = item["subject"]
    record["control"] = item["control"]
    record["variant"] = variant
    # the model: model, precision, family, backend
    for key in meta:
        record[key] = meta[key]
    # the follow-up
    record["condition"] = condition
    record["followup"] = followup
    record["gold"] = gold
    record["x"] = x
    # turn 1
    for key in t1:
        record[key] = t1[key]
    # turn 2
    if s1:
        record["a1"] = s1["a"]
        record["c1"] = s1["c"]
        record["ft_letter_1"] = s1["ft_letter"]
        record["p1"] = s1["p"]
        record["answer_mass_1"] = s1["answer_mass"]
        record["raw_1"] = s1["raw"]
        record["parse_status_1"] = s1["parse_status"]
    else:
        record["a1"] = None
        record["c1"] = None
        record["ft_letter_1"] = None
        record["p1"] = None
        record["answer_mass_1"] = None
        record["raw_1"] = None
        if skipped:
            record["parse_status_1"] = "skipped"
        else:
            record["parse_status_1"] = None

    # the flags; a flag stays None when a letter it needs is missing
    if a0:
        correct_0 = a0 == gold
    else:
        correct_0 = None
    record["correct_0"] = correct_0

    if a1:
        record["correct_1"] = a1 == gold
    else:
        record["correct_1"] = None

    if both_letters:
        record["flipped"] = a1 != a0
        record["harmful"] = correct_0 and a1 != gold
        record["beneficial"] = (not correct_0) and a1 == gold
    else:
        record["flipped"] = None
        record["harmful"] = None
        record["beneficial"] = None

    if a1:
        record["went_to_x"] = a1 == x
    else:
        record["went_to_x"] = None

    record["latency_s"] = latency
    record["error"] = error
    record["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    return record


def append(path, record):
    # add one line at the end of the file (results are never overwritten)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
