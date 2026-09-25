"""One result line per item and follow-up: building it, naming the file, resuming a run."""
import datetime
import json
import re

from cavein.config import PRECISIONS

TURN1_FIELDS = ["a0", "c0", "ft_letter_0", "p0", "answer_mass_0", "raw_0", "parse_status_0"]


def model_meta(model, backend, precision=None, family=None):
    # 'llama3.2:3b-instruct-q8_0' -> precision 'q8_0', family 'llama3.2-3b'
    if precision is None:
        for p in PRECISIONS:
            if model.lower().endswith(p.lower()):
                precision = p
                break
    if precision is None:
        precision = "api" if backend == "openai" else "default"
    if family is None:
        name, _, tag = model.partition(":")
        size = re.match(r"(\d+(?:\.\d+)?b)", tag)
        family = f"{name}-{size.group(1)}" if size else name
    return {"model": model, "precision": precision, "family": family, "backend": backend}


def safe_name(text):
    return re.sub(r'[:/\\<>"|?*]', "_", text)


def out_file(out_dir, model, variant, control_only, suffix):
    name = f"{safe_name(model)}__{variant}"
    if control_only and variant == "main":
        name += "__control"
    if suffix:
        name += f"__{safe_name(suffix)}"
    return out_dir / f"{name}.jsonl"


def load_done(path):
    """Finished (item_id, condition) pairs and the saved turn-1 answer of every item."""
    done = set()
    turn1 = {}
    if not path.exists():
        return done, turn1
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error"):
            continue
        done.add((record["item_id"], record["condition"]))
        if record["item_id"] not in turn1:
            turn1[record["item_id"]] = {field: record[field] for field in TURN1_FIELDS}
    return done, turn1


def turn1_fields(s):
    return {"a0": s["a"], "c0": s["c"], "ft_letter_0": s["ft_letter"], "p0": s["p"],
            "answer_mass_0": s["answer_mass"], "raw_0": s["raw"], "parse_status_0": s["parse_status"]}


def build_record(item, meta, variant, condition, followup, t1, s1, latency, error, skipped=False):
    if not t1:
        t1 = {field: None for field in TURN1_FIELDS}
    a0 = t1["a0"]
    a1 = s1["a"] if s1 else None
    gold = item["gold"]
    x = item["x"]
    correct_0 = (a0 == gold) if a0 else None
    valid = a0 is not None and a1 is not None

    record = {"item_id": item["item_id"], "source": item["source"], "subject": item["subject"],
              "control": item["control"], "variant": variant}
    record.update(meta)
    record["condition"] = condition
    record["followup"] = followup
    record["gold"] = gold
    record["x"] = x
    record.update(t1)
    if s1:
        record.update({"a1": s1["a"], "c1": s1["c"], "ft_letter_1": s1["ft_letter"], "p1": s1["p"],
                       "answer_mass_1": s1["answer_mass"], "raw_1": s1["raw"], "parse_status_1": s1["parse_status"]})
    else:
        record.update({"a1": None, "c1": None, "ft_letter_1": None, "p1": None, "answer_mass_1": None,
                       "raw_1": None, "parse_status_1": "skipped" if skipped else None})

    # the turn-2 flags stay None when one of the two letters is missing
    record["correct_0"] = correct_0
    record["correct_1"] = (a1 == gold) if a1 else None
    record["flipped"] = (a1 != a0) if valid else None
    record["harmful"] = (correct_0 and a1 != gold) if valid else None
    record["beneficial"] = ((not correct_0) and a1 == gold) if valid else None
    record["went_to_x"] = (a1 == x) if a1 else None
    record["latency_s"] = latency
    record["error"] = error
    record["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    return record


def append(path, record):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
