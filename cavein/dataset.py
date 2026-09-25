"""Build the fixed 300-item sample from MMLU-Pro and ARC-Challenge."""
import json
import random
import statistics
from collections import Counter

from cavein.config import LETTERS


def rng_for(seed, step):
    # every step gets its own random generator, so changing one step does not change the others
    return random.Random(f"{seed}:{step}")


def clean_options(options):
    seen = set()
    result = []
    for option in options:
        text = str(option).strip()
        if text == "" or text.upper() == "N/A" or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def mmlu_pro_candidate(row):
    options = [str(o).strip() for o in row["options"]]
    index = row["answer_index"]
    if index < 0 or index >= len(options):
        return None
    gold = options[index]
    distractors = [o for o in clean_options(options) if o != gold]
    if gold == "" or gold.upper() == "N/A" or len(distractors) < 3:
        return None
    category = str(row["category"]).strip().replace(" ", "_")
    return {
        "item_id": f"mmlupro_{category}_{int(row['question_id']):05d}",
        "source": "mmlu_pro",
        "subject": category,
        "question": str(row["question"]).strip(),
        "gold_text": gold,
        "distractors": distractors,
        "orig_id": str(row["question_id"]),
    }


def arc_candidate(row):
    texts = [str(t).strip() for t in row["choices"]["text"]]
    labels = [str(l).strip() for l in row["choices"]["label"]]
    key = str(row["answerKey"]).strip()
    if len(texts) != 4 or len(set(texts)) != 4 or key not in labels or "" in texts:
        return None
    gold = texts[labels.index(key)]
    return {
        "item_id": f"arc_{row['id']}",
        "source": "arc",
        "subject": "arc_challenge",
        "question": str(row["question"]).strip(),
        "gold_text": gold,
        "distractors": [t for t in texts if t != gold],
        "orig_id": str(row["id"]),
    }


def sample_stratified(candidates, n, rng):
    """Pick n items spread as evenly as possible over the subjects."""
    by_subject = {}
    for c in sorted(candidates, key=lambda c: c["item_id"]):
        if c["subject"] not in by_subject:
            by_subject[c["subject"]] = []
        by_subject[c["subject"]].append(c)
    subjects = sorted(by_subject)
    base, extra = divmod(n, len(subjects))
    plus_one = set(rng.sample(subjects, extra))
    picked = []
    for subject in subjects:
        k = base + 1 if subject in plus_one else base
        if k > len(by_subject[subject]):
            raise ValueError(f"subject {subject!r} has only {len(by_subject[subject])} usable items, need {k}")
        picked += rng.sample(by_subject[subject], k)
    return picked


def sample_simple(candidates, n, rng):
    pool = sorted(candidates, key=lambda c: c["item_id"])
    if n > len(pool):
        raise ValueError(f"only {len(pool)} usable items, need {n}")
    return rng.sample(pool, n)


def balanced_over_groups(sizes, letters, rng):
    """Equal shares of the letters in every group; leftovers go to the letters used least so far."""
    letters = list(letters)
    totals = {letter: 0 for letter in letters}
    result = []
    for n in sizes:
        base, rem = divmod(n, len(letters))
        least_used = sorted(letters, key=lambda letter: (totals[letter], rng.random()))
        seq = letters * base + least_used[:rem]
        rng.shuffle(seq)
        for letter in seq:
            totals[letter] += 1
        result.append(seq)
    return result


def build_items(mmlu_rows, arc_rows, n_mmlupro, n_arc, n_control, seed):
    mmlu_candidates = []
    for row in mmlu_rows:
        candidate = mmlu_pro_candidate(row)
        if candidate:
            mmlu_candidates.append(candidate)
    arc_candidates = []
    for row in arc_rows:
        candidate = arc_candidate(row)
        if candidate:
            arc_candidates.append(candidate)

    chosen = sample_stratified(mmlu_candidates, n_mmlupro, rng_for(seed, "sample_mmlupro"))
    chosen += sample_simple(arc_candidates, n_arc, rng_for(seed, "sample_arc"))
    chosen.sort(key=lambda c: c["item_id"])

    # keep the gold answer plus 3 random wrong options
    rng = rng_for(seed, "distractors")
    for c in chosen:
        c["distractors"] = rng.sample(c["distractors"], 3)

    # the gold letter is balanced over A-D inside each source
    sources = ["mmlu_pro", "arc"]
    groups = [[c for c in chosen if c["source"] == source] for source in sources]
    gold_seqs = balanced_over_groups([len(g) for g in groups], LETTERS, rng_for(seed, "gold_position"))
    gold_of = {}
    for group, seq in zip(groups, gold_seqs):
        for c, letter in zip(group, seq):
            gold_of[c["item_id"]] = letter

    items = []
    for c in chosen:
        gold = gold_of[c["item_id"]]
        distractors = list(c["distractors"])
        options = []
        for letter in LETTERS:
            if letter == gold:
                options.append(c["gold_text"])
            else:
                options.append(distractors.pop(0))
        items.append({
            "item_id": c["item_id"],
            "source": c["source"],
            "subject": c["subject"],
            "question": c["question"],
            "options": options,
            "gold": gold,
            "x": None,
            "control": False,
            "orig_id": c["orig_id"],
        })

    # the wrong letter X is never the gold letter and is balanced over the other three
    rng = rng_for(seed, "wrong_target")
    for gold in LETTERS:
        others = [letter for letter in LETTERS if letter != gold]
        groups = [[it for it in items if it["gold"] == gold and it["source"] == source] for source in sources]
        x_seqs = balanced_over_groups([len(g) for g in groups], others, rng)
        for group, seq in zip(groups, x_seqs):
            for item, x in zip(group, seq):
                item["x"] = x

    # control subset: half from each source
    rng = rng_for(seed, "control")
    n_arc_control = n_control // 2
    n_mmlu_control = n_control - n_arc_control
    for source, k in [("mmlu_pro", n_mmlu_control), ("arc", n_arc_control)]:
        pool = [it for it in items if it["source"] == source]
        for item in rng.sample(pool, min(k, len(pool))):
            item["control"] = True
    return items


def load_sources():
    from datasets import load_dataset  # slow import, so it is only done when the data is needed
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"loaded MMLU-Pro test: {len(mmlu)} rows; ARC-Challenge test: {len(arc)} rows")
    return list(mmlu), list(arc)


def print_summary(items):
    print("items:", len(items))
    for source in ["mmlu_pro", "arc"]:
        sub = [it for it in items if it["source"] == source]
        print(f"\n[{source}] n={len(sub)}  control={sum(it['control'] for it in sub)}")
        if source == "mmlu_pro":
            for subject, count in sorted(Counter(it["subject"] for it in sub).items()):
                print(f"  {subject:<20} {count}")
        print("  gold letters:", dict(sorted(Counter(it["gold"] for it in sub).items())))
        print("  X letters:   ", dict(sorted(Counter(it["x"] for it in sub).items())))
        gold_lengths = []
        other_lengths = []
        for it in sub:
            for letter, option in zip(LETTERS, it["options"]):
                if letter == it["gold"]:
                    gold_lengths.append(len(option))
                else:
                    other_lengths.append(len(option))
        if sub:
            print(f"  mean option length (chars): gold {statistics.mean(gold_lengths):.1f}, "
                  f"distractors {statistics.mean(other_lengths):.1f}")
    print("\nall gold letters:", dict(sorted(Counter(it["gold"] for it in items).items())))
    print("all X letters:   ", dict(sorted(Counter(it["x"] for it in items).items())))
    print("control items:   ", sum(it["control"] for it in items))


def to_jsonl(items):
    return "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)


def to_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_or_check(path, content):
    """Write a new file. If the file exists, only compare it and return True when identical."""
    if path.exists():
        same = path.read_text(encoding="utf-8") == content
        if same:
            print(f"{path}: exists, not overwritten; rebuilt content is IDENTICAL")
        else:
            print(f"{path}: exists, not overwritten; rebuilt content is DIFFERENT (check seed, arguments, dataset version)")
        return same
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"{path}: written")
    return True
