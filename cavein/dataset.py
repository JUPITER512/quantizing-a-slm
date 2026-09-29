"""Build the fixed 300-item sample from MMLU-Pro and ARC-Challenge.

Everything random uses a seed, so the same sample comes out every time.
"""
import json
import random
import statistics
from collections import Counter

from cavein.config import LETTERS

SOURCES = ["mmlu_pro", "arc"]


def rng_for(seed, step):
    # every step gets its own random generator, so changing one step does not change the others
    return random.Random(f"{seed}:{step}")


def get_item_id(candidate):
    return candidate["item_id"]


def clean_options(options):
    # remove empty options, 'N/A' and duplicates (keep the first one)
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
    """One MMLU-Pro row -> a candidate item, or None if it cannot be used (needs a gold answer and 3 wrong options)."""
    options = []
    for o in row["options"]:
        options.append(str(o).strip())
    index = row["answer_index"]
    if index < 0 or index >= len(options):
        return None
    gold = options[index]

    distractors = []
    for o in clean_options(options):
        if o != gold:
            distractors.append(o)
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
    """One ARC-Challenge row -> a candidate item, or None if it does not have exactly 4 different options."""
    texts = []
    for t in row["choices"]["text"]:
        texts.append(str(t).strip())
    labels = []
    for l in row["choices"]["label"]:
        labels.append(str(l).strip())
    key = str(row["answerKey"]).strip()
    if len(texts) != 4 or len(set(texts)) != 4 or key not in labels or "" in texts:
        return None
    gold = texts[labels.index(key)]

    distractors = []
    for t in texts:
        if t != gold:
            distractors.append(t)
    return {
        "item_id": f"arc_{row['id']}",
        "source": "arc",
        "subject": "arc_challenge",
        "question": str(row["question"]).strip(),
        "gold_text": gold,
        "distractors": distractors,
        "orig_id": str(row["id"]),
    }


def sample_stratified(candidates, n, rng):
    """Pick n items, spread as evenly as possible over the subjects."""
    by_subject = {}
    for c in sorted(candidates, key=get_item_id):
        if c["subject"] not in by_subject:
            by_subject[c["subject"]] = []
        by_subject[c["subject"]].append(c)
    subjects = sorted(by_subject)

    # every subject gets `base` items; `extra` random subjects get one more
    base = n // len(subjects)
    extra = n % len(subjects)
    plus_one = set(rng.sample(subjects, extra))

    picked = []
    for subject in subjects:
        if subject in plus_one:
            k = base + 1
        else:
            k = base
        if k > len(by_subject[subject]):
            raise ValueError(f"subject {subject!r} has only {len(by_subject[subject])} usable items, need {k}")
        picked = picked + rng.sample(by_subject[subject], k)
    return picked


def sample_simple(candidates, n, rng):
    pool = sorted(candidates, key=get_item_id)
    if n > len(pool):
        raise ValueError(f"only {len(pool)} usable items, need {n}")
    return rng.sample(pool, n)


def balanced_over_groups(sizes, letters, rng):
    """Equal shares of the letters in every group; the leftovers go to the letters used least so far."""
    letters = list(letters)
    totals = {}
    for letter in letters:
        totals[letter] = 0

    result = []
    for n in sizes:
        base = n // len(letters)
        rem = n % len(letters)

        # sort the letters by how often they were used; a random number breaks ties
        random_part = {}
        for letter in letters:
            random_part[letter] = rng.random()

        def sort_key(letter):
            return (totals[letter], random_part[letter])

        least_used = sorted(letters, key=sort_key)

        seq = letters * base + least_used[:rem]
        rng.shuffle(seq)
        for letter in seq:
            totals[letter] += 1
        result.append(seq)
    return result


def items_of_source(items, source):
    result = []
    for it in items:
        if it["source"] == source:
            result.append(it)
    return result


def build_items(mmlu_rows, arc_rows, n_mmlupro, n_arc, n_control, seed):
    # step 1: which rows can be used at all
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

    # step 2: the random sample (MMLU-Pro spread over the subjects)
    chosen = sample_stratified(mmlu_candidates, n_mmlupro, rng_for(seed, "sample_mmlupro"))
    chosen = chosen + sample_simple(arc_candidates, n_arc, rng_for(seed, "sample_arc"))
    chosen.sort(key=get_item_id)

    # step 3: keep the gold answer plus 3 random wrong options
    rng = rng_for(seed, "distractors")
    for c in chosen:
        c["distractors"] = rng.sample(c["distractors"], 3)

    # step 4: the gold letter is balanced over A-D inside each source
    groups = []
    sizes = []
    for source in SOURCES:
        group = items_of_source(chosen, source)
        groups.append(group)
        sizes.append(len(group))
    gold_seqs = balanced_over_groups(sizes, LETTERS, rng_for(seed, "gold_position"))
    gold_of = {}
    for g in range(len(groups)):
        for i in range(len(groups[g])):
            gold_of[groups[g][i]["item_id"]] = gold_seqs[g][i]

    # step 5: put the gold answer at its letter and the wrong options at the other letters
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

    # step 6: the wrong letter X is never the gold letter and is balanced over the other three
    rng = rng_for(seed, "wrong_target")
    for gold in LETTERS:
        others = []
        for letter in LETTERS:
            if letter != gold:
                others.append(letter)
        groups = []
        sizes = []
        for source in SOURCES:
            group = []
            for it in items:
                if it["gold"] == gold and it["source"] == source:
                    group.append(it)
            groups.append(group)
            sizes.append(len(group))
        x_seqs = balanced_over_groups(sizes, others, rng)
        for g in range(len(groups)):
            for i in range(len(groups[g])):
                groups[g][i]["x"] = x_seqs[g][i]

    # step 7: the control subset, half from each source
    rng = rng_for(seed, "control")
    n_arc_control = n_control // 2
    n_mmlu_control = n_control - n_arc_control
    for source, k in [("mmlu_pro", n_mmlu_control), ("arc", n_arc_control)]:
        pool = items_of_source(items, source)
        for item in rng.sample(pool, min(k, len(pool))):
            item["control"] = True
    return items


def load_sources():
    from datasets import load_dataset  # slow import, so it is only done when the data is needed
    mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    print(f"loaded MMLU-Pro test: {len(mmlu)} rows; ARC-Challenge test: {len(arc)} rows")
    return list(mmlu), list(arc)


def count_of(items, field):
    # e.g. {'A': 75, 'B': 75, ...}, sorted by key
    return dict(sorted(Counter(it[field] for it in items).items()))


def print_summary(items):
    print("items:", len(items))
    for source in SOURCES:
        sub = items_of_source(items, source)
        n_control = 0
        for it in sub:
            if it["control"]:
                n_control += 1
        print(f"\n[{source}] n={len(sub)}  control={n_control}")
        if source == "mmlu_pro":
            for subject, count in count_of(sub, "subject").items():
                print(f"  {subject:<20} {count}")
        print("  gold letters:", count_of(sub, "gold"))
        print("  X letters:   ", count_of(sub, "x"))

        # are the gold answers longer than the wrong ones? (a model could guess from the length)
        gold_lengths = []
        other_lengths = []
        for it in sub:
            for i in range(len(LETTERS)):
                if i >= len(it["options"]):
                    break
                if LETTERS[i] == it["gold"]:
                    gold_lengths.append(len(it["options"][i]))
                else:
                    other_lengths.append(len(it["options"][i]))
        if sub:
            print(f"  mean option length (chars): gold {statistics.mean(gold_lengths):.1f}, "
                  f"distractors {statistics.mean(other_lengths):.1f}")

    n_control = 0
    for it in items:
        if it["control"]:
            n_control += 1
    print("\nall gold letters:", count_of(items, "gold"))
    print("all X letters:   ", count_of(items, "x"))
    print("control items:   ", n_control)


def to_jsonl(items):
    lines = []
    for item in items:
        lines.append(json.dumps(item, ensure_ascii=False) + "\n")
    return "".join(lines)


def to_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_or_check(path, content):
    """Write a new file. If the file is already there, only compare it (never overwrite) and return True when identical."""
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
