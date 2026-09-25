"""Load the frozen items and follow-ups for a run, and pick which items to ask."""
import json
import random

from cavein.config import SEED


def load_followups(path, variant):
    data = json.loads(path.read_text(encoding="utf-8"))
    if variant in ["main", "reversed"]:
        return data["variants"]["main"]
    return data["variants"][variant]


def reverse_item(item):
    """Options in reverse order (A<->D, B<->C); the gold and wrong letters move with them."""
    flip = {"A": "D", "B": "C", "C": "B", "D": "A"}
    new_item = dict(item)
    new_item["options"] = list(reversed(item["options"]))
    new_item["gold"] = flip[item["gold"]]
    new_item["x"] = flip[item["x"]]
    return new_item


def load_items(path, variant):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    if variant == "reversed":
        items = [reverse_item(item) for item in items]
    return items


def select_items(items, n_items, control_only):
    """All items, only the control items, and/or a fixed random sample of n items."""
    if control_only:
        pool = [item for item in items if item["control"]]
    else:
        pool = list(items)
    if n_items is not None and n_items < len(pool):
        pool = sorted(pool, key=lambda item: item["item_id"])
        pool = random.Random(f"{SEED}:select").sample(pool, n_items)
        pool.sort(key=lambda item: item["item_id"])
    return pool
