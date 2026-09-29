"""Tests for cavein/items.py."""
from cavein import items as items_module

ITEMS = []
for i, (gold, x) in enumerate([("A", "C"), ("B", "D"), ("A", "B"), ("D", "A")]):
    ITEMS.append({"item_id": f"arc_T{i}", "source": "arc", "subject": "arc_challenge", "question": f"Q{i}?",
                  "options": ["o1", "o2", "o3", "o4"], "gold": gold, "x": x, "control": i < 2, "orig_id": f"T{i}"})


def test_reverse_item_remaps_letters():
    item = dict(ITEMS[0])
    item["options"] = ["w", "x", "y", "z"]
    reversed_item = items_module.reverse_item(item)
    assert reversed_item["options"] == ["z", "y", "x", "w"]
    assert reversed_item["gold"] == "D" and reversed_item["x"] == "B"
    assert item["gold"] == "A"  # the original item is not changed


def test_select_items_is_seeded_and_control_only():
    assert items_module.select_items(ITEMS, 2, False) == items_module.select_items(ITEMS, 2, False)
    assert [it["item_id"] for it in items_module.select_items(ITEMS, None, True)] == ["arc_T0", "arc_T1"]
    assert items_module.select_items(ITEMS, None, False) == ITEMS
