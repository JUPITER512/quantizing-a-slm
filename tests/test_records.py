"""Tests for cavein/records.py and the key redaction in cavein/backend.py."""
from cavein import backend, records

ITEM = {"item_id": "arc_T0", "source": "arc", "subject": "arc_challenge", "question": "Q0?",
        "options": ["o1", "o2", "o3", "o4"], "gold": "A", "x": "C", "control": True, "orig_id": "T0"}
META = records.model_meta("mock:3b-instruct-q4_K_M", "ollama")


def test_model_meta_and_windows_safe_file_names(tmp_path):
    m = records.model_meta("llama3.2:3b-instruct-q8_0", "ollama")
    assert (m["precision"], m["family"]) == ("q8_0", "llama3.2-3b")
    assert records.model_meta("qwen2.5:7b-instruct-fp16", "ollama")["family"] == "qwen2.5-7b"
    assert records.model_meta("gpt-4o-mini", "openai")["precision"] == "api"
    m = records.model_meta("phi4-mini:3.8b-fp16", "ollama")
    assert (m["precision"], m["family"]) == ("fp16", "phi4-mini-3.8b")
    m = records.model_meta("phi4:14b-q4_K_M", "ollama")
    assert (m["precision"], m["family"]) == ("q4_K_M", "phi4-14b")
    name = records.out_file(tmp_path, "hf.co/x/y:Q8_0", "reversed", True, None).name
    assert name == "hf.co_x_y_Q8_0__reversed.jsonl"
    assert records.out_file(tmp_path, "a:b", "main", False, "pilot20").name == "a_b__main__pilot20.jsonl"


def turn2(letter):
    return {"a": letter, "c": 0.5, "ft_letter": letter, "p": {}, "answer_mass": 1.0, "raw": letter,
            "parse_status": "ok"}


def test_flags():
    t1 = {field: None for field in records.TURN1_FIELDS}
    t1["a0"] = "A"
    t1["raw_0"] = "A"

    r = records.build_record(ITEM, META, "main", "user", "f", t1, turn2("C"), 0.1, None)
    assert (r["correct_0"], r["harmful"], r["went_to_x"], r["flipped"], r["beneficial"]) == (True, True, True, True, False)

    r = records.build_record(ITEM, META, "main", "user", "f", t1, turn2("A"), 0.1, None)
    assert (r["harmful"], r["flipped"]) == (False, False)

    wrong_t1 = dict(t1, a0="B")
    r = records.build_record(ITEM, META, "main", "user", "f", wrong_t1, turn2("A"), 0.1, None)
    assert (r["correct_0"], r["beneficial"], r["harmful"]) == (False, True, False)

    r = records.build_record(ITEM, META, "main", "user", "f", t1, None, None, None)
    assert r["harmful"] is None and r["flipped"] is None


def test_load_done_ignores_error_lines(tmp_path):
    t1 = {field: None for field in records.TURN1_FIELDS}
    t1["a0"] = "A"
    path = tmp_path / "out.jsonl"
    records.append(path, records.build_record(ITEM, META, "main", "user", "f", t1, turn2("C"), 0.1, None))
    records.append(path, records.build_record(ITEM, META, "main", "expert", "f", t1, None, None, "turn2: boom"))
    done, turn1 = records.load_done(path)
    assert done == {("arc_T0", "user")}
    assert turn1["arc_T0"]["a0"] == "A"


def test_redact(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnop")
    out = backend.redact("bad key sk-proj-abcdefghijklmnop and sk-otherkey123456")
    assert "abcdefghijklmnop" not in out and "otherkey123456" not in out
