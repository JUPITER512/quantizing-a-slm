"""The command-line scripts start and find the cavein package; verify_repro finds broken result files."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.mark.parametrize("name", ["build_items.py", "run_pushback.py", "analyze.py", "make_figures.py",
                                  "verify_repro.py", "verify_numbers.py"])
def test_script_help(name):
    result = subprocess.run([sys.executable, str(SCRIPTS / name), "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_verify_repro_finds_problems(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import verify_repro

    good = '{"model": "m", "item_id": "i1", "condition": "reask", "error": null}'
    lines = [good, good, '{"model": "m", "item_id": "i2", "condition": "reask", "error": "timeout"}', "not json"]
    (tmp_path / "m__main.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "m__main__pilot20.jsonl").write_text("not json\n", encoding="utf-8")
    problems, n_files = verify_repro.check_results(tmp_path)
    text = "\n".join(problems)
    assert n_files == 1
    assert "duplicate" in text
    assert "error record" in text
    assert "not valid JSON" in text
    assert "4 lines, expected 1200" in text
    assert "pilot" not in text
