"""The four command-line scripts start and find the cavein package."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.mark.parametrize("name", ["build_items.py", "run_pushback.py", "analyze.py", "make_figures.py"])
def test_script_help(name):
    result = subprocess.run([sys.executable, str(SCRIPTS / name), "--help"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
