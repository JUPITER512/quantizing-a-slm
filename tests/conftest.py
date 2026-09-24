"""Make the modules in scripts/ importable from tests (e.g. `import run_pushback`)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
