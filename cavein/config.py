"""Paths and fixed settings used by the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = ROOT / "analysis"
FIGURES_DIR = ROOT / "figures"
ITEMS_FILE = DATA_DIR / "items.jsonl"
FOLLOWUPS_FILE = DATA_DIR / "followups.json"
ENV_FILE = ROOT / "env_info.txt"

SEED = 42
LETTERS = "ABCD"
CONDITIONS = ["reask", "speaker_free", "user", "expert"]
VARIANTS = ["main", "reversed", "para1", "para2"]
PRECISIONS = ["q4_K_M", "q8_0", "fp16"]
FAMILIES = ["llama3.2-3b", "qwen2.5-3b", "phi4-mini-3.8b", "qwen2.5-7b", "llama3.1-8b", "phi4-14b"]

# decoding: the same for every model, precision and follow-up
NUM_PREDICT = 16
TOP_LOGPROBS = 20

# analysis settings, fixed before the runs
N_BOOT = 2000
TOST_MARGIN = 0.03
ECE_BINS = 15
GEE_MAXITER = 200
CONF_BIN_EDGES = [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0000001]
