"""Load the result files (results/*.jsonl) and prepare them for the analysis."""
import json

import pandas as pd

# these columns are True / False / missing
FLAG_COLUMNS = ["correct_0", "harmful", "went_to_x", "beneficial", "flipped"]


def run_of(path):
    """The run name from the file name.

    'm__main.jsonl' -> 'main', 'm__reversed.jsonl' -> 'reversed', 'm__main__det1.jsonl' -> 'det1'
    """
    parts = path.stem.split("__")
    if len(parts) > 2:
        return parts[2]
    return parts[1]


def read_jsonl(path):
    # one JSON object per line; empty lines are skipped
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_results(results_dir):
    """All result files except the pilots; the last error-free line per model, run, item and condition."""
    frames = []
    for path in sorted(results_dir.glob("*.jsonl")):
        # pilot files are only test runs, they are not used in the analysis
        if "__pilot" in path.name:
            continue
        rows = read_jsonl(path)
        if len(rows) > 0:
            frame = pd.DataFrame(rows)
            frame["run"] = run_of(path)
            frames.append(frame)

    if len(frames) == 0:
        raise SystemExit(f"no result files in {results_dir}")

    df = pd.concat(frames, ignore_index=True)
    # drop the lines that had an error
    df = df[df["error"].isna()]
    # if a line is there twice, keep the last one
    df = df.drop_duplicates(["model", "run", "item_id", "condition"], keep="last")
    for column in FLAG_COLUMNS:
        df[column] = df[column].astype("boolean")
    return df.reset_index(drop=True)


def main_data(df):
    """Only the main runs with the main wording."""
    is_main_run = df["run"] == "main"
    is_main_variant = df["variant"] == "main"
    return df[is_main_run & is_main_variant]


def turn1(df):
    # turn 1 is the same for the four follow-ups, so keep one row per model, run and item
    return df.drop_duplicates(["model", "run", "item_id"])


def paired(df, condition, prec_a, prec_b):
    """Items correct in turn 1 at both precisions of a family, with a valid turn-2 letter at both."""
    # step 1: one follow-up, the two precisions we compare
    d = df[(df["condition"] == condition) & df["precision"].isin([prec_a, prec_b])]
    # step 2: correct in turn 1 and a valid turn-2 answer
    d = d[(d["correct_0"] == True) & d["harmful"].notna()]
    # step 3: one row per family and item, one column per precision
    w = d.pivot_table(index=["family", "item_id"], columns="precision", values=["harmful", "c0"], aggfunc="first")
    # step 4: keep only items that are there at both precisions
    w = w.dropna(subset=[("harmful", prec_a), ("harmful", prec_b)])

    out = pd.DataFrame({
        "family": w.index.get_level_values(0),
        "item_id": w.index.get_level_values(1),
        "harmful_a": w[("harmful", prec_a)].astype(bool).to_numpy(),
        "harmful_b": w[("harmful", prec_b)].astype(bool).to_numpy(),
        "c0_a": w[("c0", prec_a)].to_numpy(dtype=float),
        "c0_b": w[("c0", prec_b)].to_numpy(dtype=float),
    })
    return out.reset_index(drop=True)
