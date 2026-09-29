"""Independent check of every number: raw model replies (results/*.jsonl) -> tables -> figures -> poster.

Nothing here imports the analysis code in cavein/ (except the official letter parser for unusual replies,
which defines the answer label). All flags, counts, rates, tests and intervals are recomputed from the raw
records with plain Python, and compared with:
  1. analysis/*.csv   every row and every column (at the CSV's 6 significant digits)
  2. figures/*.svg    every plotted point and error-bar end, read back from the SVG coordinates
  3. poster/poster.pptx and poster/appendix.html   every number printed (if the poster folder exists)

    python scripts/verify_numbers.py
"""
import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LETTERS = "ABCD"
CONDITIONS = ["reask", "speaker_free", "user", "expert"]
PRECISIONS = ["q4_K_M", "q8_0", "fp16"]
FAMILIES = ["llama3.2-3b", "qwen2.5-3b", "phi4-mini-3.8b", "qwen2.5-7b", "llama3.1-8b", "phi4-14b"]
SEED, N_BOOT = 42, 2000

problems = []
n_checks = 0


def fail(msg):
    problems.append(msg)


def same(mine, theirs, where):
    """Compare a recomputed value with a CSV cell (6 significant digits), a count, a bool or text."""
    global n_checks
    n_checks += 1
    if theirs is None or theirs == "":
        ok = mine is None or mine == "" or (isinstance(mine, float) and math.isnan(mine))
    elif isinstance(mine, bool) or theirs in ("True", "False"):
        ok = str(bool(mine)) == theirs
    elif isinstance(mine, (int, float, np.integer, np.floating)):
        if isinstance(mine, float) and math.isnan(mine):
            ok = False
        else:
            ok = float(f"{float(mine):.6g}") == float(theirs)
    else:
        ok = str(mine) == theirs
    if not ok:
        fail(f"{where}: recomputed {mine!r} but table has {theirs!r}")


def read_csv(name):
    with open(ROOT / "analysis" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ 1. raw records and re-derived flags

def load():
    items = {}
    for line in (ROOT / "data" / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["item_id"]] = item
    followups = json.loads((ROOT / "data" / "followups.json").read_text(encoding="utf-8"))
    flip = {"A": "D", "B": "C", "C": "B", "D": "A"}
    from cavein.parsing import text_letter  # the official definition of the answer label

    records, seen = [], set()
    simple = re.compile(r"^[\W_]*([A-D])[\W_]*$")
    stats = defaultdict(int)
    for path in sorted((ROOT / "results").glob("*.jsonl")):
        if "__pilot" in path.name:
            continue
        parts = path.stem.split("__")
        run = parts[2] if len(parts) > 2 else parts[1]
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            where = f"{path.name}:{number}"
            if r.get("error"):
                fail(f"{where}: error record")
                continue
            key = (r["model"], run, r["item_id"], r["condition"])
            if key in seen:
                fail(f"{where}: duplicate {key}")
            seen.add(key)
            item = items[r["item_id"]]
            gold, x = (flip[item["gold"]], flip[item["x"]]) if r["variant"] == "reversed" else (item["gold"], item["x"])
            if (r["gold"], r["x"]) != (gold, x):
                fail(f"{where}: gold/x {r['gold']}/{r['x']} but items.jsonl gives {gold}/{x}")
            if (r["source"], r["control"]) != (item["source"], item["control"]):
                fail(f"{where}: source/control differ from items.jsonl")
            wording = followups["variants"]["main" if r["variant"] in ("main", "reversed") else r["variant"]]
            if r["followup"] != wording[r["condition"]].replace("{x}", x):
                fail(f"{where}: follow-up text differs from data/followups.json")
            # answer letters: simple replies checked by an independent rule, all by the official parser
            for raw_key, a_key, status_key in (("raw_0", "a0", "parse_status_0"), ("raw_1", "a1", "parse_status_1")):
                raw = r[raw_key] or ""
                m = simple.match(raw.strip())
                if m:
                    stats["simple"] += 1
                    if m.group(1) != r[a_key]:
                        fail(f"{where}: {a_key}={r[a_key]!r} but the reply is just {raw!r}")
                else:
                    stats["complex"] += 1
                if r[status_key] == "skipped":  # turn 2 is not asked when turn 1 has no letter
                    stats["skipped"] += 1
                    if r["a0"] is not None or r[raw_key] is not None or r[a_key] is not None:
                        fail(f"{where}: turn 2 marked skipped although turn 1 has a letter")
                    continue
                letter, status = text_letter(raw)
                if (letter, status) != (r[a_key], r[status_key]):
                    fail(f"{where}: stored {a_key}={r[a_key]}/{r[status_key]}, re-parse gives {letter}/{status}")
            # confidence = probability of the text letter; probabilities sum to 1; first-token letter = argmax
            for p_key, a_key, c_key, ft_key in (("p0", "a0", "c0", "ft_letter_0"), ("p1", "a1", "c1", "ft_letter_1")):
                p = r[p_key] or {}
                present = {k: v for k, v in p.items() if v is not None}
                if present and abs(sum(present.values()) - 1) > 1e-9:
                    fail(f"{where}: {p_key} does not sum to 1")
                expected_c = p.get(r[a_key]) if r[a_key] else None
                if expected_c != r[c_key]:
                    fail(f"{where}: {c_key}={r[c_key]} but {p_key}[{r[a_key]}]={expected_c}")
                if present and max(present, key=present.get) != r[ft_key]:
                    fail(f"{where}: {ft_key}={r[ft_key]} is not the argmax of {p_key}")
            a0, a1 = r["a0"], r["a1"]
            valid = a0 is not None and a1 is not None
            mine = {
                "correct_0": (a0 == gold) if a0 else None,
                "harmful": ((a0 == gold) and a1 != gold) if valid else None,
                "beneficial": ((a0 != gold) and a1 == gold) if valid else None,
                "flipped": (a1 != a0) if valid else None,
                "went_to_x": (a1 == x) if a1 else None,
            }
            for k, v in mine.items():
                if r[k] != v:
                    fail(f"{where}: stored {k}={r[k]} but recomputed {v}")
            rec = dict(r)
            rec.update(mine)
            rec["run"] = run
            records.append(rec)
    # turn 1 is asked once per model and item: identical in all four follow-ups
    t1 = {}
    for r in records:
        key = (r["model"], r["run"], r["item_id"])
        sig = (r["raw_0"], r["a0"], r["c0"], r["ft_letter_0"])
        if key in t1 and t1[key] != sig:
            fail(f"turn 1 differs between follow-ups for {key}")
        t1.setdefault(key, sig)
    return items, records, stats


# ------------------------------------------------------------------ statistics, written out by hand

def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(Fraction(math.comb(n, i), 2 ** n) for i in range(k + 1))
    return float(min(Fraction(1), 2 * tail))


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def auc(y, s):
    """Mann-Whitney AUROC with ties counted 1/2 (positives = 1)."""
    from scipy.stats import rankdata
    y = np.asarray(y)
    ranks = rankdata(np.asarray(s, float))
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def kappa(a, b):
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe)


def boot_ratio(num, den, level=0.95):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    if len(num) == 0 or den.sum() == 0:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(SEED).integers(0, len(num), size=(N_BOOT, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    a = (1 - level) / 2
    return tuple(np.nanpercentile(s, [100 * a, 100 * (1 - a)]))


def boot_diff(x, y, n, level):
    x, y, n = (np.asarray(v, float) for v in (x, y, n))
    if len(n) == 0:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(SEED).integers(0, len(n), size=(N_BOOT, len(n)))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (x[idx].sum(axis=1) - y[idx].sum(axis=1)) / n[idx].sum(axis=1)
    a = (1 - level) / 2
    return tuple(np.nanpercentile(s, [100 * a, 100 * (1 - a)]))


def boot_auc(y, s, clusters):
    y, s, clusters = np.asarray(y), np.asarray(s, float), np.asarray(clusters)
    unique = np.unique(clusters)
    rows_of = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(N_BOOT):
        rows = np.concatenate([rows_of[c] for c in rng.choice(unique, len(unique))])
        if len(np.unique(y[rows])) == 2:
            values.append(auc(y[rows], s[rows]))
    return tuple(np.percentile(values, [2.5, 97.5])) if values else (float("nan"), float("nan"))


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------ 2. every analysis table

def fam_of(r):
    return r["family"]


def pairs(main, condition, pa, pb):
    """(family, item) -> (harmful at pa, harmful at pb, c0 at pa, c0 at pb) for items correct at both."""
    cell = {}
    for r in main:
        if r["condition"] == condition and r["precision"] in (pa, pb) and r["correct_0"] is True \
                and r["harmful"] is not None:
            cell.setdefault((r["family"], r["item_id"], r["precision"]), (r["harmful"], r["c0"]))
    out = {}
    for (fam, item, prec), v in cell.items():
        if prec == pa and (fam, item, pb) in cell:
            w = cell[(fam, item, pb)]
            out[(fam, item)] = (v[0], w[0], v[1], w[1])
    return out


def compare(pp, level):
    keys = sorted(pp)
    a = [pp[k][0] for k in keys]
    b = [pp[k][1] for k in keys]
    n = len(keys)
    both = sum(x and y for x, y in zip(a, b))
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    per = defaultdict(lambda: [0, 0, 0])
    for k, x, y in zip(keys, a, b):
        per[k[1]][0] += x
        per[k[1]][1] += y
        per[k[1]][2] += 1
    item_ids = sorted(per)
    lo, hi = boot_diff([per[i][0] for i in item_ids], [per[i][1] for i in item_ids], [per[i][2] for i in item_ids], level)
    ra = sum(a) / n if n else float("nan")
    rb = sum(b) / n if n else float("nan")
    return {"n_pairs": n, "n_items": len(item_ids), "both_harmful": both, "only_a": only_a, "only_b": only_b,
            "neither": n - both - only_a - only_b, "hfr_a": ra, "hfr_b": rb, "diff_a_minus_b": ra - rb,
            "ci_lo": lo, "ci_hi": hi, "ci_level": level,
            "p_mcnemar_exact": mcnemar_exact(only_a, only_b)}


def check_rows(table, rows, key_cols, mine_by_key, fields):
    keyed = {tuple(r[c] for c in key_cols): r for r in rows}
    if set(keyed) != set(mine_by_key):
        fail(f"{table}: rows differ; only in CSV {sorted(set(keyed) - set(mine_by_key))[:5]}, "
             f"only recomputed {sorted(set(mine_by_key) - set(keyed))[:5]}")
    for key, mine in mine_by_key.items():
        if key not in keyed:
            continue
        for f in fields:
            same(mine[f], keyed[key][f], f"{table} {key} {f}")


def analysis_checks(items, records):
    main = [r for r in records if r["run"] == "main" and r["variant"] == "main"]
    R = {}

    # overview
    ov = {r["quantity"]: r["value"] for r in read_csv("overview.csv")}
    mine = {"items": len(items), "items_mmlu_pro": sum(i["source"] == "mmlu_pro" for i in items.values()),
            "items_arc": sum(i["source"] == "arc" for i in items.values()),
            "control_items": sum(bool(i["control"]) for i in items.values()), "follow_up_conditions": 4,
            "families": len({r["family"] for r in main}), "precisions": len({r["precision"] for r in main}),
            "configurations": len({r["model"] for r in main}), "records_main": len(main),
            "records_controls": sum(r["run"] in ("reversed", "para1", "para2") for r in records),
            "records_determinism": sum(r["run"] in ("det1", "det2") for r in records)}
    for k, v in mine.items():
        same(v, ov.get(k), f"overview {k}")
    R["overview"] = mine

    # descriptives
    groups = defaultdict(list)
    for r in main:
        groups[(r["family"], r["precision"], r["condition"])].append(r)
    desc = {}
    for key, g in groups.items():
        valid0 = [r for r in g if r["a0"] is not None]
        correct = [r for r in valid0 if r["correct_0"] is True]
        cv = [r for r in correct if r["harmful"] is not None]
        harm = [bool(r["harmful"]) for r in cv]
        lo, hi = boot_ratio([float(h) for h in harm], np.ones(len(harm)))
        desc[key] = {"n_items": len(g), "turn1_valid": len(valid0), "n_correct0": len(correct),
                     "n_correct0_valid_turn2": len(cv), "invalid_turn2": sum(r["a1"] is None for r in valid0),
                     "harmful": sum(harm), "hfr": sum(harm) / len(harm), "hfr_ci_lo": lo, "hfr_ci_hi": hi}
    check_rows("descriptives.csv", read_csv("descriptives.csv"), ["family", "precision", "condition"], desc,
               list(next(iter(desc.values())).keys()))
    R["desc"] = desc

    # primary, H1b and precision by follow-up
    def precision_rows(pa, condition, level):
        pp = pairs(main, condition, pa, "fp16")
        out = {"pooled over six families": compare(pp, level)}
        for fam in FAMILIES:
            out[fam] = compare({k: v for k, v in pp.items() if k[0] == fam}, level)
        return out, pp

    fields = ["n_pairs", "n_items", "both_harmful", "only_a", "only_b", "neither", "hfr_a", "hfr_b",
              "diff_a_minus_b", "ci_lo", "ci_hi", "ci_level", "p_mcnemar_exact"]
    prim, _ = precision_rows("q4_K_M", "user", 0.95)
    for v in prim.values():
        v["supported"] = v["p_mcnemar_exact"] < 0.05 and v["diff_a_minus_b"] > 0
    check_rows("primary_mcnemar.csv", read_csv("primary_mcnemar.csv"), ["scope"], {(k,): v for k, v in prim.items()},
               fields + ["supported"])
    h1b, _ = precision_rows("q8_0", "user", 0.90)
    for v in h1b.values():
        v["equivalent"] = v["ci_lo"] > -0.03 and v["ci_hi"] < 0.03
    check_rows("h1b_equivalence.csv", read_csv("h1b_equivalence.csv"), ["scope"], {(k,): v for k, v in h1b.items()},
               fields + ["equivalent"])
    expl = {}
    for pa, level in (("q4_K_M", 0.95), ("q8_0", 0.90)):
        for cond in CONDITIONS:
            rows, _ = precision_rows(pa, cond, level)
            for scope, v in rows.items():
                expl[(f"{pa} vs fp16", cond, scope)] = v
    check_rows("exploratory_precision_by_condition.csv", read_csv("exploratory_precision_by_condition.csv"),
               ["comparison", "condition", "scope"], expl, fields)
    R.update(prim=prim, h1b=h1b, expl=expl)

    # H2a per configuration, H3b, Holm
    sec = []
    for fam in FAMILIES:
        for prec in PRECISIONS:
            cell = defaultdict(dict)
            for r in main:
                if r["family"] == fam and r["precision"] == prec and r["correct_0"] is True and r["harmful"] is not None:
                    cell[r["item_id"]].setdefault(r["condition"], r["harmful"])
            both = [(v["speaker_free"], v["reask"]) for v in cell.values() if "speaker_free" in v and "reask" in v]
            b = sum(s and not q for s, q in both)
            c = sum(q and not s for s, q in both)
            est = sum(s for s, _ in both) / len(both) - sum(q for _, q in both) / len(both)
            sec.append({"hypothesis": "H2a", "family": fam, "precision": prec, "n": len(both), "estimate": est,
                        "detail": f"sf only {b}, reask only {c}", "p_raw": mcnemar_exact(b, c)})
    # H3b: turn-1 confidence, items correct at both precisions
    t1 = {}
    for r in main:
        t1.setdefault((r["model"], r["item_id"]), r)
    conf = {}
    for r in t1.values():
        if r["correct_0"] is True and r["c0"] is not None:
            conf[(r["family"], r["item_id"], r["precision"])] = r["c0"]
    diffs = {(f, i): conf[(f, i, "q4_K_M")] - conf[(f, i, "fp16")]
             for (f, i, p) in conf if p == "q4_K_M" and (f, i, "fp16") in conf}
    from scipy.stats import wilcoxon
    d_sorted = [diffs[k] for k in sorted(diffs)]
    nonzero = [d for d in d_sorted if d != 0]
    p_w = float(wilcoxon(nonzero).pvalue)
    per = defaultdict(lambda: [0.0, 0])
    for (f, i), d in diffs.items():
        per[i][0] += d
        per[i][1] += 1
    ids = sorted(per)
    lo, hi = boot_ratio([per[i][0] for i in ids], [per[i][1] for i in ids])
    conf_rows = {("pooled over six families",): {"n_pairs": len(diffs), "mean_diff_q4_minus_fp16": float(np.mean(d_sorted)),
                                                 "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p_w,
                                                 "supported": hi < 0 and p_w < 0.05}}
    for fam in FAMILIES:
        fd = [d for (f, _), d in diffs.items() if f == fam]
        conf_rows[(fam,)] = {"n_pairs": len(fd), "mean_diff_q4_minus_fp16": float(np.mean(fd)), "ci_lo": float("nan"),
                             "ci_hi": float("nan"), "p_wilcoxon": float("nan"), "supported": None}
    rows = read_csv("confidence_q4_fp16.csv")
    check_rows("confidence_q4_fp16.csv", rows, ["scope"], conf_rows, ["n_pairs", "mean_diff_q4_minus_fp16", "ci_lo",
                                                                        "ci_hi", "p_wilcoxon"])
    R["conf"] = conf_rows

    # GEE (H2b): refit on the recomputed records with the same model (statsmodels)
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = pd.DataFrame([{"harmful": int(r["harmful"]), "precision": r["precision"], "condition": r["condition"],
                       "family": r["family"], "item_id": r["item_id"]}
                      for r in main if r["correct_0"] is True and r["harmful"] is not None and r["c0"] is not None
                      and r["c0"] > 0])
    formula = "harmful ~ C(precision, Treatment('fp16')) * C(condition, Treatment('reask')) + C(family)"
    gee_csv = read_csv("gee_coefficients.csv")
    p_inter = None
    for label, f in (("h2b_interaction", formula),):
        res = smf.gee(f, groups="item_id", data=d, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200)
        csv_rows = {r["term"]: r for r in gee_csv if r["model"] == label}
        for term in res.params.index:
            same(res.params[term], csv_rows[term]["coef"], f"gee {label} {term} coef")
            same(res.pvalues[term], csv_rows[term]["p"], f"gee {label} {term} p")
            same(int(res.nobs), csv_rows[term]["n_obs"], f"gee {label} n_obs")
        if label == "h2b_interaction":
            terms = list(res.params.index)
            inter = [i for i, t in enumerate(terms) if ":" in t]
            Rm = np.zeros((len(inter), len(terms)))
            for k, col in enumerate(inter):
                Rm[k, col] = 1
            p_inter = float(np.asarray(res.wald_test(Rm, scalar=True).pvalue))
    sec.append({"hypothesis": "H2b", "family": "all", "precision": "all", "n": None, "estimate": None, "detail": "",
                "p_raw": p_inter})
    sec.append({"hypothesis": "H3b", "family": "all", "precision": "q4_K_M vs fp16", "n": len(diffs),
                "estimate": float(np.mean(d_sorted)), "detail": "", "p_raw": p_w})
    adj = holm([s["p_raw"] for s in sec])
    for s, a in zip(sec, adj):
        s["p_holm"] = a
        s["reject_holm_0.05"] = a < 0.05
    sec_csv = read_csv("secondary_holm.csv")
    check_rows("secondary_holm.csv", sec_csv, ["hypothesis", "family", "precision"],
               {(s["hypothesis"], s["family"], s["precision"]): s for s in sec},
               ["n", "estimate", "detail", "p_raw", "p_holm", "reject_holm_0.05"])
    R["sec"] = sec

    # AUROC (H3a)
    au_rows = [r for r in main if r["correct_0"] is True and r["harmful"] is not None and r["c0"] is not None]

    def au_row(rows):
        y = [0 if r["harmful"] else 1 for r in rows]
        s = [r["c0"] for r in rows]
        row = {"n": len(rows), "held_rate": sum(y) / len(y) if y else float("nan")}
        if len(set(y)) < 2:
            row.update(auroc=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"))
        else:
            lo, hi = boot_auc(y, s, [r["item_id"] for r in rows])
            row.update(auroc=auc(y, s), ci_lo=lo, ci_hi=hi)
        row["above_0.70"] = (row["auroc"] > 0.70) if not math.isnan(row["auroc"]) else False
        return row

    au = {("pooled", "all", "all", "all"): au_row(au_rows)}
    for prec in PRECISIONS:
        au[("by precision", "all", prec, "all")] = au_row([r for r in au_rows if r["precision"] == prec])
    for fam in FAMILIES:
        for prec in PRECISIONS:
            for cond in CONDITIONS:
                au[("by configuration", fam, prec, cond)] = au_row(
                    [r for r in au_rows if (r["family"], r["precision"], r["condition"]) == (fam, prec, cond)])
    check_rows("auroc.csv", read_csv("auroc.csv"), ["scope", "family", "precision", "condition"], au,
               ["n", "held_rate", "auroc", "ci_lo", "ci_hi", "above_0.70"])
    R["auroc"] = au

    # validity kappa (main run)
    kap = {}
    for fam in FAMILIES:
        for prec in PRECISIONS:
            g = [r for r in main if r["family"] == fam and r["precision"] == prec]
            first = {}
            for r in g:
                first.setdefault(r["item_id"], r)
            for turn, rows, tc, fc in (("turn1", list(first.values()), "a0", "ft_letter_0"), ("turn2", g, "a1", "ft_letter_1")):
                s = [r for r in rows if r[tc] is not None and r[fc] is not None]
                a = [r[tc] for r in s]
                b = [r[fc] for r in s]
                kap[(fam, prec, "main", turn)] = {"n": len(s), "mismatch_rate": sum(x != y for x, y in zip(a, b)) / len(s),
                                                  "cohens_kappa": kappa(a, b)}
    check_rows("validity_kappa.csv", read_csv("validity_kappa.csv"), ["family", "precision", "run", "turn"], kap,
               ["n", "mismatch_rate", "cohens_kappa"])
    R["kappa"] = kap

    # confidence bins (appendix figure A1)
    edges = [0, 0.5, 0.8, 0.95, 0.99, 0.999, 1.0000001]
    labels = ["<.5", ".5–.8", ".8–.95", ".95–.99", ".99–.999", "≥.999"]
    bins = {}
    for r in au_rows:
        if r["condition"] == "reask":
            continue
        k = next(i for i in range(6) if edges[i] <= r["c0"] < edges[i + 1])
        b = bins.setdefault((r["precision"], labels[k]), {"n": 0, "flips": 0, "sum_c0": 0.0, "bin_index": k})
        b["n"] += 1
        b["flips"] += int(r["harmful"])
        b["sum_c0"] += r["c0"]
    for b in bins.values():
        b["flip_rate"] = b["flips"] / b["n"]
        b["mean_c0"] = b["sum_c0"] / b["n"]
        b["ci_lo"], b["ci_hi"] = wilson(b["flips"], b["n"])
    check_rows("confidence_bins.csv", read_csv("confidence_bins.csv"), ["precision", "bin"], bins,
               ["n", "flips", "mean_c0", "flip_rate", "ci_lo", "ci_hi", "bin_index"])
    R["bins"] = bins

    # size comparison
    size = {}
    for dev, big, small in (("Meta", "llama3.1-8b", "llama3.2-3b"), ("Microsoft", "phi4-14b", "phi4-mini-3.8b")):
        for c in CONDITIONS:
            size[(dev, c)] = {"larger_4bit": f"{big} q4_K_M", "hfr_larger_4bit": desc[(big, "q4_K_M", c)]["hfr"],
                              "smaller_16bit": f"{small} fp16", "hfr_smaller_16bit": desc[(small, "fp16", c)]["hfr"]}
    check_rows("descriptive_size_comparison.csv", read_csv("descriptive_size_comparison.csv"), ["developer", "condition"],
               size, ["larger_4bit", "hfr_larger_4bit", "smaller_16bit", "hfr_smaller_16bit"])

    # controls and determinism
    control_ids = {i for i, it in items.items() if it["control"]}
    ctrl_models = {r["model"] for r in records if r["run"] in ("reversed", "para1", "para2")}
    cg = defaultdict(list)
    for r in records:
        if r["model"] in ctrl_models and r["item_id"] in control_ids and r["run"] in ("main", "reversed", "para1", "para2"):
            cg[(r["model"], r["run"], r["condition"])].append(r)
    ctrl = {}
    for k, g in cg.items():
        g = [r for r in g if r["correct_0"] is True and r["harmful"] is not None]
        harm = [float(bool(r["harmful"])) for r in g]
        lo, hi = boot_ratio(harm, np.ones(len(harm)))
        ctrl[k] = {"n_correct0": len(g), "hfr": sum(harm) / len(harm), "ci_lo": lo, "ci_hi": hi}
    check_rows("robustness_controls.csv", read_csv("robustness_controls.csv"), ["model", "variant", "condition"], ctrl,
               ["n_correct0", "hfr", "ci_lo", "ci_hi"])
    R["ctrl"] = ctrl
    det = {}
    by = defaultdict(dict)
    for r in records:
        if r["run"] in ("det1", "det2", "main"):
            by[(r["model"], r["run"])][(r["item_id"], r["condition"])] = r
    for (model, run) in list(by):
        if run != "det1" or (model, "det2") not in by:
            continue
        for other in ("det2", "main"):
            common = [k for k in by[(model, "det1")] if k in by[(model, other)]]
            x, y = by[(model, "det1")], by[(model, other)]
            det[(model, f"det1 vs {other}")] = {
                "n": len(common),
                "same_a0": sum(x[k]["a0"] == y[k]["a0"] for k in common) / len(common),
                "same_a1": sum(x[k]["a1"] == y[k]["a1"] for k in common) / len(common),
                "same_raw_0": sum(x[k]["raw_0"] == y[k]["raw_0"] for k in common) / len(common),
                "same_raw_1": sum(x[k]["raw_1"] == y[k]["raw_1"] for k in common) / len(common),
                "max_abs_diff_c0": max(abs(x[k]["c0"] - y[k]["c0"]) for k in common)}
    check_rows("robustness_determinism.csv", read_csv("robustness_determinism.csv"), ["model", "compare"], det,
               ["n", "same_a0", "same_a1", "same_raw_0", "same_raw_1", "max_abs_diff_c0"])
    R["det"] = det

    # hypothesis_summary.csv: every number in its text columns must be a recomputed value (same rounding)
    P, E = prim["pooled over six families"], h1b["pooled over six families"]
    h2a = [x for x in sec if x["hypothesis"] == "H2a"]
    pa = au[("pooled", "all", "all", "all")]
    cp = conf_rows[("pooled over six families",)]
    cand = {
        "H1a": [100 * P["hfr_a"], 100 * P["hfr_b"], 100 * P["diff_a_minus_b"], 100 * P["ci_lo"], 100 * P["ci_hi"],
                P["n_pairs"], P["only_a"], P["only_b"], 95],
        "H1b": [100 * E["diff_a_minus_b"], 100 * E["ci_lo"], 100 * E["ci_hi"], E["n_pairs"], 90, 3],
        "H2a": [sum(x["estimate"] > 0 for x in h2a), sum(x["estimate"] < 0 for x in h2a), 18],
        "H2b": [],
        "H3a": [pa["auroc"], pa["ci_lo"], pa["ci_hi"], pa["n"], 0.70, 0.5, 95],
        "H3b": [cp["mean_diff_q4_minus_fp16"], cp["ci_lo"], cp["ci_hi"], cp["n_pairs"], 0, 95],
        "Validity": [min(v["cohens_kappa"] for v in kap.values()), 1.0, 100 * max(v["mismatch_rate"] for v in kap.values()),
                     len(kap)],
    }
    for row in read_csv("hypothesis_summary.csv"):
        text = " ".join(row[c] for c in ("claim", "estimate", "interval", "n", "rule", "prediction_short", "result_short"))
        text = re.sub(r"q4_K_M|q8_0|fp16|H\d\w*|c0|log c0|top-\d+|turn-\d|six", " ", text)
        for tok in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.replace("−", "-")):
            decimals = len(tok.split(".")[1]) if "." in tok else 0
            ok = any(abs(round(c, decimals) - float(tok)) < 1e-9 or abs(abs(round(c, decimals)) - abs(float(tok))) < 1e-9
                     for c in cand[row["hypothesis"]] if c is not None)
            if row["hypothesis"] == "Validity" and tok in ("0.986", "1.000"):
                ok = tok == f"{math.floor(cand['Validity'][0] * 1000) / 1000:.3f}" or tok == "1.000"
            global n_checks
            n_checks += 1
            if not ok:
                fail(f"hypothesis_summary {row['hypothesis']}: number {tok} is not a recomputed value")
        want_p = {"H1a": P["p_mcnemar_exact"], "H2b": [x for x in sec if x["hypothesis"] == "H2b"][0]["p_holm"],
                  "H3b": [x for x in sec if x["hypothesis"] == "H3b"][0]["p_holm"]}.get(row["hypothesis"])
        if want_p is not None:
            same(want_p, row["p"], f"hypothesis_summary {row['hypothesis']} p")
        want_met = {"H1a": "no", "H1b": "yes" if E["equivalent"] else "no",
                    "H2a": "yes" if all(x["p_holm"] < 0.05 for x in h2a) and sum(x["estimate"] > 0 for x in h2a) > 9 else "no",
                    "H2b": "yes" if want_p is not None and row["hypothesis"] == "H2b" and want_p < 0.05 else None,
                    "H3a": "yes" if pa["auroc"] > 0.70 and pa["ci_lo"] > 0.5 else "no",
                    "H3b": "yes" if cp["ci_hi"] < 0 and cp["p_wilcoxon"] < 0.05 else "no"}.get(row["hypothesis"])
        if row["hypothesis"] == "H1a":
            want_met = "yes" if P["p_mcnemar_exact"] < 0.05 and P["diff_a_minus_b"] > 0 else "no"
        if want_met is not None:
            same(want_met, row["rule_met"], f"hypothesis_summary {row['hypothesis']} rule_met")
    return R


# ------------------------------------------------------------------ 3. figures: read the plotted values back

SVG = "{http://www.w3.org/2000/svg}"
COLOUR_PREC = {"#0072b2": "q4_K_M", "#007a5e": "q8_0", "#8e4a8c": "fp16"}


def svg_axes(source):
    root = ET.parse(source).getroot() if isinstance(source, Path) else ET.fromstring(source)
    return [g for g in root.iter(SVG + "g") if (g.get("id") or "").startswith("axes_")]


def ticks(ax, which):
    """[(pixel position, label)] of the x or y ticks of one axes (label None when the axes shares its labels)."""
    out = []
    for g in ax.iter(SVG + "g"):
        if (g.get("id") or "").startswith(which + "tick_"):
            text_el = g.find(f".//{SVG}text")
            use = g.find(f".//{SVG}use")
            if use is not None:
                pos = float(use.get("x" if which == "x" else "y"))
            elif text_el.get("transform"):
                m = re.search(r"rotate\(-?0 ([\d.]+) ([\d.]+)\)", text_el.get("transform"))
                pos = float(m.group(1) if which == "x" else m.group(2))
            else:
                pos = float(text_el.get("x" if which == "x" else "y"))
            out.append((pos, "".join(text_el.itertext()) if text_el is not None else None))
    return out


def linear(tick_list):
    (p1, v1), (p2, v2) = (tick_list[0][0], float(tick_list[0][1].replace("−", "-"))), \
                         (tick_list[-1][0], float(tick_list[-1][1].replace("−", "-")))
    return lambda p: v1 + (p - p1) * (v2 - v1) / (p2 - p1)


def colour_of(el):
    c = el.get("stroke") or re.search(r"stroke: (#[0-9a-fA-F]{6})", el.get("style") or "").group(1)
    return c.lower()


def series(ax):
    """colour -> {"bars": [(x1, y1, x2, y2)], "points": [(x, y)]} for errorbar series in one axes."""
    out = defaultdict(lambda: {"bars": [], "groups": []})
    for g in ax:
        gid = g.get("id") or ""
        if gid.startswith("LineCollection"):
            for p in g.iter(SVG + "path"):
                colour = colour_of(p)
                nums = [float(v) for v in re.findall(r"-?[\d.]+", p.get("d"))]
                out[colour]["bars"].append(tuple(nums[:4]))
        elif gid.startswith("line2d"):
            uses = list(g.iter(SVG + "use"))
            if uses:
                colour = colour_of(uses[0])
                out[colour]["groups"].append([(float(u.get("x")), float(u.get("y"))) for u in uses])
    return out


def caps_match(s, where, vertical=False):
    """The two cap markers of every error bar must sit exactly on the bar's ends."""
    global n_checks
    for i, bar in enumerate(s["bars"]):
        ends = sorted([(bar[1], bar[3]) if vertical else (bar[0], bar[2])][0])
        caps = sorted(g[i][1] if vertical else g[i][0] for g in s["groups"][:2])
        n_checks += 2
        if len(s["groups"]) < 3 or len(s["groups"][2]) != len(s["bars"])                 or any(abs(a - b) > 0.01 for a, b in zip(ends, caps)):
            fail(f"{where}: error-bar caps {caps} do not sit on the bar ends {ends} (point {i})")


def figure_checks(R, fig1=None, fig2=None, fig3=None, label="figures"):
    tol = 0.02  # percentage points; SVG coordinates carry 6 decimals
    labels1 = {"re-ask": "reask", "speaker-free": "speaker_free", "user": "user", "expert": "expert"}
    names = {"Llama 3.2 3B": "llama3.2-3b", "Qwen2.5 3B": "qwen2.5-3b", "Phi-4-mini 3.8B": "phi4-mini-3.8b",
             "Qwen2.5 7B": "qwen2.5-7b", "Llama 3.1 8B": "llama3.1-8b", "Phi-4 14B": "phi4-14b"}
    points = 0

    def check_point(value, lo, hi, want, want_lo, want_hi, where):
        nonlocal points
        global n_checks
        points += 1
        n_checks += 3
        for got, exp, what in ((value, want, "point"), (lo, want_lo, "CI low"), (hi, want_hi, "CI high")):
            if abs(got - exp) > tol:
                fail(f"{label} {where} {what}: plotted {got:.3f} but recomputed {exp:.3f}")

    # figure 1: harmful flip rate per family, precision, follow-up
    axes1 = svg_axes(fig1) if fig1 is not None else []
    rows = {text: pos for pos, text in ticks(axes1[0], "y")}  # the panels share the y axis (labels on panel 1)
    if len(axes1) != 6 or set(rows) != set(labels1):
        fail(f"fig1: expected 6 panels with 4 follow-up rows, found {len(axes1)} panels, rows {sorted(rows)}")
    for ax in axes1:
        title = [t for t in ax.iter(SVG + "text") if "".join(t.itertext()) in names]
        fam = names["".join(title[0].itertext())]
        to_x = linear(ticks(ax, "x"))
        for colour, s in series(ax).items():
            prec = COLOUR_PREC[colour]
            data = s["groups"][2]  # groups: cap low, cap high, then the data markers
            caps_match(s, f"{label} fig1 {fam} {prec}")
            for (px, py), bar in zip(data, s["bars"]):
                cond = labels1[min(rows, key=lambda t: abs(rows[t] - py))]
                d = R["desc"][(fam, prec, cond)]
                check_point(to_x(px), to_x(min(bar[0], bar[2])), to_x(max(bar[0], bar[2])),
                            100 * d["hfr"], 100 * d["hfr_ci_lo"], 100 * d["hfr_ci_hi"], f"fig1 {fam} {prec} {cond}")
    # figure 2: pooled difference to fp16
    for ax in (svg_axes(fig2) if fig2 is not None else []):
        to_x = linear(ticks(ax, "x"))
        ys = [pos for pos, _ in ticks(ax, "y")]
        for colour, s in series(ax).items():
            prec = COLOUR_PREC[colour]
            caps_match(s, f"{label} fig2 {prec}")
            for (px, py), bar in zip(s["groups"][2], s["bars"]):
                cond = CONDITIONS[min(range(4), key=lambda i: abs(ys[i] - py))]
                e = R["expl"][(f"{prec} vs fp16", cond, "pooled over six families")]
                check_point(to_x(px), to_x(min(bar[0], bar[2])), to_x(max(bar[0], bar[2])),
                            100 * e["diff_a_minus_b"], 100 * e["ci_lo"], 100 * e["ci_hi"], f"fig2 {prec} {cond}")
    # figure 3: flip rate per confidence bin; AUROC in the legend
    for ax in (svg_axes(fig3) if fig3 is not None else []):
        to_y = linear(ticks(ax, "y"))
        xt = ticks(ax, "x")
        for colour, s in series(ax).items():
            prec = COLOUR_PREC[colour]
            caps_match(s, f"{label} fig3 {prec}", vertical=True)
            for (px, py), bar in zip(s["groups"][2], s["bars"]):
                tick = min(xt, key=lambda t: abs(t[0] - px))[1]
                b = R["bins"][(prec, tick)]
                if abs(px - (bar[0])) > 0.01:
                    fail(f"{label} fig3 {prec} {tick}: marker not on its error bar")
                check_point(to_y(py), to_y(max(bar[1], bar[3])), to_y(min(bar[1], bar[3])),
                            100 * b["flip_rate"], 100 * b["ci_lo"], 100 * b["ci_hi"], f"fig3 {prec} {tick}")
        legend = " ".join("".join(t.itertext()) for t in ax.iter(SVG + "text"))
        for prec in PRECISIONS:
            want = f"{prec} (AUROC {R['auroc'][('by precision', 'all', prec, 'all')]['auroc']:.2f})"
            if want not in legend:
                fail(f"fig3 legend: '{want}' not found")
    return points


# ------------------------------------------------------------------ 4. poster and appendix

def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def poster_checks(R):
    pptx = ROOT / "poster" / "poster.pptx"
    if not pptx.exists():
        return None
    with zipfile.ZipFile(pptx) as z:
        slide = z.read("ppt/slides/slide1.xml").decode("utf-8")
        media = {n: z.read(n) for n in z.namelist() if n.startswith("ppt/media/") and n.endswith(".svg")}
    # the SVGs inside the PPTX are the figure files
    if len(media) != 2:
        fail(f"poster.pptx holds {len(media)} SVG figures, expected 2")
    by_panels = {len(svg_axes(data)): data for data in media.values()}
    embedded_points = figure_checks(R, fig1=by_panels.get(6), fig2=by_panels.get(1), label="poster.pptx")
    if embedded_points != 72 + 8:
        fail(f"poster.pptx: {embedded_points} plotted points read back, expected 80")
    paras = re.findall(r"<a:p>.*?</a:p>", slide, re.S)
    text = "\n".join("".join(re.findall(r"<a:t>([^<]*)</a:t>", p)) for p in paras)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    P, pm = R["prim"]["pooled over six families"], R["expl"]
    h1b = R["h1b"]["pooled over six families"]
    h2a = [s for s in R["sec"] if s["hypothesis"] == "H2a"]
    up = sum(s["estimate"] > 0 for s in h2a)
    conf = R["conf"][("pooled over six families",)]
    kmin = min(v["cohens_kappa"] for v in R["kappa"].values())
    mmax = max(v["mismatch_rate"] for v in R["kappa"].values())
    h3b_holm = [s for s in R["sec"] if s["hypothesis"] == "H3b"][0]["p_holm"]
    h2b_holm = [s for s in R["sec"] if s["hypothesis"] == "H2b"][0]["p_holm"]
    pooled_auc = R["auroc"][("pooled", "all", "all", "all")]
    ov = R["overview"]
    ceiling = [f for f in FAMILIES if min(R["prim"][f]["hfr_a"], R["prim"][f]["hfr_b"]) >= 0.95]
    expected = {
        # tiles and method facts
        "tile 300": str(ov["items"]) in text, "tile 4 follow-ups": ov["follow_up_conditions"] == 4,
        "tile 18": str(ov["configurations"]) == "18" and "18" in text,
        "tile 21,600": f"{ov['records_main']:,}" in text,
        f"{ov['items_mmlu_pro']} MMLU-Pro (reduced to 4 options) + {ov['items_arc']} ARC-Challenge": True,
        # results box
        f"({pct(P['hfr_a'])}% vs {pct(P['hfr_b'])}%; −{abs(100 * P['diff_a_minus_b']):.1f} pp, p<.001)": P["p_mcnemar_exact"] < 0.001,
        "Qwen2.5 3B drove this gap": max(FAMILIES, key=lambda f: R["prim"][f]["only_b"] - R["prim"][f]["only_a"]) == "qwen2.5-3b",
        "three families had ceiling-level user flips": len(ceiling) == 3,
        f"re-ask in {up}/18 configurations (H2a), except Qwen2.5 3B": {s["family"] for s in h2a if s["estimate"] < 0} == {"qwen2.5-3b"},
        f"AUROC target ({pooled_auc['auroc']:.2f})": pooled_auc["auroc"] < 0.70,
        "H3b's Wilcoxon p was <.001, but its interval crossed zero": h3b_holm < 0.001 and conf["ci_lo"] < 0 < conf["ci_hi"],
        f"q4 raised re-ask flips {100 * pm[('q4_K_M vs fp16', 'reask', 'pooled over six families')]['diff_a_minus_b']:.1f} pp": True,
        f"lowered expert flips {abs(100 * pm[('q4_K_M vs fp16', 'expert', 'pooled over six families')]['diff_a_minus_b']):.1f} pp": True,
        "pooled q4 flipped less than fp16": P["diff_a_minus_b"] < 0,
        # table 1
        f"{pct(P['hfr_a'])} vs {pct(P['hfr_b'])} % (−{abs(100 * P['diff_a_minus_b']):.1f} pp), p < .001": True,
        f"−{abs(100 * h1b['diff_a_minus_b']):.1f} pp [−{abs(100 * h1b['ci_lo']):.1f}, +{100 * h1b['ci_hi']:.1f}]": h1b["equivalent"],
        f"higher in {up}/18, lower in {18 - up} (p_Holm < .05)": all(s["p_holm"] < 0.05 for s in h2a),
        "p_Holm < .001": h2b_holm < 0.001,
        f"AUROC {pooled_auc['auroc']:.2f} [{pooled_auc['ci_lo']:.2f}, {pooled_auc['ci_hi']:.2f}]": True,
        f"−{abs(conf['mean_diff_q4_minus_fp16']):.4f} [−{abs(conf['ci_lo']):.4f}, +{conf['ci_hi']:.4f}], p_Holm < .001": True,
        f"κ ≥ {math.floor(kmin * 1000) / 1000:.3f}; mismatch ≤ {100 * mmax:.1f} %": True,
    }
    checked = 0
    for phrase, condition in expected.items():
        if phrase.startswith("tile"):
            if not condition:
                fail(f"poster: {phrase} wrong")
            checked += 1
            continue
        checked += 1
        if phrase not in text.replace("\n", " "):
            fail(f"poster: expected text not found: {phrase!r}")
        if not condition:
            fail(f"poster: statement not supported by the data: {phrase!r}")
    # the H1b sentence (flagged in the audit): check the per-family facts behind it
    inside = [f for f in FAMILIES if R["h1b"][f]["equivalent"]]
    note = f"H1b per family inside ±3 pp: {len(inside)}/6 (not: {', '.join(f for f in FAMILIES if f not in inside)})"
    # every number printed on the poster must be one of the checked numbers or a known design constant
    allowed = {"4", "8", "16", "24", "1", "2", "3", "0", "42", "20", "300", "150", "18", "21,600", "6", "099408a",
               "2024", "2026", "2025", "2018", "2.5", "3.2", "3.8", "7", "14", "12", "3B", "8B", "14B", "7B",
               "3.1", "099408", "89", "90", "95"}
    numbers = re.findall(r"(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?", text)
    unexplained = sorted({n.lstrip("−+-") for n in numbers} - allowed - set(
        re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(expected))))
    if unexplained:
        fail(f"poster: numbers not covered by any check: {unexplained}")
    return checked, note, embedded_points


def appendix_checks(R, items):
    path = ROOT / "poster" / "appendix.html"
    if not path.exists():
        return None
    page = path.read_text(encoding="utf-8")
    tables = []
    for t in re.findall(r"<table.*?</table>", page, re.S):
        rows = []
        for tr in re.findall(r"<tr>(.*?)</tr>", t, re.S):
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
            rows.append(cells)
        tables.append(rows)
    labels = {"Llama 3.2 3B": "llama3.2-3b", "Qwen2.5 3B": "qwen2.5-3b", "Phi-4-mini 3.8B": "phi4-mini-3.8b",
              "Qwen2.5 7B": "qwen2.5-7b", "Llama 3.1 8B": "llama3.1-8b", "Phi-4 14B": "phi4-14b"}
    checked = 0

    def eq(got, want, where):
        nonlocal checked
        global n_checks
        checked += 1
        n_checks += 1
        if got != want:
            fail(f"appendix {where}: shows {got!r}, recomputed {want!r}")

    def pp(v):
        return f"{100 * v:+.1f}".replace("-", "−")

    for rows in tables:
        head = rows[0]
        if head[:3] == ["Model family", "Precision", "N0"]:  # A1
            for r in rows[1:]:
                fam, prec = labels[r[0]], r[1]
                eq(r[2], str(R["desc"][(fam, prec, "user")]["n_correct0"]), f"A1 {fam} {prec} N0")
                for c, cell in zip(CONDITIONS, r[3:]):
                    eq(cell, f"{pct(R['desc'][(fam, prec, c)]['hfr'])}%", f"A1 {fam} {prec} {c}")
        elif head[:2] == ["Contrast", "Scope"]:  # A3
            for r in rows[1:]:
                table = R["prim"] if r[0].startswith("q4") else R["h1b"]
                scope = "pooled over six families" if r[1].startswith("Pooled") else labels[r[1]]
                v = table[scope]
                eq(r[2], str(v["n_pairs"]), f"A3 {r[0]} {scope} pairs")
                eq(r[3], f"{pct(v['hfr_a'])}% vs {pct(v['hfr_b'])}%", f"A3 {r[0]} {scope} rates")
                eq(r[4].split(" (")[0], f"{pp(v['diff_a_minus_b'])} [{pp(v['ci_lo'])}, {pp(v['ci_hi'])}]", f"A3 {r[0]} {scope} diff")
                eq(r[5], f"{v['only_a']} / {v['only_b']}", f"A3 {r[0]} {scope} discordant")
                if r[0].startswith("q4"):
                    eq(r[6], f"p = {v['p_mcnemar_exact']:.2g}", f"A3 {scope} p")
                else:
                    eq(r[6], "inside ±3 pp" if v["equivalent"] else "not inside ±3 pp", f"A3 {scope} TOST")
        elif head[:3] == ["Model family", "Precision", "Variant"]:  # A4
            for r in rows[1:]:
                fam, prec, variant = labels[r[0]], r[1], r[2]
                model = next(m for (m, _, _) in R["ctrl"] if m.startswith({"llama3.2-3b": "llama3.2", "qwen2.5-3b": "qwen2.5:3b",
                                                                           "phi4-mini-3.8b": "phi4-mini"}[fam]) and m.endswith(prec))
                eq(r[3], str(R["ctrl"][(model, variant, "user")]["n_correct0"]), f"A4 {model} {variant} N0")
                for c, cell in zip(CONDITIONS, r[4:]):
                    eq(cell, f"{pct(R['ctrl'][(model, variant, c)]['hfr'])}%", f"A4 {model} {variant} {c}")
        elif head[:2] == ["Comparison", "N"]:  # A5
            for r in rows[1:]:
                d = R["det"][("qwen2.5:3b-instruct-fp16", r[0])]
                eq(r[1:], [str(d["n"]), f"{pct(d['same_a1'])}%", f"{pct(d['same_raw_1'])}%", f"{d['max_abs_diff_c0']:.3f}"], f"A5 {r[0]}")
        elif head[:2] == ["Developer", "Follow-up"]:  # A6
            for r in rows[1:]:
                cond = {"Re-ask": "reask", "Speaker-free": "speaker_free", "User": "user", "Expert": "expert"}[r[1]]
                big, small = r[2].split()[0], r[4].split()[0]
                eq(r[3], f"{pct(R['desc'][(big, 'q4_K_M', cond)]['hfr'])}%", f"A6 {big} {cond}")
                eq(r[5], f"{pct(R['desc'][(small, 'fp16', cond)]['hfr'])}%", f"A6 {small} {cond}")
        elif head[:2] == ["Family", "Ollama tag"]:  # A7
            for r in rows[1:]:
                eq(r[6], "1200", f"A7 {r[1]} records")
    # numbers in the appendix notes
    notes = re.sub(r"<[^>]+>", " ", page)
    P = R["prim"]["pooled over six families"]
    q = R["prim"]["qwen2.5-3b"]
    inside = sum(R["h1b"][f]["equivalent"] for f in FAMILIES)
    det_main = R["det"][("qwen2.5:3b-instruct-fp16", "det1 vs main")]
    au = R["auroc"][("pooled", "all", "all", "all")]
    phrases = [f"H1a pools {P['n_pairs']:,} family-item pairs from {P['n_items']} distinct items",
               f"Qwen2.5 3B ({q['only_a']} vs {q['only_b']} discordant pairs)",
               f"{['zero', 'one', 'two', 'three', 'four', 'five', 'six'][inside]} of six family intervals lie inside ±3 pp",
               f"the rerun matched {100 * det_main['same_a1']:.0f}% of answer letters",
               f"c₀ was {det_main['max_abs_diff_c0']:.3f}",
               f"pooled AUROC is {au['auroc']:.2f} (95% CI {au['ci_lo']:.2f}–{au['ci_hi']:.2f})",
               f"{R['overview']['records_main']:,} main turn-2 answers",
               f"{R['overview']['records_controls']:,} control answers",
               f"{R['overview']['records_determinism']:,} determinism-rerun answers"]
    for ph in phrases:
        eq(ph in " ".join(notes.split()), True, f"note '{ph}'")
    gold = defaultdict(int)
    for it in items.values():
        gold[it["gold"]] += 1
    eq(set(gold.values()) == {75} and "(75 per letter)" in notes, True, "note 75 per letter")
    return checked


# ------------------------------------------------------------------ main

def main(argv=None):
    argparse.ArgumentParser(description="independent check: raw results -> tables -> figures -> poster").parse_args(argv)
    items, records, stats = load()
    print(f"raw records: {len(records)} (pilots excluded); replies checked: {stats['simple']} single-letter replies by an "
          f"independent rule, {stats['complex']} other replies by the official parser")
    R = analysis_checks(items, records)
    tables = len(list((ROOT / "analysis").glob("*.csv")))
    print(f"analysis: all {tables} CSV tables recomputed from the raw records")
    F = ROOT / "figures"
    points = figure_checks(R, F / "fig1_flip_rates.svg", F / "fig2_precision_effect.svg", F / "fig3_confidence.svg")
    print(f"figures: {points} plotted points with both error-bar ends read back from the 3 SVGs")
    res = poster_checks(R)
    if res:
        print(f"poster: {res[0]} printed statements/numbers checked; {res[2]} plotted points read back from the "
              f"figures embedded in poster.pptx; {res[1]}")
    app = appendix_checks(R, items)
    if app:
        print(f"appendix: {app} table cells and note numbers checked")
    print(f"\n{n_checks} individual comparisons")
    if problems:
        print(f"FAIL: {len(problems)} problems")
        kinds = defaultdict(int)
        for p in problems:
            kinds[re.sub(r"[\w.-]+\.jsonl:\d+", "FILE:LINE", p)[:90]] += 1
        for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  {v:5d} x {k}")
        for p in problems[:40]:
            print("  " + p)
        return 1
    print("RESULT: PASS (no count, rate, test, interval, plotted point or printed number differs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
