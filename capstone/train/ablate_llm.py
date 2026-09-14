"""Does the LLM teacher actually teach the student? Ablate it and find out.

WHY THIS EXISTS
DESIGN v2.0 §9.5 claims the student is distilled from an LLM teacher. `train_mixed.py`
folds `data/traces_llm/train.jsonl` into the mix and prints a NOTE when the file is
missing -- but nothing ever measured whether those rows change the model. They are 89
rows against 36,783 (0.7% after the x3 weight), so the honest prior is "no".

A single training run cannot answer this. `train_mixed.py` hardcodes random_state=0, and
seed-to-seed spread on this task is about +/-2 points -- larger than any effect 89 rows
could plausibly have. So this runs BOTH arms across several seeds and reports the mean
and the spread. Anything inside the spread is not a result.

    python3 train/ablate_llm.py --seeds 0 1 2 3

Needs scikit-learn. Result as of 14 Sep 2026: +0.14 +/- 2.51 points, i.e. nothing.
See data/student_v9/llm_ablation.txt.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "tm", os.path.join(HERE, "train", "train_mixed.py"))
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)          # reuse ITS cost matrix and call relabelling, so the
                                      # ablation cannot drift from the real training path


def load(path: str, src: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        r = json.loads(line)
        r["source"] = src
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refsim", default="../data/traces_all")
    ap.add_argument("--ns3", default="../data/traces_ns3")
    ap.add_argument("--bridge", default="../data/traces_bridge/train.jsonl")
    ap.add_argument("--llm", default="../data/traces_llm/train.jsonl")
    ap.add_argument("--llm-weight", type=int, default=3)
    ap.add_argument("--bridge-weight", type=int, default=4)
    ap.add_argument("--held-out-family", default="sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    a = ap.parse_args()

    tr = load(os.path.join(a.refsim, "train.jsonl"), "refsim")
    n3 = load(os.path.join(a.ns3, "train.jsonl"), "ns3"); tm.relabel_calls(n3)
    br = load(a.bridge, "bridge")
    lm = load(a.llm, "llm")
    va = load(os.path.join(a.refsim, "val.jsonl"), "refsim")
    va3 = load(os.path.join(a.ns3, "val.jsonl"), "ns3"); tm.relabel_calls(va3)
    te = [json.loads(l) for l in open(os.path.join(a.ns3, "test.jsonl"))]

    if not lm:
        print(f"no LLM rows at {a.llm}; nothing to ablate"); return
    fams = {}
    for r in lm:
        fams[r.get("family", "?")] = fams.get(r.get("family", "?"), 0) + 1
    base = len(tr) + len(n3) + len(br) * a.bridge_weight
    print(f"LLM rows: {len(lm)} across {len(fams)} families {fams}")
    print(f"mix: {len(lm) * a.llm_weight} of {base + len(lm) * a.llm_weight} "
          f"= {100 * len(lm) * a.llm_weight / (base + len(lm) * a.llm_weight):.2f}%\n")

    Xt = np.asarray([r["features"] for r in te], np.float32)
    yt = np.asarray([r["cause"] for r in te])
    famt = np.asarray([r.get("family", "?") for r in te])
    keep = famt != a.held_out_family        # the held-out family swamps the comparison
    val = va + va3
    yvc = np.asarray([r["cause"] for r in val])

    def arm(with_llm: bool, seed: int) -> float:
        rows = tr + n3 + br * a.bridge_weight + (lm * a.llm_weight if with_llm else [])
        X = np.asarray([r["features"] for r in rows], np.float32)
        y = np.asarray([r["cause"] for r in rows])
        enc = LabelEncoder().fit(np.concatenate([y, yvc]))
        m = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=800, alpha=1e-3,
                          random_state=seed, early_stopping=True,
                          n_iter_no_change=30).fit(X, enc.transform(y))
        cl = list(enc.inverse_transform(m.classes_))
        idx = [tm.CAUSE_ORDER.index(c) for c in cl]
        P = m.predict_proba(Xt)
        pred = np.asarray(cl)[(P @ tm.COST[np.ix_(idx, idx)]).argmin(axis=1)]
        return float((pred == yt)[keep].mean())

    print(f"{'seed':>5s} {'oracle only':>13s} {'+LLM rows':>11s} {'delta':>9s}")
    deltas = []
    for s in a.seeds:
        x, y = arm(False, s), arm(True, s)
        deltas.append(y - x)
        print(f"{s:5d} {100*x:12.2f}% {100*y:10.2f}% {100*(y-x):+8.2f}%")
    d = np.asarray(deltas)
    print(f"\n  mean delta {100*d.mean():+.2f} points, sd {100*d.std(ddof=1):.2f}, "
          f"range {100*d.min():+.2f} .. {100*d.max():+.2f}")
    if abs(d.mean()) < d.std(ddof=1):
        print("\n  The effect is inside the seed noise. The LLM rows do not measurably")
        print("  change the student net. Do not claim that they do.")


if __name__ == "__main__":
    main()
