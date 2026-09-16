"""Re-extract ns-3 feature traces from CACHED percept CSVs.

Re-running 540 ns-3 episodes to change a feature constant costs ~6 hours on two cores and
produces byte-identical CSVs, because the simulator is deterministic and the change is in
the PERCEPT layer, not the world. This reads the cached CSVs instead. Same output format
as gen_ns3_corpus.py; use that when the simulator or scenarios change, and this when only
feature code changes.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario.schema import load_scenario
from sim.ns3_adapter import features_from_ns3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus_all")
    ap.add_argument("--split", default="train")
    ap.add_argument("--work", default="/tmp/ns3v2", help="cached *.percept.csv dir")
    ap.add_argument("--out", default="../data/traces_ns3_v3")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.corpus, a.split, "*.yaml")))
    os.makedirs(a.out, exist_ok=True)
    rows, ok, bad = [], 0, 0
    for fp in files:
        sc = load_scenario(fp)
        p = os.path.join(a.work, sc.name + ".percept.csv")
        if not os.path.exists(p):
            bad += 1
            continue
        try:
            feats = features_from_ns3(p, sc.n_channels)
        except Exception:
            bad += 1
            continue
        if len(feats) < 20:
            bad += 1
            continue
        ok += 1
        t0 = sc.truth.onset_t
        for i, f in enumerate(feats):
            t = i * 0.1
            if t < t0 + 0.5 or (i % 10):
                continue
            rows.append({"scenario": sc.name, "family": sc.family,
                         "cause": sc.truth.cause, "t": round(t, 1),
                         "features": f, "call": "no_op", "args": {},
                         "available": [], "source": "ns3"})
    out = os.path.join(a.out, f"{a.split}.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  {a.split}: episodes ok={ok} missing/failed={bad}  rows={len(rows)} -> {out}")


if __name__ == "__main__":
    main()
