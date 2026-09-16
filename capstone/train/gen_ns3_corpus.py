"""
Run the whole scenario corpus through authoritative ns-3 and extract labelled
feature traces.

The student was previously trained only on refsim. ns-3 is the authoritative
simulator (design A2), so training on its features closes the last gap between the
distribution the model learns and the one it is graded on.
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario.schema import load_scenario
from sim.export_ns3 import to_config


def run_one(args):
    fp, ns3bin, workdir = args
    sc = load_scenario(fp)
    base = os.path.join(workdir, sc.name)
    cfg = base + ".cfg"
    open(cfg, "w").write(to_config(sc))
    # CHECK THE EXIT CODE. Only testing that the CSV exists let 23-of-30 crashed runs
    # through as "ok" -- a crashed ns-3 still leaves a partial file, and the whole
    # training corpus was silently built from truncated episodes.
    try:
        # AUDIT F3: --tdmaSlots was never passed, so the C++ default of 0 (CSMA only)
        # applied and change_tdma_slot() was a no-op in every corpus episode.
        r = subprocess.run([ns3bin, f"--config={cfg}", f"--out={base}",
                            "--tdmaSlots=4"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=300)
    except Exception:
        return (sc.name, None)
    if r.returncode != 0:
        return (sc.name, None)
    p = base + ".percept.csv"
    if not os.path.exists(p):
        return (sc.name, None)
    # a complete run must reach close to the configured duration
    try:
        with open(p) as fh:
            last = None
            for last in fh:
                pass
        t_last = float(last.split(",")[0])
        if t_last < 0.8 * sc.duration_s:
            return (sc.name, None)
    except Exception:
        return (sc.name, None)
    return (sc.name, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus_all")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ns3", required=True)
    ap.add_argument("--work", default="/tmp/ns3corpus")
    ap.add_argument("--out", default="../data/traces_ns3")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.corpus, a.split, "*.yaml")))
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    print(f"{a.split}: running {len(files)} scenarios through ns-3 on {a.jobs} jobs")

    done = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(run_one, [(f, a.ns3, a.work) for f in files])):
            done.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(files)}")

    from sim.ns3_adapter import features_from_ns3
    rows, ok, bad = [], 0, 0
    for fp in files:
        sc = load_scenario(fp)
        p = dict(done).get(sc.name)
        if not p or not os.path.exists(p):
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
        # label every post-onset sample with the true cause; subsample to ~1 Hz to
        # match the agent's decision cadence.
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
    print(f"  episodes ok={ok} failed={bad}  labelled rows={len(rows)} -> {out}")


if __name__ == "__main__":
    main()
