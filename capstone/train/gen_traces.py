"""
Generate the teacher trace corpus (design §9.1-9.2), with rejection sampling.

Runs the privileged teacher over every scenario and records (features -> cause, call)
at each decision epoch. Episodes the VERIFIER says the teacher got wrong are DISCARDED:
we distil a filtered teacher, which is measurably better than the raw one.
"""
from __future__ import annotations

import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from agent.controller import Controller
from agent.teacher import OracleLabeller
from scenario.schema import load_scenario
from sim.refsim import RefSim
from verify.verifier import score_episode

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def rollout(sc, agent=None):
    sim = RefSim(sc)
    ag = agent or OracleLabeller(sc.truth.cause, sc.truth.recoverable)
    ctrl = Controller(sim, sc, ag, CONTRACT)
    log = ctrl.run()
    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable}
    s = score_episode(log.to_dict(), truth, COSTCFG, CONTRACT, sc.family)
    return ctrl.records, s, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--corpus", default="../data/corpus")
    ap.add_argument("--out", default="../data/traces")
    ap.add_argument("--no-reject", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.corpus, a.split, "*.yaml")))
    os.makedirs(a.out, exist_ok=True)
    kept = dropped = 0
    rows = []
    for fp in files:
        sc = load_scenario(fp)
        recs, s, log = rollout(sc)
        good = s.classification_ok and (s.refusal_ok is not False) and not s.false_positive_acted
        if (not a.no_reject) and not good:
            dropped += 1
            continue
        kept += 1
        for r in recs:
            rows.append({"scenario": sc.name, "family": sc.family,
                         "cause": sc.truth.cause, "t": r["t"],
                         "features": r["features"], "call": r["call"],
                         "args": r["args"], "available": r["available"]})
    out = os.path.join(a.out, f"{a.split}.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{a.split:5s}: episodes kept={kept:4d} dropped_by_rejection={dropped:3d}  "
          f"decision rows={len(rows):6d}  -> {out}")


if __name__ == "__main__":
    main()
