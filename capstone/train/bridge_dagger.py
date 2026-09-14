"""
Closed-loop DAgger THROUGH ns-3 (the authoritative simulator).

Offline DAgger relabelled states the student reached in the fast refsim. But live in
ns-3 the agent's own actions carry it somewhere else again -- which is exactly why live
accuracy (61%) trailed offline ns-3 feature accuracy (82.7%). This closes that loop:
roll the student out live, record the states it actually reaches in ns-3, ask the
privileged teacher what it would have done there, and add those to the training set.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.teacher import TeacherAgent
from agent.api import Context
from scenario.schema import load_scenario
from sim.bridge_server import run as bridge_run

ALL_CALLS = ["no_op", "spectrum_scan", "silent_listen", "neighbor_probe", "load_test",
             "mobility_test", "hop_channel", "set_tx_power", "change_tdma_slot",
             "reroute", "fallback_to_lora", "declare_link_lost"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus_all")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ns3", required=True)
    ap.add_argument("--bundle", default="../data/student_v2/student_bundle.json")
    ap.add_argument("--out", default="../data/traces_bridge/train.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.corpus, a.split, "*.yaml")))[:a.limit]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows, ok, bad = [], 0, 0
    t0 = time.time()
    for i, fp in enumerate(files):
        sc = load_scenario(fp)
        try:
            r = bridge_run(fp, a.ns3, "student", verbose=False, bundle=a.bundle)
        except Exception:
            bad += 1
            continue
        ok += 1
        # one teacher per episode, states fed in order (a fresh teacher per state
        # resets its "already tested?" counter and poisons the labels)
        teach = TeacherAgent(sc.truth.cause, sc.truth.recoverable)
        teach.reset()
        for rec in r["records"]:
            ctx = Context(t=rec["t"], channel=sc.channel, n_channels=sc.n_channels,
                          peers=[], budget={},
                          available=rec["available"] or ALL_CALLS,
                          last_scan=None, lora_available=sc.lora_available)
            d = teach.decide(rec["features"], ctx)
            rows.append({"scenario": sc.name, "family": sc.family,
                         "cause": sc.truth.cause, "t": rec["t"],
                         "features": rec["features"], "call": d.call, "args": {},
                         "available": rec["available"], "source": "bridge"})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)}  rows={len(rows)}  {el:.0f}s "
                  f"({el/(i+1):.1f}s/ep)")
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"live ns-3 rollouts ok={ok} failed={bad}  labelled rows={len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
