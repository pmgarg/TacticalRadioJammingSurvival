#!/usr/bin/env python3
"""
Does the hybrid actually recover the held-out family?

The claim under test, from DESIGN v2.0 section 9.4: the induced rules encode physics that
TRANSFERS, so they should fire on `sweep` -- the family deliberately excluded from every
training row -- while the distilled net, which has never seen one, scores zero.

If that holds, the hybrid is not an engineering preference, it is the measured answer to
"does this generalise to an attack we did not anticipate". If it does not hold, say so: a
hybrid that adds nothing is complexity for its own sake and should be deleted.

    python3 tests/test_hybrid_generalisation.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.baseline import BaselineAgent
from agent.controller import Controller
from agent.hybrid import HybridAgent
from agent.student import StudentAgent
from scenario.schema import load_scenario
from scenario.split import split_of
from sim.refsim import RefSim
from verify.verifier import score_episode

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def run(sc, agent):
    ctrl = Controller(RefSim(sc), sc, agent, CONTRACT)
    log = ctrl.run()
    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable, "notes": ""}
    return score_episode(log.to_dict(), truth, COSTCFG, CONTRACT, sc.family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bundle", default="../data/student_v8_clean/student_bundle.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    d = os.path.join(a.corpus, a.split)
    files = sorted(f for f in os.listdir(d) if f.endswith(".yaml"))
    if a.limit:
        by_fam = {}
        keep = []
        for f in files:
            fam = f.rsplit("_", 1)[0]
            by_fam.setdefault(fam, 0)
            if by_fam[fam] < a.limit:
                by_fam[fam] += 1
                keep.append(f)
        files = keep

    agents = {
        "baseline": lambda: BaselineAgent(),
        "net":      lambda: StudentAgent(a.bundle),
        "hybrid":   lambda: HybridAgent(a.bundle),
    }
    res = {k: {} for k in agents}
    print("=" * 74)
    print(f"HYBRID GENERALISATION -- {len(files)} episodes from {a.corpus}/{a.split}")
    print("=" * 74)

    for name, mk in agents.items():
        for f in files:
            sc = load_scenario(os.path.join(d, f))
            s = run(sc, mk())
            r = res[name].setdefault(sc.family, [0, 0, 0.0])
            r[0] += int(s.classification_ok)
            r[1] += 1
            r[2] += s.expected_cost

    fams = sorted({f for v in res.values() for f in v})
    held = split_of("sweep_1")            # the held-out family name, from the split itself
    print(f"\n{'family':14s} {'n':>4s}  {'baseline':>9s} {'net':>9s} {'hybrid':>9s}   delta(hybrid-net)")
    tot = {k: [0, 0, 0.0] for k in agents}
    for fam in fams:
        line = f"{fam:14s}"
        n = res["net"].get(fam, [0, 0, 0])[1]
        line += f" {n:4d} "
        vals = {}
        for k in agents:
            ok, nn, c = res[k].get(fam, [0, 0, 0.0])
            vals[k] = ok / nn if nn else 0.0
            tot[k][0] += ok
            tot[k][1] += nn
            tot[k][2] += c
            line += f" {vals[k]:8.0%}"
        dl = vals["hybrid"] - vals["net"]
        mark = "  <== HELD OUT" if fam == "sweep" else ""
        line += f"   {dl:+7.0%}{mark}"
        print(line)

    print()
    for k in agents:
        ok, nn, c = tot[k]
        print(f"  {k:9s} overall {ok}/{nn} = {ok/max(1,nn):.1%}   expected cost {c/max(1,nn):.3f}")

    sw_net = res["net"].get("sweep", [0, 0, 0])
    sw_hyb = res["hybrid"].get("sweep", [0, 0, 0])
    print("\n" + "=" * 74)
    if sw_net[1] == 0:
        print("VERDICT: no sweep episodes in this split -- nothing to conclude.")
        sys.exit(0)
    net_acc = sw_net[0] / sw_net[1]
    hyb_acc = sw_hyb[0] / sw_hyb[1]
    print(f"HELD-OUT FAMILY 'sweep':  net {net_acc:.0%}   hybrid {hyb_acc:.0%}")
    if hyb_acc > net_acc:
        print("The induced rule fires on a family the net never trained on. The rules carry")
        print("physics that transfers; the net carries structure that does not. That is the")
        print("whole argument for the hybrid, and it is now a number.")
    else:
        print("The hybrid did NOT beat the net on the held-out family. Then it is complexity")
        print("without payoff on this axis -- say so, and justify it on auditability alone.")
    print("=" * 74)


if __name__ == "__main__":
    main()
