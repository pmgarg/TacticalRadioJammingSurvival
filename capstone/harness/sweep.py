"""Detection-rate sweep over the mutation corpus.

Produces the robustness CURVE the brief asks for: accuracy as a function of how far the
world has been deformed from anything the agent was trained on, broken out by axis and by
family. A single headline accuracy hides the cliff; this finds it.

    python3 harness/sweep.py --agent student --bundle ../data/student_FINAL/student_bundle.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import yaml                                          # noqa: E402
from scenario.schema import load_scenario            # noqa: E402
from sim.refsim import RefSim                        # noqa: E402
from agent.controller import Controller              # noqa: E402
from verify.verifier import score_episode            # noqa: E402

CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def make_agent(which: str, bundle: str | None, sc):
    if which == "student":
        from agent.student import StudentAgent
        return StudentAgent(bundle)
    if which == "baseline":
        from agent.baseline import BaselineAgent
        return BaselineAgent()
    if which == "rules":
        from agent.rule_agent import RuleAgent
        return RuleAgent()
    if which == "oracle":
        from agent.teacher import TeacherAgent
        return TeacherAgent(sc.truth.cause, sc.truth.recoverable)
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "..", "data", "mutations.jsonl"))
    ap.add_argument("--agent", default="student")
    ap.add_argument("--bundle", default=os.path.join(HERE, "..", "data", "student_FINAL",
                                                     "student_bundle.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "sweep_results.json"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    print(f"sweeping {len(rows)} mutations with agent={a.agent}\n")

    shared = None
    if a.agent == "student":
        from agent.student import StudentAgent
        shared = StudentAgent(a.bundle)

    recs, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        path = os.path.join(HERE, r["path"]) if not os.path.isabs(r["path"]) else r["path"]
        try:
            sc = load_scenario(path)
            ag = shared or make_agent(a.agent, a.bundle, sc)
            log = Controller(RefSim(sc), sc, ag, CONTRACT).run()
            truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
                     "recoverable": sc.truth.recoverable}
            s = score_episode(log.to_dict(), truth, COSTCFG, CONTRACT, sc.family)
            recs.append({**r, "correct": bool(s.classification_ok),
                         "declared": s.declared_cause,
                         "cost": float(s.expected_cost),
                         "fp": bool(s.false_positive),
                         "fp_acted": bool(s.false_positive_acted),
                         "survived": bool(s.survived)})
        except Exception as e:                        # noqa: BLE001
            recs.append({**r, "correct": False, "declared": None, "cost": None,
                         "error": f"{type(e).__name__}: {e}"})
        if i % 50 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  {el:.0f}s  "
                  f"acc={sum(x['correct'] for x in recs)/len(recs):.1%}")

    # ---- aggregate -------------------------------------------------------
    def agg(keyfn):
        d = defaultdict(lambda: [0, 0, 0])            # n, correct, fp_acted
        for x in recs:
            k = keyfn(x)
            d[k][0] += 1
            d[k][1] += int(x["correct"])
            d[k][2] += int(x.get("fp_acted", False))
        return {str(k): {"n": v[0], "acc": round(v[1] / v[0], 3),
                         "fp_acted": v[2]} for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}

    overall = sum(x["correct"] for x in recs) / max(1, len(recs))
    fp_acted = sum(int(x.get("fp_acted", False)) for x in recs)
    fading = [x for x in recs if x["family"] == "fading"]
    out = {
        "agent": a.agent, "n": len(recs),
        "overall_accuracy": round(overall, 3),
        "fp_acted_total": fp_acted,
        "fp_acted_rate_on_fading": round(
            sum(int(x.get("fp_acted", False)) for x in fading) / max(1, len(fading)), 3),
        "by_axis": agg(lambda x: x["axis"]),
        "by_family": agg(lambda x: x["family"]),
        "by_axis_magnitude": agg(lambda x: f"{x['axis']}@{x['magnitude']:+g}"),
        "errors": sum(1 for x in recs if "error" in x),
    }
    json.dump({"summary": out, "records": recs}, open(a.out, "w"), indent=1)

    print(f"\n{'='*66}\nDETECTION-RATE SWEEP — agent={a.agent}\n{'='*66}")
    print(f"overall accuracy under mutation : {overall:.1%}  (n={len(recs)})")
    print(f"false positives ACTED on (fading->hop): {fp_acted}"
          f"   rate on fading family: {out['fp_acted_rate_on_fading']:.1%}")
    print(f"\n{'axis':<12}{'n':>5}{'acc':>8}   robustness curve")
    for axis in sorted(out["by_axis"]):
        v = out["by_axis"][axis]
        pts = [(k.split("@")[1], m["acc"]) for k, m in out["by_axis_magnitude"].items()
               if k.startswith(axis + "@")]
        pts.sort(key=lambda p: float(p[0]))
        curve = "  ".join(f"{mg}:{ac:.0%}" for mg, ac in pts)
        print(f"{axis:<12}{v['n']:>5}{v['acc']:>8.1%}   {curve}")
    print(f"\n{'family':<14}{'n':>5}{'acc':>8}")
    for fam, v in out["by_family"].items():
        print(f"{fam:<14}{v['n']:>5}{v['acc']:>8.1%}")
    print(f"\nwritten: {os.path.relpath(a.out, HERE)}")


if __name__ == "__main__":
    main()
