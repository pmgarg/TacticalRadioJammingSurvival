"""The headline table (DESIGN v2.0 section 10.6): LLM teacher vs student vs baseline.

WHAT CHANGED, AND WHY IT MATTERS
This script used to compare `baseline / oracle* / student`, on refsim. Two problems, and
they are the reason requirement 5 of the brief was unmet:

  * The LLM teacher was not an arm. DESIGN 9.3 is explicit that the honest comparison is
    "LLM teacher vs student vs classical baseline, all three without ground truth", and
    10.6's headline table has a teacher column. It was never filled in, so the two gaps
    that ARE the thesis -- teacher->student small (distillation kept the reasoning) and
    student >> baseline (the learning was necessary) -- had no numbers.
  * It ran on refsim, the fast approximation, while DESIGN 9.2 requires ns-3.

Both are fixed. `--world ns3` is the default; the teacher is an arm; and the script also
reports the number the brief actually asks for -- TEACHER-STUDENT AGREEMENT, per decision
and per episode, on the same scenarios.

`oracle*` remains available behind --with-oracle as a CEILING. It is constructed with the
true cause; reporting its 100% as an agent result is the error DESIGN A5b forbids, so it
is starred everywhere and excluded by default.

    python3 train/evaluate.py --world ns3 --ns3 <binary> --limit-per-family 3
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from agent.baseline import BaselineAgent
from agent.teacher import OracleLabeller
from agent.student import StudentAgent
from agent.controller import Controller
from scenario.schema import load_scenario
from sim.refsim import RefSim
from verify.verifier import score_episode, aggregate, gates

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def _episode(sc, agent, world, ns3_bin):
    """One episode, in whichever world. The SAME controller and the SAME verifier score
    both, because two scorers would be two truths."""
    if world == "ns3":
        from sim.bridge_server import run as run_bridge
        return run_bridge(sc, ns3_bin, agent_obj=agent, verbose=False)["episode_log"]
    return Controller(RefSim(sc), sc, agent, CONTRACT).run().to_dict()


TEACHER_PROVIDER = "claude"      # set by main(); "stub" makes the arm free for dry runs


def _make_agent(which, sc, bundle):
    if which == "baseline":
        return BaselineAgent()
    if which == "oracle*":
        from agent.teacher import OracleLabeller
        return OracleLabeller(sc.truth.cause, sc.truth.recoverable)
    if which == "teacher":
        # The brief's teacher: an LLM, same 56 features as the student, no ground truth.
        from agent.llm_teacher import LlmOracleLabeller
        from gateway.provider import make_provider
        from harness.loop import HarnessAgent
        from harness.trace import TraceStore
        return HarnessAgent(provider=make_provider(TEACHER_PROVIDER),
                            trace=TraceStore(os.devnull), episode=sc.name, family=sc.family)
    return StudentAgent(bundle)


def run_split(files, which, bundle=None, world="ns3", ns3_bin=None, traces=None):
    """Score one arm, and keep every per-tick claim so arms can be compared decision by
    decision rather than only on their final verdicts."""
    scores, claims = [], {}
    stu = StudentAgent(bundle) if which == "student" else None
    for fp in files:
        sc = load_scenario(fp)
        ag = stu if which == "student" else _make_agent(which, sc, bundle)
        log = _episode(sc, ag, world, ns3_bin)
        truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
                 "recoverable": sc.truth.recoverable}
        scores.append(score_episode(log, truth, COSTCFG, CONTRACT, sc.family))
        # per-decision record, keyed by (scenario, t), for the agreement metric
        claims[sc.name] = {round(float(r["t"]), 1):
                           (None if r.get("abstained") else (r.get("declared") or r.get("top")))
                           for r in log.get("classification_trace", [])}
        if traces is not None:
            traces.setdefault(which, {})[sc.name] = log
    return scores, claims


def agreement(a: dict, b: dict) -> dict:
    """Teacher-student agreement: on the ticks where BOTH committed to a cause, how often
    is it the same cause? Ticks where either abstained are reported separately rather than
    counted as agreement -- two agents saying nothing is not two agents agreeing."""
    both = same = only_a = only_b = neither = 0
    per_scenario = {}
    for name, ta in a.items():
        tb = b.get(name, {})
        s_both = s_same = 0
        for t, ca in ta.items():
            cb = tb.get(t, "__absent__")
            if cb == "__absent__":
                continue
            if ca is not None and cb is not None:
                both += 1; s_both += 1
                if ca == cb:
                    same += 1; s_same += 1
            elif ca is None and cb is None:
                neither += 1
            elif ca is None:
                only_b += 1
            else:
                only_a += 1
        if s_both:
            per_scenario[name] = s_same / s_both
    return {"decisions_compared": both,
            "agree": same,
            "rate": (same / both) if both else None,
            "one_abstained": only_a + only_b,
            "both_abstained": neither,
            "per_scenario": per_scenario}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--world", default="ns3", choices=["refsim", "ns3"],
                    help="ns3 (DEFAULT): the authoritative simulator (DESIGN 9.2)")
    ap.add_argument("--ns3", default=None, help="path to the jamming-sim binary")
    ap.add_argument("--arms", default="baseline,teacher,student",
                    help="comma-separated. `teacher` bills real model calls.")
    ap.add_argument("--with-oracle", action="store_true",
                    help="add the privileged oracle as a CEILING (starred, not a competitor)")
    ap.add_argument("--limit-per-family", type=int, default=0,
                    help="cap scenarios per family; the teacher arm costs real calls")
    ap.add_argument("--teacher-provider", default="claude", choices=["claude", "stub"],
                    help="stub exercises the whole teacher arm and the agreement metric "
                         "without billing a single model call -- use it to prove the "
                         "pipeline before committing to a real run")
    ap.add_argument("--out", default="../data/headline_table.json")
    a = ap.parse_args()
    global TEACHER_PROVIDER
    TEACHER_PROVIDER = a.teacher_provider

    if a.world == "ns3" and not a.ns3:
        print("--world ns3 needs --ns3 <path to jamming-sim binary>"); sys.exit(2)

    files = sorted(glob.glob(os.path.join(a.corpus, a.split, "*.yaml")))
    if a.limit_per_family:
        seen, keep = {}, []
        for fp in files:
            fam = os.path.basename(fp).rsplit("_", 1)[0]
            if seen.get(fam, 0) < a.limit_per_family:
                seen[fam] = seen.get(fam, 0) + 1
                keep.append(fp)
        files = keep

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if a.with_oracle and "oracle*" not in arms:
        arms.append("oracle*")
    print(f"world={a.world}  arms={arms}  scenarios={len(files)} ({a.split})\n")

    rows, claims = {}, {}
    for which in arms:
        sc, cl = run_split(files, which, a.bundle, a.world, a.ns3)
        claims[which] = cl
        agg = aggregate(sc)
        rows[which] = (agg, gates(agg, {"fp_max": 0.05}), sc)
        print(f"  {which} done")
    print()

    hdr = f"{'metric':<34}" + "".join(f"{w:>12}" for w in rows)
    print(hdr); print("-" * len(hdr))
    def line(label, fn):
        print(f"{label:<34}" + "".join(f"{fn(rows[w][0]):>12}" for w in rows))
    line("classification accuracy", lambda a: f"{a['classification_acc']:.1%}")
    line("expected cost (lower better)", lambda a: f"{a['expected_cost']:.3f}")
    line("detection latency median (s)", lambda a: (f"{a['detection_latency_median_s']:.2f}"
          if a['detection_latency_median_s'] is not None else "-"))
    line("censoring rate", lambda a: f"{a['censoring_rate']:.1%}")
    line("FP fading->jam (belief)", lambda a: f"{a['fp_fading_belief']:.3f}" if a['fp_fading_belief'] is not None else "-")
    line("FP fading->jam (ACTED)", lambda a: f"{a['fp_fading_acted']:.3f}" if a['fp_fading_acted'] is not None else "-")
    line("recovery median (s)", lambda a: (f"{a['recovery_median_s']:.2f}"
          if a['recovery_median_s'] is not None else "-"))
    line("survival rate", lambda a: f"{a['survival_rate']:.1%}")
    line("actions consumed (mean)", lambda a: f"{a['actions_mean']:.2f}")
    line("channel hops (mean)", lambda a: f"{a['hops_mean']:.2f}")
    line("refusal gate pass", lambda a: f"{a['refusal_gate_pass']:.1%}" if a['refusal_gate_pass'] is not None else "-")
    print("-" * len(hdr))
    for w in rows:
        g = rows[w][1]
        print(f"{w:<12} FP gate={'PASS' if g['false_positive_gate']['pass'] else 'FAIL'}  "
              f"refusal gate={'PASS' if g['refusal_gate']['pass'] else 'FAIL'}")
    print("\nper-family classification accuracy")
    fams = sorted(next(iter(rows.values()))[0]["by_family"])
    print(f"{'family':<14}" + "".join(f"{w:>12}" for w in rows))
    for fam in fams:
        print(f"{fam:<14}" + "".join(
            f"{rows[w][0]['by_family'][fam]['classification_acc']:>11.0%} " for w in rows))
    from scenario.corpus import HELD_OUT_FAMILY
    print(f"\nSEEN families only (excluding the held-out '{HELD_OUT_FAMILY}')")
    print(f"{'metric':<34}" + "".join(f"{w:>12}" for w in rows))
    for label, key in (("classification accuracy", "acc"), ("expected cost", "cost"),
                       ("survival rate", "surv")):
        cells = []
        for w in rows:
            sub = [s for s in rows[w][2] if s.family != HELD_OUT_FAMILY]
            if key == "acc":
                v = f"{sum(x.classification_ok for x in sub)/len(sub):.1%}"
            elif key == "cost":
                v = f"{sum(x.expected_cost for x in sub)/len(sub):.3f}"
            else:
                v = f"{sum(x.survived for x in sub)/len(sub):.1%}"
            cells.append(f"{v:>12}")
        print(f"{label:<34}" + "".join(cells))
    json.dump({w: rows[w][0] for w in rows}, open("/tmp/eval.json", "w"), indent=1, default=str)


    # ---------------------------------------------------------------- agreement
    # REQUIREMENT 5: "test the student against ns-3 and match the results with the LLM
    # teacher's decisions". Episode verdicts alone do not answer that -- two agents can
    # both land on `spot` having reasoned completely differently. So compare them decision
    # by decision, on the same scenarios, at the same simulated times.
    if "teacher" in claims and "student" in claims:
        ag = agreement(claims["teacher"], claims["student"])
        print("\n" + "=" * 72)
        print("TEACHER -> STUDENT AGREEMENT  (DESIGN 10.6: the distillation gap)")
        print("=" * 72)
        if not ag["decisions_compared"]:
            print("  no overlapping committed decisions to compare")
        else:
            print(f"  decisions where BOTH committed to a cause : {ag['decisions_compared']}")
            print(f"  same cause                                : {ag['agree']}"
                  f"  ({ag['rate']:.1%})")
            print(f"  exactly one abstained                     : {ag['one_abstained']}")
            print(f"  both abstained                            : {ag['both_abstained']}")
            print("\n  Ticks where either side abstained are reported, not counted as")
            print("  agreement: two agents saying nothing is not two agents agreeing.")
            worst = sorted(ag["per_scenario"].items(), key=lambda kv: kv[1])[:6]
            if worst:
                print("\n  scenarios where the student diverges most from its teacher:")
                for name, r in worst:
                    print(f"    {name:<28} {r:.0%} agreement")
        json.dump(ag, open(a.out.replace(".json", "_agreement.json"), "w"), indent=1)

    payload = {"world": a.world, "split": a.split, "arms": arms,
               "n_scenarios": len(files),
               "metrics": {w: rows[w][0] for w in rows},
               "gates": {w: rows[w][1] for w in rows}}
    json.dump(payload, open(a.out, "w"), indent=1, default=str)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
