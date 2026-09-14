"""Three-way evaluation: baseline vs teacher vs student, on the held-out test split."""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from agent.baseline import BaselineAgent
from agent.teacher import TeacherAgent
from agent.student import StudentAgent
from agent.controller import Controller
from scenario.schema import load_scenario
from sim.refsim import RefSim
from verify.verifier import score_episode, aggregate, gates

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def run_split(files, which, bundle=None):
    scores = []
    stu = StudentAgent(bundle) if which == "student" else None
    for fp in files:
        sc = load_scenario(fp)
        if which == "baseline":
            ag = BaselineAgent()
        elif which == "teacher":
            ag = TeacherAgent(sc.truth.cause, sc.truth.recoverable)
        else:
            ag = stu
        log = Controller(RefSim(sc), sc, ag, CONTRACT).run()
        truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
                 "recoverable": sc.truth.recoverable}
        scores.append(score_episode(log.to_dict(), truth, COSTCFG, CONTRACT, sc.family))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/corpus")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bundle", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.corpus, a.split, "*.yaml")))
    print(f"evaluating {len(files)} held-out scenarios ({a.split})\n")
    rows = {}
    for which in ("baseline", "teacher", "student"):
        sc = run_split(files, which, a.bundle)
        agg = aggregate(sc)
        g = gates(agg, {"fp_max": 0.05})
        rows[which] = (agg, g, sc)

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
    fams = sorted(rows["teacher"][0]["by_family"])
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


if __name__ == "__main__":
    main()
