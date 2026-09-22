"""Run the student live against ns-3 on the EXACT same ad-hoc seeds already tested with
the LLM teacher (harness/run_llm.py's scenario.corpus.make(family, seed) generation, not
the canonical data/corpus/*.yaml files train/evaluate.py globs -- a different scenario
universe, which is why evaluate.py --limit-per-family cannot be pointed at these seeds).

Costs nothing in LLM calls: the student is a local forward pass, and the teacher's side of
the comparison is read from an already-rescored trace summary (harness/rescore_traces.py),
not re-run.

    python3 train/compare_student_vs_teacher_seeds.py \\
        --ns3 <path to ns3.48-jamming-sim-default> \\
        --bundle ../data/student_v10/student_bundle.json \\
        --family spot --seeds 9100,9102,9103,9104,9105,9106,9108,9109,9110,9111 \\
        --teacher-json ../data/_archive/spot_batch_promptfix_rescored.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from agent.student import StudentAgent
from scenario.corpus import make
from sim.bridge_server import run as run_bridge
from verify.verifier import score_episode

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3", required=True)
    ap.add_argument("--bundle",
                    default=os.path.join(HERE, "..", "data", "student_v10", "student_bundle.json"))
    ap.add_argument("--family", default="spot")
    ap.add_argument("--seeds", required=True, help="comma-separated seeds")
    ap.add_argument("--teacher-json", default=None,
                    help="output of harness/rescore_traces.py for the teacher's own run "
                         "on these seeds -- read, never re-run")
    a = ap.parse_args()

    teacher = {}
    if a.teacher_json:
        d = json.load(open(a.teacher_json))
        for e in d["episodes"]:
            teacher[(e["family"], str(e["seed"]))] = e
        print(f"loaded {len(teacher)} teacher verdicts from {a.teacher_json}\n")

    seeds = [int(s) for s in a.seeds.split(",")]
    rows = []
    for seed in seeds:
        sc = make(a.family, seed)
        truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
                 "recoverable": sc.truth.recoverable}
        agent = StudentAgent(a.bundle)
        log = run_bridge(sc, a.ns3, agent_obj=agent, verbose=False)["episode_log"]
        s = score_episode(log, truth, COSTCFG, CONTRACT, sc.family)
        t = teacher.get((a.family, str(seed)))
        row = {"seed": seed, "true_cause": sc.truth.cause,
              "student_declared": s.declared_cause, "student_correct": bool(s.classification_ok),
              "teacher_declared": t.get("declared") if t else None,
              "teacher_correct": t.get("correct") if t else None}
        rows.append(row)
        s_mark = "ok  " if row["student_correct"] else "MISS"
        t_mark = ("ok  " if row["teacher_correct"] else "MISS") if t else "?   "
        print(f"  {a.family}_{seed:<6} true={sc.truth.cause:<10} "
              f"student={str(row['student_declared']):<10}{s_mark}  "
              f"teacher={str(row['teacher_declared']):<10}{t_mark}")

    n = len(rows)
    stu_ok = sum(r["student_correct"] for r in rows)
    print(f"\nstudent accuracy: {stu_ok}/{n} = {stu_ok / n:.2f}")
    if teacher:
        # teacher_correct is already False for an abstained (declared=None) episode, so
        # every scored row belongs in the denominator -- excluding None rows here would
        # silently drop the teacher's abstentions from its own accuracy.
        scored = [r for r in rows if r["teacher_correct"] is not None]
        t_ok = sum(r["teacher_correct"] for r in scored)
        print(f"teacher accuracy (from trace file, same seeds): {t_ok}/{len(scored)} "
              f"= {t_ok / max(1, len(scored)):.2f}")
        both = [r for r in rows if r["teacher_declared"] is not None and r["student_declared"] is not None]
        agree = sum(1 for r in both if r["student_declared"] == r["teacher_declared"])
        print(f"teacher-student agreement (both committed): {agree}/{len(both)} "
              f"= {agree / max(1, len(both)):.2f}")


if __name__ == "__main__":
    main()
