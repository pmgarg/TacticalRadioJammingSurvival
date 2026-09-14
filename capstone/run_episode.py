#!/usr/bin/env python3
"""
Run one scenario (or the whole corpus) end to end and score it.

    python3 run_episode.py --scenario ../scenarios/spot_single_channel.yaml -v
    python3 run_episode.py --corpus --agent baseline
    python3 run_episode.py --corpus --seeds 5 --out ../data/runs

Records-first (design §10.5): every run writes a complete raw record to disk BEFORE
any predicate executes; scoring is a separate pure pass over those records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from scenario.schema import load_scenario, Scenario
from scenario.library import FAMILIES, build_corpus
from sim.refsim import RefSim
from agent.controller import Controller
from agent.baseline import BaselineAgent
from agent.teacher import TeacherAgent
from agent.student import StudentAgent
from verify.verifier import score_episode, aggregate, gates

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))

# "rules" (agent.rule_agent.RuleAgent) is intentionally not registered here yet --
# its default policy file (data/policy_v3.json) was never committed; see
# personal/CODE_ALIGNMENT_REVIEW.md. Register it once harness/induce.py has produced
# and saved that file.
STUDENT_BUNDLE = os.path.join(HERE, "..", "data", "student_final", "student_bundle.json")
AGENTS = {
    # zero-arg factories only; "oracle" needs the scenario's own truth and is built
    # per-episode in run_one() below, since a privileged agent cannot be a stateless
    # singleton the way the others are.
    "baseline": BaselineAgent,
    "student": lambda: StudentAgent(STUDENT_BUNDLE),
    "oracle": None,
}


def run_one(sc: Scenario, agent_name: str = "baseline", verbose: bool = False,
            out_dir: str | None = None) -> tuple[dict, dict]:
    sim = RefSim(sc)
    if agent_name == "oracle":
        agent = TeacherAgent(sc.truth.cause, sc.truth.recoverable)
    else:
        agent = AGENTS[agent_name]()
    ctrl = Controller(sim, sc, agent, CONTRACT, verbose=verbose)
    log = ctrl.run()
    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable, "notes": sc.truth.notes}
    record = {
        "run_id": f"{sc.name}#{sc.seed}#{agent_name}",
        "task_id": sc.name, "agent": agent_name, "family": sc.family,
        "config": {"scenario": sc.name, "seed": sc.seed,
                   "n_channels": sc.n_channels, "channel": sc.channel,
                   "nodes": len(sc.nodes), "duration_s": sc.duration_s},
        "contract_version": CONTRACT["contract_version"],
        "episode_log": log.to_dict(),
        "ground_truth": truth,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, record["run_id"].replace("/", "_") + ".json"), "w") as fh:
            json.dump(record, fh, indent=2)
    return record, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", help="path to a scenario YAML")
    ap.add_argument("--corpus", action="store_true", help="run the whole corpus")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--agent", default="baseline", choices=list(AGENTS))
    ap.add_argument("--out", default=None, help="write raw records here")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    scens: list[Scenario] = []
    if a.scenario:
        scens = [load_scenario(a.scenario)]
    elif a.corpus:
        scens = build_corpus(a.seeds)
    else:
        ap.error("give --scenario or --corpus")

    scores = []
    for sc in scens:
        rec, truth = run_one(sc, a.agent, a.verbose, a.out)
        s = score_episode(rec["episode_log"], truth, COSTCFG, CONTRACT, sc.family)
        scores.append(s)
        if a.verbose or len(scens) == 1:
            log = rec["episode_log"]
            print(f"\n=== {sc.name}  (truth: {sc.truth.cause}, "
                  f"recoverable={sc.truth.recoverable}) ===")
            print(f"  onset detected t={log['attack_onset_t']}")
            for row in log["classification_trace"][:8]:
                print(f"    t={row['t']:5.1f}  {row['top']:<12s} p={row['p']:.2f}")
            if len(log["classification_trace"]) > 8:
                print(f"    ... {len(log['classification_trace'])-8} more decisions")
            print("  tests:  ", [f"{x['test']}@{x['t']}" for x in log["tests_run"]] or "none")
            print("  actions:", [f"{x['fn']}@{x['t']}" for x in log["actions"]] or "none")
            print(f"  declared={s.declared_cause}  ok={s.classification_ok}  "
                  f"det_lat={s.detection_latency_s}  recov={s.recovery_time_s}  "
                  f"survived={s.survived}  refusal_ok={s.refusal_ok}")
            for x in log["actions"][:3]:
                if x.get("why"):
                    print(f"    why[{x['fn']}]: {x['why']}")

    agg = aggregate(scores)
    print("\n" + "=" * 74)
    print(f"AGENT={a.agent}   episodes={agg['n']}")
    print(f"  classification accuracy : {agg['classification_acc']:.2%}")
    print(f"  expected cost (lower=better): {agg['expected_cost']:.2f}")
    print(f"  detection latency (median) : {agg['detection_latency_median_s']}")
    print(f"  censoring rate             : {agg['censoring_rate']:.2%}")
    print(f"  FP fading->jamming (belief): {agg['fp_fading_belief']}")
    print(f"  FP fading->jamming (ACTED) : {agg['fp_fading_acted']}")
    print(f"  recovery time (median)     : {agg['recovery_median_s']}")
    print(f"  survival rate              : {agg['survival_rate']:.2%}")
    print(f"  actions / hops (mean)      : {agg['actions_mean']:.2f} / {agg['hops_mean']:.2f}")
    print(f"  refusal gate pass          : {agg['refusal_gate_pass']}")
    print("\n  per family:")
    for fam, v in agg["by_family"].items():
        print(f"    {fam:13s} n={v['n']:<3d} acc={v['classification_acc']:.0%}  "
              f"cost={v['expected_cost']:.2f}  surv={v['survival_rate']:.0%}  "
              f"actions={v['actions_mean']:.1f}")
    g = gates(agg, {"fp_max": 0.05})
    print("\n  GATES:")
    for k, v in g.items():
        print(f"    {k:22s} value={v['value']}  target={v['target']}  "
              f"{'PASS' if v['pass'] else 'FAIL'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
