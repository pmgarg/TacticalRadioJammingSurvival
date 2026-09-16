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
from agent.teacher import TeacherAgent
from scenario.schema import load_scenario
from sim.refsim import RefSim
from verify.verifier import score_episode

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def rollout(sc, agent=None):
    sim = RefSim(sc)
    ag = agent or TeacherAgent(sc.truth.cause, sc.truth.recoverable)
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
    ap.add_argument("--agent", choices=["teacher", "llm"], default="teacher",
                    help="teacher = privileged oracle (default); llm = real "
                         "Claude-teacher demonstrations via the harness (P1)")
    ap.add_argument("--limit-per-family", type=int, default=0,
                    help="cap scenarios processed per family, 0 = unlimited "
                         "(the llm agent bills real API calls per decision -- "
                         "use this to bound cost on a first pass)")
    ap.add_argument("--audit-trace-dir", default=None,
                    help="llm agent only: write one DecisionRecord JSONL per "
                         "episode here (prompt/response/belief per step), same "
                         "one-file-per-episode convention dashboard/server.py "
                         "watches -- default data/traces/llm (its own default "
                         "--trace-dir), so a running gen_traces.py --agent llm "
                         "shows up live in the dashboard with no extra setup")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.corpus, a.split, "*.yaml")))
    os.makedirs(a.out, exist_ok=True)

    harness_agent = None
    audit_dir = None
    if a.agent == "llm":
        from gateway.provider import ClaudeCliProvider
        from harness.loop import HarnessAgent
        from harness.registry import ToolRegistry
        from harness.trace import TraceStore
        audit_dir = a.audit_trace_dir or os.path.join(
            HERE, "..", "data", "traces", "llm")
        os.makedirs(audit_dir, exist_ok=True)
        # One shared provider (cache persists across episodes within this
        # process) but a fresh TraceStore per episode -- set_episode() resets
        # the agent's own per-episode state (event gate, last belief, step
        # counter); the trace file is swapped alongside it so each episode
        # gets its own dashboard-visible file, matching harness/run_llm.py.
        harness_agent = HarnessAgent(provider=ClaudeCliProvider(), registry=ToolRegistry(),
                                     verbose=True)
        print(f"llm agent: caching via gateway/provider.py, audit traces -> {audit_dir}/*.jsonl")

    kept = dropped = 0
    rows = []
    per_family_seen: dict[str, int] = {}
    for fp in files:
        sc = load_scenario(fp)
        if a.limit_per_family:
            n = per_family_seen.get(sc.family, 0)
            if n >= a.limit_per_family:
                continue
            per_family_seen[sc.family] = n + 1
        if harness_agent is not None:
            harness_agent.set_episode(sc.name, sc.family)
            harness_agent.trace = TraceStore(os.path.join(audit_dir, f"{sc.name}.jsonl"))
        recs, s, log = rollout(sc, agent=harness_agent)
        if harness_agent is not None:
            harness_agent.trace.close()
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
    if harness_agent is not None:
        print(f"  provider stats: {harness_agent.stats()}")


if __name__ == "__main__":
    main()
