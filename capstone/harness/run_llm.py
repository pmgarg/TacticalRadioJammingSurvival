"""Run LLM-teacher episodes and bank the traces (the golden-set generator).

Episodes are independent, so they run in separate processes. Serially, a 60-second
episode at 1 Hz with ~50 s model latency is over half an hour; the event gate cuts the
call count by roughly 5x and process-level parallelism cuts the wall clock by the worker
count. Both matter: without them an LLM teacher is not affordable enough to teach anything.

    python3 harness/run_llm.py --per-family 2 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FAMILIES = ["barrage", "spot", "reactive", "sweep", "fading",
            "node_loss", "congestion", "hidden_term", "refusal"]


def run_one(args) -> dict:
    fam, seed, trace_dir, compare = args
    sys.path.insert(0, HERE)
    import yaml
    from scenario.corpus import make
    from sim.refsim import RefSim
    from agent.controller import Controller
    from verify.verifier import score_episode
    from harness.loop import HarnessAgent
    from harness.trace import TraceStore
    from gateway.provider import ClaudeCliProvider

    contract = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
    costcfg = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))
    sc = make(fam, seed)
    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable}

    ts = TraceStore(os.path.join(trace_dir, f"{fam}_{seed}.jsonl"))
    ag = HarnessAgent(provider=ClaudeCliProvider(), trace=ts,
                      episode=f"{fam}_{seed}", family=fam)
    t0 = time.time()
    try:
        log = Controller(RefSim(sc), sc, ag, contract).run()
        s = score_episode(log.to_dict(), truth, costcfg, contract, sc.family)
        out = {"family": fam, "seed": seed, "ok": True,
               "correct": bool(s.classification_ok), "declared": s.declared_cause,
               "cost": float(s.expected_cost), "survived": bool(s.survived),
               "fp_acted": bool(s.false_positive_acted),
               "refusal_ok": s.refusal_ok,
               "wall_s": round(time.time() - t0, 1), **ag.stats()}
    except Exception as e:                                   # noqa: BLE001
        out = {"family": fam, "seed": seed, "ok": False,
               "error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 1)}
    finally:
        ts.close()

    if compare:
        from agent.teacher import TeacherAgent
        try:
            log2 = Controller(RefSim(make(fam, seed)), make(fam, seed),
                              TeacherAgent(sc.truth.cause, sc.truth.recoverable),
                              contract).run()
            s2 = score_episode(log2.to_dict(), truth, costcfg, contract, sc.family)
            out["oracle_correct"] = bool(s2.classification_ok)
            out["oracle_cost"] = float(s2.expected_cost)
        except Exception:                                    # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=9100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--trace-dir", default=os.path.join(HERE, "..", "data", "traces", "llm"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "llm_golden.json"))
    ap.add_argument("--compare-oracle", action="store_true", default=True)
    a = ap.parse_args()

    os.makedirs(a.trace_dir, exist_ok=True)
    fams = [f for f in a.families.split(",") if f]
    # Only VALIDATED scenarios. corpus.build() filters with validate(); calling make()
    # directly does not, and a degenerate scenario (pre-onset delivery already collapsed,
    # so there is no clean baseline and the anomaly gate never fires) scores the agent as
    # having failed to diagnose something that never happened. Found the hard way.
    from scenario.corpus import make as _make, validate as _validate
    jobs, rejected = [], 0
    for f in fams:
        got, seed = 0, a.seed0
        while got < a.per_family and seed < a.seed0 + 5000:
            ok, _why = _validate(_make(f, seed))
            if ok:
                jobs.append((f, seed, a.trace_dir, a.compare_oracle))
                got += 1
            else:
                rejected += 1
            seed += 1
    print(f"{len(jobs)} episodes, {a.workers} workers "
          f"({rejected} candidate seeds rejected by scenario validation)\n")

    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fu in as_completed(futs):
            r = fu.result()
            res.append(r)
            mark = "ok " if r.get("correct") else "MISS" if r.get("ok") else "ERR "
            print(f"  {mark} {r['family']:<12} seed={r['seed']} "
                  f"declared={str(r.get('declared')):<12} "
                  f"calls={r.get('calls', '-'):>3} "
                  f"skip={r.get('skipped_stable', 0) + r.get('skipped_quiet', 0):>3} "
                  f"{r['wall_s']:>6.0f}s {r.get('error', '')[:60]}")

    ok = [r for r in res if r.get("ok")]
    acc = sum(r["correct"] for r in ok) / max(1, len(ok))
    orc = [r for r in ok if "oracle_correct" in r]
    agree = sum(1 for r in orc if r["correct"] == r["oracle_correct"]) / max(1, len(orc))
    by_fam = {}
    for r in ok:
        d = by_fam.setdefault(r["family"], [0, 0])
        d[0] += 1
        d[1] += int(r["correct"])
    summary = {
        "episodes": len(res), "completed": len(ok),
        "accuracy": round(acc, 3),
        "oracle_agreement": round(agree, 3),
        "fp_acted_total": sum(int(r.get("fp_acted", False)) for r in ok),
        "llm_calls": sum(r.get("calls", 0) for r in ok),
        "calls_skipped_by_gate": sum(r.get("skipped_stable", 0) + r.get("skipped_quiet", 0)
                                     for r in ok),
        "parse_failures": sum(r.get("parse_failures", 0) for r in ok),
        "repairs": sum(r.get("repairs", 0) for r in ok),
        "provider_errors": sum(r.get("provider_errors", 0) for r in ok),
        "wall_s": round(time.time() - t0, 1),
        "by_family": {k: {"n": v[0], "acc": round(v[1] / v[0], 3)}
                      for k, v in sorted(by_fam.items())},
    }
    json.dump({"summary": summary, "episodes": res}, open(a.out, "w"), indent=1)
    print("\n" + "=" * 66)
    print(json.dumps(summary, indent=1))
    print(f"\ntraces: {os.path.relpath(a.trace_dir, HERE)}")


if __name__ == "__main__":
    main()
