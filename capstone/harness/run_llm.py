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
    fam, seed, trace_dir, compare, world, provider_kind, ns3_bin = args
    sys.path.insert(0, HERE)
    import yaml
    from scenario.corpus import make
    from sim.refsim import RefSim
    from agent.controller import Controller
    from verify.verifier import score_episode
    from harness.loop import HarnessAgent
    from harness.trace import TraceStore
    from gateway.provider import make_provider

    contract = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
    costcfg = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))
    sc = make(fam, seed)
    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable}

    ts = TraceStore(os.path.join(trace_dir, f"{fam}_{seed}.jsonl"))
    ag = HarnessAgent(provider=make_provider(provider_kind), trace=ts,
                      episode=f"{fam}_{seed}", family=fam)
    t0 = time.time()
    try:
        if world == "ns3":
            # THE edge the brief grades: the teacher reasoning against the AUTHORITATIVE
            # simulator, not the fast approximation. A student trained only on refsim scores
            # 96% on refsim and 50% on ns-3 (RESULTS.md section 3); there is no reason to
            # think a teacher validated only on refsim is any safer.
            from sim.bridge_server import run as run_bridge

            class _L:
                def __init__(self, d):
                    self._d = d

                def to_dict(self):
                    return self._d

            log = _L(run_bridge(sc, ns3_bin, agent_obj=ag, verbose=False)["episode_log"])
        else:
            log = Controller(RefSim(sc), sc, ag, contract).run()
        s = score_episode(log.to_dict(), truth, costcfg, contract, sc.family)
        out = {"family": fam, "seed": seed, "ok": True,
               # PROVENANCE. DESIGN v2.0 9.2 requires every headline teacher number to come
               # from ns-3, and 9.1 requires a real model. Recording both on the episode is
               # what lets tests/test_teacher_provenance.py verify it later instead of
               # taking a filename's word for it.
               "world": world, "provider_kind": provider_kind,
               "correct": bool(s.classification_ok), "declared": s.declared_cause,
               "cost": float(s.expected_cost), "survived": bool(s.survived),
               "fp_acted": bool(s.false_positive_acted),
               "refusal_ok": s.refusal_ok,
               "wall_s": round(time.time() - t0, 1), **ag.stats()}
    except Exception as e:                                   # noqa: BLE001
        out = {"family": fam, "seed": seed, "ok": False,
               "world": world, "provider_kind": provider_kind,
               "error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 1)}
    finally:
        ts.close()

    if compare:
        from agent.teacher import OracleLabeller
        try:
            log2 = Controller(RefSim(make(fam, seed)), make(fam, seed),
                              OracleLabeller(sc.truth.cause, sc.truth.recoverable),
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
    ap.add_argument("--episode-timeout-s", type=int, default=1800,
                    help="give up on a single episode after this long; one stuck child "
                         "must not cost the whole run's summary")
    ap.add_argument("--trace-dir", default=os.path.join(HERE, "..", "data", "traces", "llm"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "llm_golden.json"))
    ap.add_argument("--compare-oracle", action="store_true", default=True)
    ap.add_argument("--world", default="ns3", choices=["refsim", "ns3"],
                    help="ns3 (DEFAULT) = the AUTHORITATIVE simulator via the live bridge. "
                         "DESIGN v2.0 9.2 requires every headline teacher number to come "
                         "from ns3. This used to default to refsim, which is how the "
                         "shipped teacher artefacts came to be measured against the fast "
                         "approximation: the flag enforcing the design was opt-in.")
    ap.add_argument("--provider", default="auto", choices=["auto", "claude", "stub"],
                    help="stub = deterministic offline stand-in, for testing the HARNESS "
                         "with no model. Never report a stub run as a teacher result.")
    ap.add_argument("--ns3", default=None, help="path to the jamming-sim binary (--world ns3)")
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
                jobs.append((f, seed, a.trace_dir, a.compare_oracle, a.world, a.provider, a.ns3))
                got += 1
            else:
                rejected += 1
            seed += 1
    print(f"{len(jobs)} episodes, {a.workers} workers "
          f"({rejected} candidate seeds rejected by scenario validation)\n")

    res, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        # A per-episode deadline. Without one, a single child that never returns -- an ns-3
        # subprocess that does not exit, a model call that never comes back -- blocks
        # as_completed forever and the summary is never written, destroying the record of
        # every episode that DID finish. That cost a 32-episode run with 320 real model
        # calls. Traces are written incrementally so nothing is truly lost (see
        # harness/rescore_traces.py), but the run should not need rescuing.
        pending = dict(futs)
        for fu in as_completed(futs, timeout=a.episode_timeout_s * len(jobs)):
            try:
                r = fu.result(timeout=a.episode_timeout_s)
            except Exception as e:                            # noqa: BLE001
                j = pending[fu]
                r = {"family": j[0], "seed": j[1], "ok": False, "world": j[4],
                     "provider_kind": j[5], "wall_s": round(time.time() - t0, 1),
                     "error": f"{type(e).__name__}: {e}"}
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
