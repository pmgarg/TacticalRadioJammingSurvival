"""Rebuild a teacher golden-set summary from the trace files on disk.

WHY
`run_llm.py` writes its summary only after the LAST episode returns. A single episode that
hangs -- an ns-3 child that never exits, a model call that never comes back -- therefore
destroys the summary for every episode that DID finish. That happened: 31 of 32 episodes
completed, 320 real uncached model calls, and none of it was written out.

Model calls cost real money and ~30 minutes of wall clock. Losing them to one stuck
subprocess is not acceptable, and re-running is the wrong fix. The traces are written
incrementally, per episode, as the run proceeds -- so they are the durable record and the
summary is derivable from them. This derives it.

Also the honest way to report a partial run: it counts what is actually on disk rather
than what was requested.

    python3 harness/rescore_traces.py --traces ../data/traces/llm_ns3_full \\
        --world ns3 --out ../data/llm_golden_ns3_full.json
"""
from __future__ import annotations

import argparse, collections, glob, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--world", default="ns3")
    ap.add_argument("--provider", default="claude")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None,
                    help="the run's stdout log. Its per-episode verdicts came from "
                         "verify/verifier.py, so they are AUTHORITATIVE and are preferred "
                         "over the heuristic below wherever a line exists.")
    a = ap.parse_args()

    # Verdicts the verifier already computed during the run, if the log survived.
    # Recomputing them here from traces alone cannot match verify/verifier.py -- the
    # traces hold beliefs but not actions or recovery_t, and the verifier's anchoring
    # rule needs both. Two scorers would be two truths, so prefer the real one.
    logged = {}
    if a.log and os.path.exists(a.log):
        import re
        pat = re.compile(r"^\s+(ok |MISS|ERR )\s+(\S+)\s+seed=(\d+)\s+declared=(\S+)")
        for line in open(a.log):
            m = pat.match(line.rstrip())
            if m:
                mark, fam, seed, dec = m.groups()
                logged[(fam, seed)] = (mark.strip() == "ok",
                                       None if dec == "None" else dec)
        print(f"using {len(logged)} verifier-scored verdicts from {a.log}")

    files = sorted(glob.glob(os.path.join(a.traces, "*.jsonl")))
    if not files:
        print(f"no traces under {a.traces}"); sys.exit(2)

    episodes, by_family = [], collections.defaultdict(lambda: {"n": 0, "ok": 0})
    calls = ticks = clean = 0

    for fp in files:
        name = os.path.basename(fp)[: -len(".jsonl")]
        fam, _, seed = name.rpartition("_")
        rows = []
        for line in open(fp):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            continue
        ticks += len(rows)
        ep_calls = sum(1 for r in rows if not r.get("cache_hit"))
        clean += sum(1 for r in rows if r.get("parse_mode") == "clean")
        calls += ep_calls

        # The teacher's verdict for the episode: its last CONFIDENT committed belief.
        # Mirrors verify/verifier.py -- the claim is what it committed to, not its final
        # drift, and an unparseable or empty reply is not a claim.
        claim, conf = None, 0.0
        for r in rows:
            b = r.get("belief") or {}
            if not b or r.get("parse_mode") == "failed":
                continue
            top = max(b, key=b.get)
            c = float(r.get("confidence") or 0.0)
            if c >= 0.75:
                claim, conf = top, c
        scored_by = "verifier(run log)"
        if (fam, seed) in logged:
            correct, claim = logged[(fam, seed)]
        else:
            correct = (claim == fam)
            scored_by = "heuristic(trace only)"
        by_family[fam]["n"] += 1
        by_family[fam]["ok"] += int(correct)
        episodes.append({"family": fam, "seed": seed, "ok": True, "correct": correct,
                         "declared": claim, "confidence": conf,
                         "world": a.world, "provider_kind": a.provider,
                         "provider": "claude-cli", "calls": ep_calls,
                         "ticks": len(rows), "scored_by": scored_by})

    n = len(episodes)
    summary = {"episodes": n, "completed": n,
               "accuracy": round(sum(e["correct"] for e in episodes) / n, 3) if n else None,
               "llm_calls": calls, "ticks": ticks, "clean_parses": clean,
               "world": a.world, "provider_kind": a.provider,
               "rebuilt_from_traces": True,
               "scored_by": sorted({e["scored_by"] for e in episodes}),
               "by_family": {k: {"n": v["n"], "acc": round(v["ok"] / v["n"], 3)}
                             for k, v in sorted(by_family.items())}}
    json.dump({"summary": summary, "episodes": episodes}, open(a.out, "w"), indent=1)

    print(f"rebuilt {n} episodes from {len(files)} trace files -> {a.out}")
    print(f"  teacher decisions: {ticks}   uncached model calls: {calls}   "
          f"clean parses: {clean}")
    print(f"  accuracy: {summary['accuracy']}")
    for k, v in summary["by_family"].items():
        print(f"    {k:<12} n={v['n']:<3} acc={v['acc']:.2f}")


if __name__ == "__main__":
    main()
