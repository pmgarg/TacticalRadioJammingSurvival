"""G6 -- the teacher is an LLM, and it ran against ns-3.

WHY THIS GATE EXISTS
Every other gate in this project checks the agent. None of them checked WHO TAUGHT IT.
That omission is how the project drifted from its own brief without a single test going
red:

  * `run_llm.py --world` defaulted to `refsim`, so the flag that enforces DESIGN v2.0 9.2
    ("every headline teacher number comes from the ns-3 bridge") was opt-in. Teacher
    artefacts were produced against the fast approximation and nothing objected.
  * `make llm-stub` wrote its fabricated decisions to the same path as the real teacher
    corpus, so a CI run could silently replace a real teacher with a stub.
  * `agent/teacher.py` (OracleLabeller) is handed the true cause and plays a fixed
    playbook. Its docstring says it is not the teacher; nothing enforced that, and the
    shipped student was distilled from it.

A gate suite that verifies the measurement but not the provenance of the supervision can
be fully green while the thesis is unproven. So: this checks the supervision.

    python3 tests/test_teacher_provenance.py --golden ../data/llm_golden_ns3_full.json
"""
from __future__ import annotations

import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

PASS, FAIL = [], []


def check(ok: bool, name: str, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="../data/llm_golden_ns3_full.json")
    ap.add_argument("--traces", default="../data/traces/llm_ns3_full")
    ap.add_argument("--min-episodes", type=int, default=24)
    ap.add_argument("--min-families", type=int, default=8)
    ap.add_argument("--min-real-calls", type=int, default=100)
    a = ap.parse_args()

    print("=" * 70)
    print("G6  TEACHER PROVENANCE -- the teacher is an LLM and it ran against ns-3")
    print("=" * 70)

    if not os.path.exists(a.golden):
        check(False, "a teacher golden-set artefact exists", a.golden)
        summarise(); return
    g = json.load(open(a.golden))
    eps = g.get("episodes", [])
    s = g.get("summary", {})

    # --- 1. the world -------------------------------------------------------------
    worlds = {e.get("world", "unrecorded") for e in eps}
    check(worlds == {"ns3"}, "every teacher episode ran against ns-3 (DESIGN 9.2)",
          f"worlds seen: {sorted(worlds)}")

    # --- 2. the model -------------------------------------------------------------
    kinds = {e.get("provider_kind", "unrecorded") for e in eps}
    provs = {e.get("provider", "unrecorded") for e in eps}
    check("stub" not in kinds and "stub" not in provs,
          "no stubbed decisions in the teacher corpus", f"providers: {sorted(provs)}")

    # --- 3. it actually called the model ------------------------------------------
    calls = int(s.get("llm_calls", 0))
    check(calls >= a.min_real_calls,
          f"the teacher made >= {a.min_real_calls} real model calls",
          f"{calls} calls over {len(eps)} episodes")

    # --- 4. coverage --------------------------------------------------------------
    fams = set(s.get("by_family", {}))
    check(len(fams) >= a.min_families,
          f"the teacher saw >= {a.min_families} scenario families", f"{sorted(fams)}")
    check(len(eps) >= a.min_episodes,
          f"the teacher ran >= {a.min_episodes} episodes", f"{len(eps)} episodes")

    # --- 5. it was not fed the answer ---------------------------------------------
    src = open(os.path.join(HERE, "agent", "llm_teacher.py")).read()
    leaked = [w for w in ("true_cause", "sc.truth", "ground_truth") if w in src]
    check(not leaked, "the LLM teacher never receives ground truth (DESIGN 9.1)",
          f"found {leaked}" if leaked else "no ground-truth symbol in llm_teacher.py")

    # --- 6. nothing that flies imports the oracle ---------------------------------
    offenders = []
    for mod in ("agent/student.py", "agent/hybrid.py", "agent/rule_agent.py",
                "agent/controller.py", "agent/policy_table.py", "percept/features.py"):
        p = os.path.join(HERE, mod)
        if os.path.exists(p) and "from .teacher import" in open(p).read():
            offenders.append(mod)
    check(not offenders, "no deployed module imports OracleLabeller (DESIGN 9.3)",
          f"offenders: {offenders}" if offenders else "clean")

    # --- 7. the traces on disk match the claim ------------------------------------
    files = sorted(glob.glob(os.path.join(a.traces, "*.jsonl")))
    ticks = real = 0
    for f in files:
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ticks += 1
            real += (not d.get("cache_hit")) and d.get("parse_mode") == "clean"
    check(real >= a.min_real_calls,
          "the trace files hold that many genuine uncached teacher decisions",
          f"{ticks} ticks, {real} uncached and cleanly parsed, in {len(files)} files")

    summarise()


def summarise() -> None:
    print("=" * 70)
    print(f"TEACHER PROVENANCE: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
