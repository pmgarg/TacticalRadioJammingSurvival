"""Executable verification of every edge in the architecture DAG.

`docs/ARCHITECTURE.md §1.1` claims fifteen links work. A table in a document is an
assertion; this file is the evidence. Run it and every claim is either PASS with a
measured number, or FAIL with the reason.

    python3 harness/verify_links.py            # everything except the LLM edges
    python3 harness/verify_links.py --llm      # include the live LLM edges (slow)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

RESULTS: list[tuple[str, str, bool, str]] = []


def check(lid: str, what: str):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                ok, detail = fn(*a, **k)
            except Exception as e:                     # noqa: BLE001
                ok, detail = False, f"{type(e).__name__}: {e}"
            RESULTS.append((lid, what, ok, f"{detail}  [{time.time()-t0:.1f}s]"))
            print(f"  {'PASS' if ok else 'FAIL'}  {lid:<4} {what:<46} {detail}")
            return ok
        return run
    return deco


# --------------------------------------------------------------------- L1/L3
@check("L1", "corpus -> scenarios build")
def l1():
    from scenario.corpus import build
    splits = build(60, seed0=777)
    keys = ("train", "val", "test")
    n = sum(len(splits[k]) for k in keys)
    fams = {s.family for k in keys for s in splits[k]}
    return n > 0, f"{n} scenarios, {len(fams)} families"


@check("L3", "refsim -> 56-feature vector")
def l3():
    from scenario.library import scn_barrage
    from sim.refsim import RefSim
    from percept.features import FeatureExtractor, FEATURE_NAMES, N_FEATURES
    sc = scn_barrage()
    sim = RefSim(sc)
    ex = FeatureExtractor(sc.n_channels, dt=sim.dt)
    f = None
    for _ in range(400):
        obs = sim.step()
        f = ex.update(obs)
    assert len(f) == N_FEATURES == len(FEATURE_NAMES)
    bad = [FEATURE_NAMES[i] for i, v in enumerate(f) if not (-1.0001 <= v <= 1.0001)]
    return not bad, f"{len(f)} features, all in [-1,1]" if not bad else f"out of range: {bad}"


# --------------------------------------------------------------------- L5/L8/L9/L10/L13
def _episode(agent_factory, family: str, seed: int = 3):
    from scenario.corpus import make
    from sim.refsim import RefSim
    from agent.controller import Controller
    from verify.verifier import score_episode
    import yaml
    contract = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
    costcfg = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))
    sc = make(family, seed)
    ag = agent_factory(sc)
    log = Controller(RefSim(sc), sc, ag, contract).run()
    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable}
    return log, score_episode(log.to_dict(), truth, costcfg, contract, sc.family)


@check("L5", "oracle teacher -> controller -> refsim")
def l5():
    from agent.teacher import TeacherAgent
    hits = 0
    fams = ["barrage", "spot", "fading", "node_loss", "congestion"]
    for fam in fams:
        _, s = _episode(lambda sc: TeacherAgent(sc.truth.cause, sc.truth.recoverable), fam)
        hits += int(s.classification_ok)
    return hits == len(fams), f"{hits}/{len(fams)} correct"


@check("L8", "student -> controller -> refsim")
def l8():
    from agent.student import StudentAgent
    b = os.path.join(HERE, "..", "data", "student_FINAL", "student_bundle.json")
    if not os.path.exists(b):
        cand = glob.glob(os.path.join(HERE, "..", "data", "student*", "student_bundle.json"))
        if not cand:
            return False, "no student bundle on disk"
        b = cand[0]
    stu = StudentAgent(b)
    fams = ["barrage", "spot", "fading", "node_loss", "congestion", "hidden_term"]
    hits = sum(int(_episode(lambda sc: stu, f)[1].classification_ok) for f in fams)
    size = os.path.getsize(b) / 1024
    return hits >= len(fams) * 0.6, f"{hits}/{len(fams)} correct, bundle {size:.1f} KB"


@check("L9", "safety envelope (action masking + failsafe)")
def l9():
    import subprocess
    r = subprocess.run([sys.executable, "tests/test_agent_safety.py"],
                       cwd=HERE, capture_output=True, text=True, timeout=900)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if "SAFETY SUITE" in l]
    return r.returncode == 0, (lines[-1].strip()[:90] if lines else "no summary line")


@check("L13", "episode -> verifier -> scored metrics")
def l13():
    from agent.teacher import TeacherAgent
    log, s = _episode(lambda sc: TeacherAgent(sc.truth.cause, sc.truth.recoverable), "fading")
    ok = (s.declared_cause is not None) and isinstance(s.expected_cost, float)
    return ok, (f"declared={s.declared_cause} cost={s.expected_cost:.3f} "
                f"fp={s.false_positive} fp_acted={s.false_positive_acted}")


# --------------------------------------------------------------------- harness
@check("H1", "tool registry generated from contract")
def h1():
    from harness.registry import ToolRegistry
    r = ToolRegistry()
    from agent.api import ALL_CALLS
    missing = set(ALL_CALLS) - set(r.tools)
    return not missing, f"{len(r.tools)} tools, contract==api" if not missing else str(missing)


@check("H2", "parser: clean / repaired / failed all handled")
def h2():
    from harness.parser import parse_decision
    cases = [('{"call":"no_op"}', "clean"),
             ('```json\n{"call":"no_op",}\n```', "repaired"),
             ("prose only, no object", "failed"),
             ('text {"call":"hop_channel","args":{"channel":1}} trailing {oops', "clean")]
    bad = [(s[:24], parse_decision(s).mode, want) for s, want in cases
           if parse_decision(s).mode != want]
    return not bad, "4/4 modes correct" if not bad else str(bad)


@check("H3", "trace store: append -> reload -> summarise")
def h3():
    from harness.trace import TraceStore, DecisionRecord
    p = os.path.join(HERE, "..", "data", "traces", "_selftest.jsonl")
    if os.path.exists(p):
        os.remove(p)
    with TraceStore(p) as ts:
        for i in range(3):
            ts.append(DecisionRecord(episode="e1", step=i, t_sim=float(i),
                                     agent="t", call="no_op", parse_mode="clean",
                                     latency_s=0.5))
    s = TraceStore.summary(p)
    os.remove(p)
    return s["records"] == 3 and s["episodes"] == 1, json.dumps(s)


# --------------------------------------------------------------------- LLM edges
@check("L7", "gateway -> Claude CLI (live)")
def l7():
    from gateway.provider import ClaudeCliProvider
    p = ClaudeCliProvider(cache=False)
    out = p.complete("Reply with exactly this and nothing else: HARNESS_OK")
    return "HARNESS_OK" in out, f"{p.stats()['mean_wall_s']}s round trip"


@check("L6", "HarnessAgent diagnoses live refsim episodes")
def l6(families):
    from harness.loop import HarnessAgent
    from harness.trace import TraceStore
    from gateway.provider import ClaudeCliProvider
    p = ClaudeCliProvider()
    tp = os.path.join(HERE, "..", "data", "traces", "verify_links.jsonl")
    ts = TraceStore(tp)
    ag = HarnessAgent(provider=p, trace=ts)
    hits, seen = 0, []
    for fam in families:
        ag.set_episode(f"verify_{fam}", fam)
        _, s = _episode(lambda sc: ag, fam)
        ok = bool(s.classification_ok)
        hits += int(ok)
        seen.append(f"{fam}{'+' if ok else '-'}")
    ts.close()
    st = ag.stats()
    return hits >= len(families) * 0.5, (
        f"{hits}/{len(families)} [{' '.join(seen)}] "
        f"calls={st['calls']} cache={st['cache_hits']} "
        f"parse_fail={st['parse_failures']} repairs={st['repairs']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="include live LLM edges (slow)")
    ap.add_argument("--families", default="barrage,fading,reactive,node_loss")
    a = ap.parse_args()

    print("\nARCHITECTURE DAG — link verification\n" + "=" * 78)
    print("\n[world / percept]")
    l1(); l3()
    print("\n[agents / controller / verifier]")
    l5(); l8(); l9(); l13()
    print("\n[harness]")
    h1(); h2(); h3()
    if a.llm:
        print("\n[LLM edges — live]")
        l7(); l6([f for f in a.families.split(",") if f])
    else:
        print("\n[LLM edges skipped — pass --llm to include]")

    print("\n" + "=" * 78)
    npass = sum(1 for *_, ok, _ in RESULTS if ok)
    print(f"{npass}/{len(RESULTS)} links verified")
    out = os.path.join(HERE, "..", "data", "link_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump([{"link": l, "what": w, "pass": ok, "detail": d} for l, w, ok, d in RESULTS],
              open(out, "w"), indent=1)
    print(f"written: {os.path.relpath(out, HERE)}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
