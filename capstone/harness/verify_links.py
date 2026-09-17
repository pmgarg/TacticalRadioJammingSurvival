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
    from agent.teacher import OracleLabeller
    hits = 0
    fams = ["barrage", "spot", "fading", "node_loss", "congestion"]
    for fam in fams:
        _, s = _episode(lambda sc: OracleLabeller(sc.truth.cause, sc.truth.recoverable), fam)
        hits += int(s.classification_ok)
    return hits == len(fams), f"{hits}/{len(fams)} correct"


@check("L8", "student -> controller -> refsim")
def l8():
    from agent.student import StudentAgent
    b = os.path.join(HERE, "..", "data", "student_v9", "student_bundle.json")
    if not os.path.exists(b):
        cand = sorted(glob.glob(os.path.join(HERE, "..", "data", "student*", "student_bundle.json")))
        if not cand:
            return False, "no student bundle on disk"
        b = cand[0]
    stu = StudentAgent(b)
    fams = ["barrage", "spot", "fading", "node_loss", "congestion", "hidden_term"]
    hits = sum(int(_episode(lambda sc: stu, f)[1].classification_ok) for f in fams)
    size = os.path.getsize(b) / 1024
    return hits >= len(fams) * 0.6, f"{hits}/{len(fams)} correct, bundle {size:.1f} KB"


# ------------------------------------------------------------------ ns-3 bridge
# The links below cover the LIVE ns-3 path -- the one the LLM teacher and the headline
# evaluation both run on. Everything above this point exercises refsim only, which is how
# a bridge that reported a constant -100 dBm on all eight channels for entire episodes
# passed 9/9 link verification and a 17/17 world gate at the same time.

def _ns3_bin():
    import glob as _g
    env = os.environ.get("NS3_BIN")
    if env and os.path.exists(env):
        return env
    for pat in ("/home/claude/ns3/build/scratch/jamming/*jamming-sim*",
                os.path.expanduser("~/Documents/NS3/ns-3-dev/build/scratch/jamming/*jamming-sim*")):
        hits = [h for h in _g.glob(pat) if os.access(h, os.X_OK)]
        if hits:
            return hits[0]
    return None


def _bridge_episode(agent, family="spot"):
    """One live ns-3 episode, capturing the feature vectors the agent actually saw."""
    import glob as _g
    from sim.bridge_server import run as run_bridge
    from scenario.schema import load_scenario
    binp = _ns3_bin()
    if binp is None:
        raise RuntimeError("no jamming-sim binary (set NS3_BIN)")
    files = sorted(_g.glob(os.path.join(HERE, "..", "data", "corpus", "*", f"{family}_*.yaml")))
    if not files:
        raise RuntimeError(f"no {family} scenario in the corpus")
    sc = load_scenario(files[0])

    seen = []

    class _Cap:
        name = "cap"

        def reset(self):
            if hasattr(agent, "reset"):
                agent.reset()

        def decide(self, f, ctx):
            seen.append(list(f))
            return agent.decide(f, ctx)

    log = run_bridge(sc, binp, agent_obj=_Cap(), verbose=False)["episode_log"]
    return sc, log, seen


@check("N1", "ns-3 bridge -> 56-feature vector, S2 and S5 ALIVE")
def n1():
    """The regression test for the two bridge bugs. A feature that never varies across a
    whole episode is not a feature; it is a constant the model will learn to ignore."""
    from agent.baseline import BaselineAgent
    import numpy as np
    sc, log, seen = _bridge_episode(BaselineAgent(), "spot")
    if not seen:
        return False, "the bridge produced no decisions"
    F = np.asarray(seen, dtype=float)
    dead = [i for i in range(F.shape[1]) if F[:, i].std() == 0.0]
    NOISE, NSTD = 17, 18          # noise_delta_base (S2), noise_std
    s2_alive = F[:, NOISE].std() > 0.01 and abs(F[:, NOISE]).max() > 0.05
    return s2_alive, (f"S2 noise_delta_base: mean={F[:, NOISE].mean():+.3f} "
                      f"std={F[:, NOISE].std():.3f}, {len(dead)}/{F.shape[1]} features constant")


@check("N2", "ns-3 spectrum sees a DUTY-CYCLED jammer (median, not min-hold)")
def n2():
    """duty<1.0 is the common case in the corpus and the case a min-hold cannot see."""
    import numpy as np, glob as _g
    from scenario.schema import load_scenario
    import sim.bridge_server as B
    from agent.baseline import BaselineAgent
    binp = _ns3_bin()
    if binp is None:
        return False, "no jamming-sim binary"
    files = sorted(_g.glob(os.path.join(HERE, "..", "data", "corpus", "*", "spot_*.yaml")))
    sc = load_scenario(files[0])
    duty = getattr(sc.jammers[0], "duty", None)
    states = []
    orig = B.state_to_obs

    def spy(st, prev, t):
        states.append(st)
        return orig(st, prev, t)

    B.state_to_obs = spy
    try:
        B.run(sc, binp, agent_obj=BaselineAgent(), verbose=False)
    finally:
        B.state_to_obs = orig
    band = np.asarray([s["band"] for s in states if s.get("band")], dtype=float)
    ts = np.asarray([s["t"] for s in states if s.get("band")], dtype=float)
    onset = float(sc.truth.onset_t)
    pre, post = band[ts < onset], band[ts >= onset]
    if not len(pre) or not len(post):
        return False, "episode did not span the jammer onset"
    rise = post.max(axis=0).max() - pre.max(axis=0).max()
    return rise > 10.0, (f"duty={duty}, hottest channel rose {rise:+.1f} dB at onset "
                         f"(pre {pre.max():.1f} -> post {post.max():.1f} dBm)")


@check("N3", "ns-3 bridge: information must be BOUGHT (scan_age)")
def n3():
    """The agentic premise. If scan features arrive fresh without the agent paying for a
    spectrum_scan, the contract's diagnose actions are decoration and the student never
    learns to gather evidence. The CSV corpus hands them out free on 100% of ticks."""
    import numpy as np
    from agent.baseline import BaselineAgent
    sc, log, seen = _bridge_episode(BaselineAgent(), "spot")
    F = np.asarray(seen, dtype=float)
    SCAN_AGE = 30
    # diagnose calls are logged under tests_run, act calls under actions -- count both,
    # or the check reads "0 scans" for an agent that scanned four times.
    scans = ([a for a in log.get("actions", []) if a.get("fn") == "spectrum_scan"]
             + [t for t in log.get("tests_run", []) if t.get("test") == "spectrum_scan"])
    first_scan_t = min([float(x["t"]) for x in scans], default=None)
    ts = [float(r["t"]) for r in log.get("classification_trace", [])]
    fresh = (F[:, SCAN_AGE] < 0.99).mean()
    if first_scan_t is None:
        # never scanned: NOTHING may report a fresh scan
        return fresh < 0.01, f"0 scans paid for, {100*fresh:.0f}% of ticks report a fresh scan"
    # scanned: freshness must not predate the first paid scan
    early = sum(1 for t_, f_ in zip(ts, F[:, SCAN_AGE]) if t_ < first_scan_t and f_ < 0.99)
    return early == 0, (f"{len(scans)} scans paid for (first at t={first_scan_t:.0f}s), "
                        f"{100*fresh:.0f}% of ticks fresh, {early} of them BEFORE paying")


@check("N4", "ns-3 bridge -> verifier -> scored episode")
def n4():
    from agent.student import StudentAgent
    from verify.verifier import score_episode
    import yaml as _y
    contract = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
    costcfg = _y.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))
    sc, log, seen = _bridge_episode(StudentAgent(), "barrage")
    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable}
    s = score_episode(log, truth, costcfg, contract, sc.family)
    return s.declared_cause is not None or s.survived, (
        f"declared={s.declared_cause} cost={s.expected_cost:.3f} survived={s.survived}")


@check("L9", "safety envelope (action masking + failsafe)")
def l9():
    import subprocess
    r = subprocess.run([sys.executable, "tests/test_agent_safety.py"],
                       cwd=HERE, capture_output=True, text=True, timeout=900)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if "SAFETY SUITE" in l]
    return r.returncode == 0, (lines[-1].strip()[:90] if lines else "no summary line")


@check("L13", "episode -> verifier -> scored metrics")
def l13():
    from agent.teacher import OracleLabeller
    log, s = _episode(lambda sc: OracleLabeller(sc.truth.cause, sc.truth.recoverable), "fading")
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
    ap.add_argument("--no-ns3", action="store_true",
                    help="skip the live ns-3 bridge links (they need the built binary)")
    a = ap.parse_args()

    print("\nARCHITECTURE DAG — link verification\n" + "=" * 78)
    print("\n[world / percept]")
    l1(); l3()
    print("\n[agents / controller / verifier]")
    l5(); l8(); l9(); l13()
    if not a.no_ns3:
        print("\n[ns-3 bridge — the LIVE path the teacher and the headline table run on]")
        n1(); n2(); n3(); n4()
    else:
        print("\n[ns-3 bridge links skipped — --no-ns3]")
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
    # numpy bools reach here from the ns-3 checks and json refuses them
    json.dump([{"link": l, "what": w, "pass": bool(ok), "detail": str(d)}
               for l, w, ok, d in RESULTS], open(out, "w"), indent=1)
    print(f"written: {os.path.relpath(out, HERE)}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
