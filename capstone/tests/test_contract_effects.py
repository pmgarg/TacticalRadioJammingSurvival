#!/usr/bin/env python3
"""
G1 -- every call in the contract must DO something.

AUDIT F2: eight of the agent's calls reached ns-3 and fell through the if/else chain with
no effect whatsoever, while `agent_contract.json` advertised them with costs and a list of
what they "separate". `change_tdma_slot` was inert because TDMA was never enabled -- and
three of the eight playbook entries depend on it. The brief's whole premise is "run the
cheapest distinguishing test", so a contract that lists tests the world ignores is not a
small documentation bug: it means the agent is paying budget for nothing and the evidence
it thinks it bought never arrives.

The test is deliberately dumb and therefore hard to fool: run an episode in which the agent
issues ONE call, repeatedly, and compare the resulting telemetry against an identical
episode in which it issues `no_op`. If the two are bit-identical, the call did nothing.

This is the reference-sim half of the gate; it needs no ns-3 and runs in seconds.
`tests/test_world_contract_ns3.md` describes the same assertions against the live bridge,
which is where the ns-3-side no-ops (AUDIT F2) actually live.

    python3 tests/test_contract_effects.py
    python3 tests/test_contract_effects.py --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.api import Context, Decision, CAUSES
from agent.controller import Controller
from scenario.library import FAMILIES
from sim.refsim import RefSim

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))

# Calls whose effect is on the AGENT's own bookkeeping rather than the channel, so a
# telemetry diff is the wrong instrument. They are checked by budget movement instead.
BOOKKEEPING = {"spectrum_scan", "silent_listen", "declare_link_lost", "no_op"}

# Calls known to be inert, with the reason, so the gate stays green for REGRESSIONS while
# still printing the open problem every run. Removing an entry here is how it gets fixed.
KNOWN_OPEN = {
    "neighbor_probe":
        "implemented in refsim (refsim.neighbor_probe) but its ANSWER does not reach the "
        "agent: the percept layer has no probe-result feature, so probing a live-but-"
        "jammed peer is indistinguishable from not probing at all. Fixing it means adding "
        "a feature, which changes N_FEATURES and invalidates every trained bundle -- do it "
        "with the next retrain, not as a bolt-on.",
    "set_tx_power":
        "inert by physics, not by omission: raising OUR transmit power cannot improve the "
        "frames WE receive, and pdr/recovery are measured on the forward link. But "
        "PLAYBOOK['fading'] = set_tx_power, so the recommended remedy for the single most "
        "important family provably cannot move the metric it is judged on. DECISION NEEDED: "
        "either score the reverse link too (refsim already models it, refsim.py:401), or "
        "change the fading action to reroute/move.",
}

ARGS = {
    "hop_channel": {"channel": 2}, "channel_hop_probe": {"channel": 2},
    "set_tx_power": {"dbm": 4}, "change_tdma_slot": {"slot": 3},
    "reroute": {"via": None}, "load_test": {"factor": 0.25},
    "move": {"dx": 60.0, "dy": 0.0, "dz": 0.0},
    "mobility_test": {"dx": 60.0, "dy": 0.0, "dz": 0.0},
    "fallback_to_lora": {}, "spectrum_scan": {"dwell_ms": 20},
    "silent_listen": {"duration_ms": 200}, "neighbor_probe": {"peer": "P0"},
    "listen_test": {}, "transmit_probe": {},
}


class OneCallAgent:
    """Issues exactly one call, every decision, unconditionally."""

    def __init__(self, call: str, pre_scan: bool = False):
        self.call = call
        self.pre_scan = pre_scan
        self.name = f"only:{call}"
        self.scanned = False

    def reset(self):
        self.scanned = False

    def decide(self, features, ctx: Context) -> Decision:
        b = {c: 1.0 / len(CAUSES) for c in CAUSES}
        # hop_channel is masked until a fresh scan exists, so some calls can only be
        # exercised after one. That scan is itself an intervention, so it is applied to
        # BOTH arms of the comparison or not at all -- never to the probe alone.
        if self.pre_scan and not self.scanned:
            self.scanned = True
            if "spectrum_scan" in ctx.available:
                return Decision(b, "spectrum_scan", {"dwell_ms": 20}, 0.5,
                                why="probe: unmask the hop rules")
        call = self.call if self.call in ctx.available else "no_op"
        return Decision(b, call, dict(ARGS.get(call, {})), confidence=0.5,
                        why=f"contract-effect probe: {self.call}")


def fingerprint(log, ctrl) -> tuple:
    """What the WORLD did, and what the AGENT saw.

    A recovery action has to move the world. A diagnostic test does not -- its whole job
    is to change the agent's OBSERVATION, so judging `neighbor_probe` by packet delivery
    would call a working test inert. Both streams are captured and the caller picks.
    """
    world = (tuple(tuple(x) for x in log.pdr_series),
             round(log.packets_lost, 3), log.recovery_t, log.declared_lost_t)
    seen = tuple(tuple(round(v, 6) for v in r["features"]) for r in ctrl.records)
    return world, seen


def run(family: str, call: str, seed: int, pre_scan: bool = False):
    sc = FAMILIES[family](seed=seed)
    sim = RefSim(sc)
    ctrl = Controller(sim, sc, OneCallAgent(call, pre_scan), CONTRACT)
    log = ctrl.run()
    return log, ctrl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="spot,barrage,fading,congestion,hidden_term,node_loss,reactive")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    diagnose = set(CONTRACT["diagnose"])
    calls = list(CONTRACT["diagnose"]) + [c for c in CONTRACT["act"] if c != "no_op"]
    families = [f.strip() for f in a.families.split(",") if f.strip()]

    print("=" * 78)
    print(f"G1  CONTRACT-EFFECT GATE  (refsim, seed={a.seed})")
    print(f"    families: {', '.join(families)}")
    print("    A call is INERT only if it changes nothing in ANY family. Many calls")
    print("    legitimately do nothing in most families -- change_tdma_slot only bites")
    print("    where there is contention -- so a single-family probe over-reports.")
    print("=" * 78)

    works_in: dict[str, list[str]] = {c: [] for c in calls}
    tried_in: dict[str, list[str]] = {c: [] for c in calls}

    # Two arms per family: without a preparatory scan, and with one. The control is run
    # under the SAME arm as the probe, so the scan never counts as the probe's effect.
    for fam in families:
        controls = {}
        for ps in (False, True):
            cl, cc = run(fam, "no_op", a.seed, ps)
            controls[ps] = (fingerprint(cl, cc), cc, cl)

        for call in calls:
            hit = False
            for ps in (False, True):
                (base_world, base_seen), base_ctrl, base_log = controls[ps]
                log, ctrl = run(fam, call, a.seed, ps)
                issued = (sum(1 for x in log.tests_run if x["test"] == call)
                          + sum(1 for x in log.actions if x["fn"] == call))
                if issued == 0:
                    continue
                if fam not in tried_in[call]:
                    tried_in[call].append(fam)
                world, seen = fingerprint(log, ctrl)
                moved = (world != base_world) or (call in diagnose and seen != base_seen)
                if moved or (call in BOOKKEEPING and ctrl.used != base_ctrl.used):
                    hit = True
                    if a.verbose:
                        print(f"    {call:20s} {fam:12s} pre_scan={ps} packets_lost "
                              f"{base_log.packets_lost} -> {log.packets_lost}")
            if hit and fam not in works_in[call]:
                works_in[call].append(fam)

    inert, gated, effective = [], [], []
    for call in calls:
        if not tried_in[call]:
            gated.append(call)
            print(f"  [----] {call:20s} never available in any family tested (mask)")
        elif works_in[call]:
            effective.append(call)
            print(f"  [OK  ] {call:20s} effective in {len(works_in[call])}/"
                  f"{len(tried_in[call])} families: {','.join(works_in[call])}")
        else:
            inert.append(call)
            print(f"  [INERT] {call:19s} issued in {','.join(tried_in[call])} "
                  f"-- changed NOTHING anywhere")

    print("\n" + "=" * 78)
    print(f"effective: {len(effective)}   inert: {len(inert)}   never-available: {len(gated)}")
    if gated:
        print(f"  never available: {gated}  (the mask may be correct; check the budget rules)")
    open_known = [c for c in inert if c in KNOWN_OPEN]
    regressions = [c for c in inert if c not in KNOWN_OPEN]
    if open_known:
        print("\n  KNOWN-OPEN (documented, not a regression):")
        for c in open_known:
            print(f"    {c}: {KNOWN_OPEN[c]}")
    inert = regressions
    if inert:
        print(f"\n  INERT CALLS: {inert}")
        print("  Per DESIGN v2.0 A13 a call stays in the contract only if the world")
        print("  implements it. Implement these or delete them from agent_contract.json.")
        print("  An agent that spends budget on a call the world ignores is buying nothing.")
    print("=" * 78)
    sys.exit(1 if inert else 0)


if __name__ == "__main__":
    main()
