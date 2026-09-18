"""Which belief does the verifier score? Three cases that each broke the other's fix.

This file exists because the anchoring rule was changed twice, by two people, and each
change fixed one real defect while reintroducing another. The cases are cheap, synthetic
and independent of any trained model, so they pin the rule instead of the numbers.

  1. HEDGE-THEN-CORRECT  a one-tick guess that is abandoned immediately must not be the
     claim. Anchoring to literally the first action scored an agent that bumped TX power
     on a hunch, self-corrected, and then handled a reactive jammer correctly, as a miss.
  2. ACTED-AND-RECOVERED a correct diagnosis whose action WORKED must still be the claim.
     Requiring the belief to survive into the next tick breaks this: the agent says spot,
     hops, the hop works, the post-hop world no longer looks like spot -- and the agent is
     scored wrong precisely because it was right.
  3. ABSTAINED TICK      a refusal to classify is not a claim and must never be scored as
     one, in either direction.

Run: python3 tests/test_verifier_anchoring.py
"""
from __future__ import annotations

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from verify.verifier import score_episode

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))


def tick(t, cause, p, abstained=False):
    return {"t": t, "top": cause, "p": p, "declared": None if abstained else cause,
            "belief": {cause: p}, "unrecoverable": 0.0, "abstained": abstained}


def episode(trace, actions, recovery_t=None):
    return {"classification_trace": trace, "actions": actions,
            "recovery_t": recovery_t, "tests_run": [],
            "episode_id": "synthetic", "scenario": "synthetic", "agent": "synthetic"}


CASES = [
    ("hedge-then-correct: a one-tick guess must not anchor",
     episode([tick(5, "fading", 0.90), tick(6, "reactive", 0.95),
              tick(7, "reactive", 0.99), tick(8, "reactive", 0.99),
              tick(9, "reactive", 0.99)],
             [{"t": 5, "fn": "set_tx_power"}, {"t": 7, "fn": "change_tdma_slot"}],
             recovery_t=9.0),
     "reactive", "reactive"),

    ("acted-and-recovered: a correct call whose fix WORKED is still the claim",
     episode([tick(29, "spot", 0.99), tick(30, "spot", 1.00),
              tick(31, "sweep", 0.75), tick(32, "sweep", 0.75)],
             [{"t": 30, "fn": "hop_channel"}], recovery_t=32.0),
     "spot", "spot"),

    ("abstained tick is not a claim",
     episode([tick(10, "barrage", 0.59, abstained=True), tick(11, "fading", 0.99),
              tick(12, "fading", 0.99)],
             [{"t": 11, "fn": "set_tx_power"}], recovery_t=12.0),
     "fading", "fading"),

    ("investigate-and-revise: the claim is what the agent ENDED on",
     # barrage_20000, real. The teacher hedges `fading` and bumps TX power at t=7, a
     # spurious recovery_t=9 fires, and it then spends four spectrum_scans, concludes
     # `barrage` at p=0.85, holds it for seventeen ticks and declares the link lost --
     # the correct remedy for an unrecoverable barrage. Anchoring to the FIRST action
     # scored the whole episode `fading`. An agent that investigates and changes its mind
     # is doing the job.
     episode([tick(7, "fading", 0.80), tick(8, "fading", 0.80), tick(16, "fading", 0.80),
              tick(18, "barrage", 0.85), tick(22, "barrage", 0.85),
              tick(34, "barrage", 0.85)],
             [{"t": 7, "fn": "set_tx_power"}, {"t": 35, "fn": "declare_link_lost"}],
             recovery_t=9.0),
     "barrage", "barrage"),

    ("spurious mid-episode recovery must not truncate the fallback",
     # reactive_10192: recovery_t=14 is a gap between jammer bursts, not the end of the
     # incident. The agent is confused until 14, then correct at p=1.00 for 20 more
     # seconds with no recovery action. Truncating the fallback at recovery_t scored it
     # `hidden_term` and threw that away.
     episode([tick(12, "hidden_term", 0.99), tick(14, "hidden_term", 0.81),
              tick(26, "reactive", 1.00), tick(30, "reactive", 1.00),
              tick(34, "reactive", 1.00)],
             [{"t": 40, "fn": "declare_link_lost"}], recovery_t=14.0),
     "reactive", "reactive"),
]


def main():
    print("=" * 70)
    print("VERIFIER ANCHORING -- which belief gets scored")
    print("=" * 70)
    passed = failed = 0
    for name, log, true_cause, want in CASES:
        truth = {"cause": true_cause, "onset_t": 4.0, "recoverable": True, "notes": ""}
        got = score_episode(log, truth, COSTCFG, CONTRACT, true_cause).declared_cause
        ok = got == want
        passed += ok; failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         declared={got} want={want}")
    print("=" * 70)
    print(f"VERIFIER ANCHORING: {passed} passed, {failed} failed")
    print("=" * 70)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
