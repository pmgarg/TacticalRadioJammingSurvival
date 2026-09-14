"""An agent that runs ONLY the LLM-induced rule set.

Its purpose is measurement, not deployment: it answers "are the rules the LLM wrote
actually any good?" on the same corpus, through the same safety envelope, scored by the
same verifier as every other agent. A rule set that reads well and measures badly is a
story; this makes it a number.

Policy when no rule fires: spend the cheapest unspent diagnostic that could make one fire,
then abstain. It never guesses, so its errors are omissions rather than false positives --
which is the correct failure mode given the cost matrix.
"""
from __future__ import annotations

import os

from .api import Agent, Context, Decision, CAUSES, uniform_belief
from harness.policy import RuleSet

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "..", "data", "policy_v3.json")

# Cheapest-first ladder of tests to run when no rule matches. Ordered by contract cost.
LADDER = ["spectrum_scan", "silent_listen", "neighbor_probe", "load_test"]

RECOVERY = {
    "barrage": "fallback_to_lora", "spot": "hop_channel", "sweep": "hop_channel",
    "reactive": "change_tdma_slot", "fading": "set_tx_power",
    "node_loss": "reroute", "congestion": "change_tdma_slot",
    "hidden_term": "change_tdma_slot",
}


class RuleAgent:
    name = "rules"

    def __init__(self, policy_path: str = DEFAULT, confidence: float = 0.9):
        self.rs = RuleSet.load(policy_path)
        self.confidence = confidence
        self.reset()

    def reset(self) -> None:
        self.done: set[str] = set()

    def decide(self, features: list[float], ctx: Context) -> Decision:
        hit = self.rs.suggest(features)
        hyp = hit.get("hypothesis")
        if hyp in CAUSES:
            # A matched rule is a firm claim: put the mass on it so the verifier counts a
            # commitment, and take the matching recovery if the envelope allows it.
            b = {c: (1.0 - self.confidence) / (len(CAUSES) - 1) for c in CAUSES}
            b[hyp] = self.confidence
            call = hit.get("call") or RECOVERY.get(hyp, "no_op")
            if call not in ctx.available:
                call = "no_op"
            return Decision(b, call, {}, confidence=self.confidence,
                            why=f"rule {hit.get('rule')}: {hit.get('note','')[:120]}")

        for t in LADDER:
            if t in ctx.available and t not in self.done:
                self.done.add(t)
                return Decision(uniform_belief(), t, {}, 0.0,
                                why=f"no rule matched; buying evidence via {t}")
        return Decision(uniform_belief(), "no_op", {}, 0.0,
                        why="no rule matched and no unspent test; abstaining")
