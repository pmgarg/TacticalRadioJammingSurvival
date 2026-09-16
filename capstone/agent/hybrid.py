"""
The hybrid student — DESIGN v2.0 §9.4, finally built.

Two heads that fail in different ways, arbitrated on purpose:

  * the INDUCED RULES encode physics that transfers. "The hot channel moved between scans"
    is true of every sweeping jammer that ever existed, including ones never in the corpus.
  * the NET encodes joint, non-linear structure the rules provably cannot express. Policy
    induction reached 1.00 held-out precision on barrage / spot / sweep and was REJECTED for
    fading (0.60), node_loss (0.36) and hidden_term (0.41).

That split is not a curiosity, it is the architecture: the attacker half of this problem is a
conjunction of thresholds and the benign half is not. So rules lead where they were accepted,
the net leads everywhere else, and the two disagreeing is itself an uncertainty signal.

The payoff this is built for: `sweep` is the held-out family. The net scores 0% on it because
it has never seen one. The sweep RULE does not care -- it was derived from the physics, not
from sweep training rows.

Every claim above is measured by tests/test_hybrid_generalisation.py.
"""
from __future__ import annotations

import os

from .api import Agent, Context, Decision, CAUSES, normalise, uniform_belief
from .policy_table import PLAYBOOK
from .student import StudentAgent

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POLICY = os.path.join(HERE, "..", "data", "policy_v3.json")

# A rule is trusted to LEAD only if induction accepted it at this held-out precision. Below
# it the rule still votes, but it cannot override the net.
LEAD_PRECISION = 0.95

# Features this project has MEASURED to be unreliable. A rule that leans on one of them does
# not get to overrule a trained net, whatever precision induction reported for it -- that
# precision was measured on sampled states, not on the closed-loop states the agent actually
# reaches.
#
#   rssi_pdr_corr  (S1) -- FIDELITY.md: largely unmeasurable. RSSI is only observed on frames
#                          that DECODE, and deeply faded frames do not, so survivor bias
#                          flattens the correlation. The design originally called this "the
#                          primary false-positive defence"; it is not.
#   loss_load_corr (S10) -- RESULTS_LLM.md section 3: measures 0.00 on EVERY family. It carries
#                          no information on this radio at all.
#
# This is not a patch for a symptom. `reactive_decorrelated_shadow_loss` and
# `congestion_selfshadow_zero_silent` share the clause `tx_shadow_loss_delta >= 0.3` and are
# separated ONLY by rssi_pdr_corr -- so in closed loop the reactive rule was capturing
# congestion episodes the net had already classified correctly (congestion 100% -> 75%).
# Induction found a discriminator the project had already proved does not exist.
UNRELIABLE_FEATURES = {"rssi_pdr_corr", "loss_load_corr"}


class HybridAgent:
    """Induced rules + distilled net, arbitrated. This is what would actually fly."""

    name = "hybrid"

    def __init__(self, bundle_path: str | None = None, policy_path: str | None = None,
                 lead_precision: float = LEAD_PRECISION):
        self.net = StudentAgent(bundle_path)
        self.lead_precision = lead_precision
        self.rs = None
        self.rule_prec: dict[str, float] = {}
        self.demoted: dict[str, list] = {}
        try:
            from harness.policy import RuleSet
            self.rs = RuleSet.load(policy_path or DEFAULT_POLICY)
            for r in getattr(self.rs, "rules", []):
                rid = getattr(r, "id", "")
                prec = float(getattr(r, "precision", 0.0) or 0.0)
                feats = {c[0] for c in getattr(r, "clauses", []) if len(c) == 3}
                leans = feats & UNRELIABLE_FEATURES
                if leans:
                    self.demoted[rid] = sorted(leans)
                    prec = 0.0          # may still vote, may never lead
                self.rule_prec[rid] = prec
            # Most-specific-first: when two rules match, the one with more clauses is the
            # narrower claim and should win. RuleSet.suggest() takes the first match in file
            # order, which is arbitrary.
            self.rs.rules.sort(key=lambda r: -len(getattr(r, "clauses", [])))
        except Exception as e:                       # a missing policy file is not fatal
            self._policy_error = str(e)
        self.reset()

    def reset(self) -> None:
        self.net.reset()
        self.n_rule_led = 0
        self.n_net_led = 0
        self.n_disagree = 0

    # ------------------------------------------------------------------ #
    def _rule_hit(self, features: list[float]) -> tuple[str | None, str, float]:
        if self.rs is None:
            return None, "", 0.0
        try:
            hit = self.rs.suggest(features)
        except Exception:
            return None, "", 0.0
        hyp = hit.get("hypothesis")
        if hyp not in CAUSES:
            return None, "", 0.0
        rid = hit.get("rule", "?")
        return hyp, rid, self.rule_prec.get(rid, 0.0)

    def decide(self, features: list[float], ctx: Context) -> Decision:
        net_d = self.net.decide(features, ctx)
        hyp, rule_id, prec = self._rule_hit(features)

        if hyp is None:
            self.n_net_led += 1
            return net_d

        agree = (net_d.declared == hyp)
        if not agree:
            self.n_disagree += 1

        if prec >= self.lead_precision:
            # The rule leads. It is auditable, it was validated on held-out data, and --
            # the point of the whole exercise -- it still fires on a family the net has
            # never been trained on.
            self.n_rule_led += 1
            belief = dict(net_d.belief)
            # keep the net's shape but move the mass, so the trace still shows what the
            # net thought: an examiner can see the disagreement rather than a rewritten past
            for c in belief:
                belief[c] *= 0.25
            belief[hyp] = belief.get(hyp, 0.0) + 0.75
            belief = normalise(belief)

            call, why = PLAYBOOK.get(hyp, ("no_op", "no playbook entry"))
            if call not in ctx.available:
                call = "no_op"
            return Decision(belief, call, self.net._args(call, ctx),
                            confidence=max(net_d.confidence, 0.9), declared=hyp,
                            why=(f"rule {rule_id} (held-out precision {prec:.2f}) -> {hyp}; {why}"
                                 + ("" if agree else f" [net said {net_d.declared}]")),
                            expect="PDR recovers to >=0.8 of baseline within 3 s")

        # A low-precision rule does not get to override a trained net; it only breaks ties.
        self.n_net_led += 1
        if net_d.declared is None:
            return Decision(net_d.belief, net_d.call, net_d.args, net_d.confidence,
                            declared=hyp,
                            why=f"net abstained; low-precision rule {rule_id} suggests {hyp}")
        return net_d

    def stats(self) -> dict:
        return {"rule_led": self.n_rule_led, "net_led": self.n_net_led,
                "disagreements": self.n_disagree}
