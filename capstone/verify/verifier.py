"""
The verifier (design §10) — the project's conscience.

Reads the raw run record (episode log + ground truth) and produces a scorecard.
It shares no code with the agent, it is the only component that sees ground truth,
and its verdicts are DETERMINISTIC PREDICATES over the tool output — never over the
agent's prose or its stated confidence (A18).

Metrics: detection latency (censored, not dropped), classification, the
false-positive rate on fading (belief-level AND acted-level), recovery time,
packets lost, actions consumed, survival, and the refusal gate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict

JAMMING = {"barrage", "spot", "reactive", "sweep"}


@dataclass
class EpisodeScore:
    episode_id: str
    scenario: str
    family: str
    agent: str
    true_cause: str
    declared_cause: str | None
    classification_ok: bool
    detection_latency_s: float | None      # None == censored (never got it right)
    censored: bool
    false_positive: bool                   # believed jamming when truth was fading
    false_positive_acted: bool             # actually hopped when truth was fading
    recovery_time_s: float | None
    packets_lost: float
    actions_consumed: int
    hops_consumed: int
    tests_before_correct: int | None
    survived: bool
    refusal_required: bool
    refusal_ok: bool | None
    expected_cost: float
    notes: str = ""


def _cost(matrix: dict, order: list[str], true_c: str, decl: str | None,
          abstain_cost: float) -> float:
    if decl is None or decl == "unknown":
        return abstain_cost
    if true_c not in matrix or decl not in order:
        return abstain_cost
    return float(matrix[true_c][order.index(decl)])


def score_episode(log: dict, truth: dict, cost_cfg: dict,
                  contract: dict, family: str) -> EpisodeScore:
    th = contract["thresholds"]
    b = contract["budgets"]
    true_cause = truth["cause"]
    onset = truth["onset_t"]
    recoverable = truth.get("recoverable", True)
    trace = log.get("classification_trace", [])

    # ---- first correct classification ----
    # A classification counts once the agent is CONFIDENTLY correct and that claim
    # is not a single-tick flicker: it must either be sustained for >=2 consecutive
    # decisions, or be acted upon (the agent committed to it). An agent that
    # diagnoses correctly, fixes the problem and stops talking about it must not be
    # scored as "never detected".
    # What the agent CLAIMED. An agent that separates its posterior from its
    # minimum-expected-cost decision records the decision in `declared`; one that does not
    # leaves it absent and argmax is used. Scoring argmax while advertising a cost rule was
    # AUDIT F4.3 -- every headline number reflected a rule the code never applied.
    def claim(row):
        # An abstention is a refusal to classify. `declared` is None on that row, but so
        # it is for any agent that never separates posterior from decision -- so the None
        # alone cannot distinguish the two. Key on the explicit flag instead, and return
        # a sentinel that can never equal a cause name, so an abstained tick is neither
        # credited as a detection nor charged as a misdiagnosis.
        if row.get("abstained"):
            return None
        return row.get("declared") or row["top"]

    act_times = [a["t"] for a in log.get("actions", [])]
    first_t = None
    for i, row in enumerate(trace):
        if claim(row) != true_cause or row["p"] < th["act_confidence"]:
            continue
        sustained = (i + 1 < len(trace)
                     and claim(trace[i + 1]) == true_cause
                     and trace[i + 1]["p"] >= th["act_confidence"])
        acted = any(row["t"] <= at <= row["t"] + 2.0 for at in act_times)
        if sustained or acted:
            first_t = row["t"]
            break

    censored = first_t is None
    det_lat = None if censored else max(0.0, first_t - onset)

    # The declared cause is the confident classification the agent COMMITTED TO --
    # the one in force when it spent budget on a recovery action. That is where the
    # operational cost of a misdiagnosis is actually incurred, so it is what the cost
    # matrix should score. Belief drift after the incident is resolved is not a new
    # claim.
    #
    # TWO failure modes, both real, and the rule has to survive both. They were found
    # one at a time and each fix on its own breaks the other case.
    #
    # (a) A single-tick hedge must not anchor. Anchoring to literally the first action
    #     scored an agent that bumped TX power on a guess, self-corrected on the very
    #     next tick, then diagnosed and mitigated a reactive jammer correctly for the
    #     rest of the episode, as a total miss.
    # (b) A correct diagnosis whose action WORKED must not be thrown away. Requiring the
    #     belief to survive into the next tick does exactly that: in spot_10075 the agent
    #     says `spot` at p=1.00, hops, the hop works (recovery at t=34) and the post-hop
    #     observations no longer look like spot -- so the belief that drove the successful
    #     action is discarded and the drift AFTER the incident is scored instead. The
    #     agent is marked wrong precisely because it was right. That is what the original
    #     "belief drift after the incident is resolved is not a new claim" protected.
    #
    # So a belief anchors if it is SUSTAINED into the next tick, OR if the action it drove
    # is the one that produced recovery. This mirrors first_t above, which has always used
    # `sustained or acted` rather than `sustained` alone.
    #
    # Independently (F5.1b): abstained ticks are never eligible to be that belief. An
    # abstention is never "a confident classification the agent committed to", however
    # high the posterior of the class it declined to name.
    confident = [(i, r) for i, r in enumerate(trace)
                 if r["p"] >= th["act_confidence"] and not r.get("abstained")]
    acts = [a for a in log.get("actions", []) if a["fn"] != "declare_link_lost"]
    rec_t = log.get("recovery_t")
    # The action that plausibly CAUSED the recovery: the episode's LAST recovery action,
    # with recovery following it. "Last" is doing the work. A mid-episode `recovery_t` is
    # not proof that the action before it was the fix -- a reactive jammer is intermittent,
    # so PDR recovers between bursts and reactive_10192 reports recovery_t=14.0 while the
    # incident runs to t=40. But an agent that acted, recovered, and then stopped acting
    # did fix it, and the belief that drove that action is the claim being scored.
    causal = acts[-1] if (rec_t is not None and acts and acts[-1]["t"] <= rec_t) else None

    final = None
    for a in acts:
        prior = [(i, r) for i, r in confident if r["t"] <= a["t"]]
        if not prior:
            continue
        # carry the index rather than recovering it with trace.index(anchor): that
        # compares dicts by value and would return the FIRST equal tick, not this one.
        i, anchor = prior[-1]
        sustained = (i + 1 < len(trace) and claim(trace[i + 1]) == claim(anchor)
                     and trace[i + 1]["p"] >= th["act_confidence"])
        if sustained or (causal is not None and a is causal):
            final = claim(anchor)
            break
    if final is None and confident:
        # Fall back to the last confident classification, full stop. An earlier version of
        # this truncated the fallback at recovery_t on the same "drift after resolution is
        # not a new claim" reasoning -- which is right when recovery ends the incident and
        # WRONG when it is a spurious intermittent recovery. reactive_10192: recovery_t is
        # 14.0, the agent is confused until t=14, then says `reactive` at p=1.00 for the
        # next 25 s and is scored `hidden_term`. Truncating threw away 25 s of a correct,
        # confident diagnosis. Without an action to attribute it to, recovery_t is not
        # evidence about which claim the agent committed to.
        final = claim(confident[-1][1])
    classification_ok = (final == true_cause)

    # ---- the expensive error ----
    believed_jam = any(claim(r) in JAMMING and r["p"] >= th["act_confidence"] for r in trace)
    hopped = any(a["fn"] in ("hop_channel", "channel_hop_probe")
                 for a in log.get("actions", []))
    fp = (true_cause == "fading") and believed_jam
    fp_acted = (true_cause == "fading") and hopped

    # ---- tests spent before getting it right ----
    tbc = None
    if first_t is not None:
        tbc = sum(1 for x in log.get("tests_run", []) if x["t"] <= first_t)

    rec = log.get("recovery_t")
    rec_time = None if rec is None else max(0.0, rec - onset)

    # ---- refusal gate (pass/fail, not averaged) ----
    refusal_required = not recoverable
    refusal_ok = None
    if refusal_required:
        dl = log.get("declared_lost_t")
        refusal_ok = bool(
            dl is not None
            and (dl - onset) <= b["declare_lost_deadline_s"]
            and log.get("hops_consumed", 0) <= b["max_channel_hops"]
        )

    # AUDIT F4.2: survival is a TRUTH-DEPENDENT metric (declaring the link lost is the
    # correct outcome only when the episode really was unrecoverable), so it is computed
    # here -- the one component allowed to see ground truth -- rather than read back from
    # a field the controller filled in from `sc.truth`.
    recovered = log.get("recovery_t") is not None
    declared_lost = log.get("declared_lost_t") is not None
    survived = bool(recovered or (declared_lost and not recoverable))
    ec = _cost(cost_cfg["matrix"], cost_cfg["order"], true_cause, final,
               cost_cfg.get("abstain_cost", 2.0))

    return EpisodeScore(
        episode_id=log.get("episode_id", "?"), scenario=log.get("scenario", "?"),
        family=family, agent=log.get("agent", "?"), true_cause=true_cause,
        declared_cause=final, classification_ok=classification_ok,
        detection_latency_s=det_lat, censored=censored,
        false_positive=fp, false_positive_acted=fp_acted,
        recovery_time_s=rec_time, packets_lost=float(log.get("packets_lost", 0.0)),
        actions_consumed=int(log.get("actions_consumed", 0)),
        hops_consumed=int(log.get("hops_consumed", 0)),
        tests_before_correct=tbc, survived=survived,
        refusal_required=refusal_required, refusal_ok=refusal_ok,
        expected_cost=ec, notes=log.get("end_reason", ""))


# --------------------------------------------------------------------------- #
def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def aggregate(scores: list[EpisodeScore]) -> dict:
    if not scores:
        return {}
    fading = [s for s in scores if s.true_cause == "fading"]
    refus = [s for s in scores if s.refusal_required]
    out = {
        "n": len(scores),
        "classification_acc": sum(s.classification_ok for s in scores) / len(scores),
        "expected_cost": sum(s.expected_cost for s in scores) / len(scores),
        "detection_latency_median_s": _median([s.detection_latency_s for s in scores]),
        "censoring_rate": sum(s.censored for s in scores) / len(scores),
        "fp_fading_belief": (sum(s.false_positive for s in fading) / len(fading)) if fading else None,
        "fp_fading_acted": (sum(s.false_positive_acted for s in fading) / len(fading)) if fading else None,
        "recovery_median_s": _median([s.recovery_time_s for s in scores]),
        "never_recovered_rate": sum(1 for s in scores if s.recovery_time_s is None) / len(scores),
        "packets_lost_mean": sum(s.packets_lost for s in scores) / len(scores),
        "actions_mean": sum(s.actions_consumed for s in scores) / len(scores),
        "hops_mean": sum(s.hops_consumed for s in scores) / len(scores),
        "survival_rate": sum(s.survived for s in scores) / len(scores),
        "refusal_gate_pass": (sum(bool(s.refusal_ok) for s in refus) / len(refus)) if refus else None,
    }
    by_fam: dict[str, dict] = {}
    fams = sorted({s.family for s in scores})
    for fam in fams:
        sub = [s for s in scores if s.family == fam]
        by_fam[fam] = {
            "n": len(sub),
            "classification_acc": sum(s.classification_ok for s in sub) / len(sub),
            "expected_cost": sum(s.expected_cost for s in sub) / len(sub),
            "detection_latency_median_s": _median([s.detection_latency_s for s in sub]),
            "survival_rate": sum(s.survived for s in sub) / len(sub),
            "actions_mean": sum(s.actions_consumed for s in sub) / len(sub),
        }
    out["by_family"] = by_fam
    out["confusion"] = confusion(scores)
    return out


def confusion(scores: list[EpisodeScore]) -> dict:
    m: dict[str, dict[str, int]] = {}
    for s in scores:
        m.setdefault(s.true_cause, {})
        k = s.declared_cause or "unknown"
        m[s.true_cause][k] = m[s.true_cause].get(k, 0) + 1
    return m


def gates(agg: dict, targets: dict) -> dict:
    """The two pass/fail gates that can fail an otherwise good agent (design §8.2).

    A gate with nothing to measure is N/A, not FAIL. Running one spot episode does not
    fail the fading false-positive gate -- there was no fading episode in it. Reporting
    that as FAIL trains the reader to ignore the gates, which is the opposite of the point.
    Both gates are only meaningful over a corpus that contains the relevant families.
    """
    g = {}
    fp = agg.get("fp_fading_acted")
    fp_max = targets.get("fp_max", 0.05)
    g["false_positive_gate"] = {
        "value": fp, "target": fp_max, "applicable": fp is not None,
        "pass": (fp is None or fp <= fp_max),
        "note": "" if fp is not None else "no fading episodes in this run",
    }
    rg = agg.get("refusal_gate_pass")
    g["refusal_gate"] = {
        "value": rg, "target": 1.0, "applicable": rg is not None,
        "pass": (rg is None or rg >= 1.0),
        "note": "" if rg is not None else "no unrecoverable episodes in this run",
    }
    return g
