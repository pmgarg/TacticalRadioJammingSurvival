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
    act_times = [a["t"] for a in log.get("actions", [])]
    first_t = None
    for i, row in enumerate(trace):
        if row["top"] != true_cause or row["p"] < th["act_confidence"]:
            continue
        sustained = (i + 1 < len(trace)
                     and trace[i + 1]["top"] == true_cause
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
    # claim. Falls back to the last confident classification if it never acted.
    confident = [r for r in trace if r["p"] >= th["act_confidence"]]
    acts = [a for a in log.get("actions", []) if a["fn"] != "declare_link_lost"]
    final = None
    if acts:
        t_act = acts[0]["t"]
        prior = [r for r in confident if r["t"] <= t_act]
        if prior:
            final = prior[-1]["top"]
    if final is None:
        final = confident[-1]["top"] if confident else None
    classification_ok = (final == true_cause)

    # ---- the expensive error ----
    believed_jam = any(r["top"] in JAMMING and r["p"] >= th["act_confidence"] for r in trace)
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

    survived = bool(log.get("survived", False))
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
    """The two pass/fail gates that can fail an otherwise good agent (design §8.2)."""
    g = {}
    fp = agg.get("fp_fading_acted")
    g["false_positive_gate"] = {
        "value": fp, "target": targets.get("fp_max", 0.05),
        "pass": (fp is not None and fp <= targets.get("fp_max", 0.05)),
    }
    rg = agg.get("refusal_gate_pass")
    g["refusal_gate"] = {
        "value": rg, "target": 1.0,
        "pass": (rg is not None and rg >= 1.0),
    }
    return g
