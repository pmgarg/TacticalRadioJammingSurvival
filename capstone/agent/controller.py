"""
The agent controller — the deterministic wrapper around whatever model is driving.

It owns everything that must NOT be learned (design §7.6):
  1. action masking      - unavailable calls are removed before the agent chooses,
                           so "hop forever" is impossible by construction
  2. escalation depth    - RECORDED, not enforced. `self.rung` is the deepest rung the
                           episode reached; it is a reported statistic. An earlier
                           docstring called it a monotone constraint, but nothing read it
                           (AUDIT F4.8). Masking and the budget are the real constraints.
  3. dead-man failsafe   - forces declare_link_lost() even if the model hangs,
                           is wrong, or never asks for it

It also runs the episode loop: percept at 10 Hz, decisions at 1 Hz, cost accounting,
recovery verification, and emission of the episode log the verifier reads.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from percept.features import FeatureExtractor
from .api import Agent, Context, Decision, CAUSES, UNKNOWN, DIAGNOSE

# A reciprocal report older than this is stale -- fall back to the forward direction alone
# rather than acting on what a peer said about us ten seconds ago.
REVERSE_REPORT_MAX_AGE_S = 3.0


def _link_health(l) -> float:
    """Usable delivery on a link: the worse of the two directions. See DECISION D9."""
    rev = getattr(l, "pdr_reverse", None)
    age = getattr(l, "report_age_s", 999.0)
    if rev is None or age > REVERSE_REPORT_MAX_AGE_S:
        return l.pdr
    return min(l.pdr, float(rev))

# escalation ladder: index = how far down we have gone
LADDER = ["no_op", "set_tx_power", "change_tdma_slot", "reroute",
          "hop_channel", "move", "fallback_to_lora", "declare_link_lost"]

COSTLY = {"hop_channel", "channel_hop_probe", "fallback_to_lora", "move", "mobility_test"}
BUDGET_COST = {"hop_channel": 1, "channel_hop_probe": 1, "fallback_to_lora": 1,
               "move": 2, "mobility_test": 2}


@dataclass
class EpisodeLog:
    episode_id: str
    scenario: str
    agent: str
    attack_onset_t: float | None = None
    first_correct_classification_t: float | None = None
    classification_trace: list[dict] = field(default_factory=list)
    tests_run: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    recovery_t: float | None = None
    packets_lost: float = 0.0
    actions_consumed: int = 0
    hops_consumed: int = 0
    tests_before_correct_classification: int | None = None
    recovered: bool = False
    survived: bool | None = None      # computed by the verifier, not here
    declared_jamming: bool = False
    declared_lost_t: float | None = None
    end_reason: str = ""
    pdr_series: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Controller:
    def __init__(self, sim, scenario, agent: Agent, contract: dict,
                 decision_hz: float = 1.0, verbose: bool = False):
        self.sim, self.sc, self.agent = sim, scenario, agent
        self.k = contract
        self.b = dict(contract["budgets"])
        self.th = dict(contract["thresholds"])
        self.decision_every = max(1, int(round((1.0 / decision_hz) / sim.dt)))
        self.verbose = verbose
        self.ex = FeatureExtractor(scenario.n_channels, dt=sim.dt)
        self.reset()

    def reset(self) -> None:
        self.used = {"hops": 0, "scans": 0, "silent": 0, "moves": 0, "costly": 0, "lora": 0}
        self.tried: dict[tuple[str, str], float] = {}   # (hypothesis, call) -> t
        self.rung = 0
        self.last_scan_t = -1e9
        self.baseline_pdr = None
        self.onset_t = None
        self.recovery_hold_start = None
        self.declared = False
        self.records: list[dict] = []      # (features, decision) for distillation
        self.log = EpisodeLog(episode_id=f"{self.sc.name}#{self.sc.seed}",
                              scenario=self.sc.name, agent=self.agent.name)
        self.agent.reset()

    # ------------------------------------------------------------------ #
    def _available(self, t: float, no_link_for: float) -> list[str]:
        """The action mask. This is what makes the refusal structural."""
        # listen_test / transmit_probe removed in contract 1.2.0 -- no simulator
        # implemented them (tests/test_contract_effects.py), so they cost budget and
        # returned nothing. DESIGN v2.0 A13.
        av = ["no_op", "set_tx_power", "reroute", "change_tdma_slot",
              "neighbor_probe", "load_test"]
        if self.used["scans"] < self.b["max_spectrum_scans"] and \
           (t - self.last_scan_t) >= self.b["min_scan_interval_s"]:
            av.append("spectrum_scan")
        if self.used["silent"] < self.b["max_silent_listens"]:
            av.append("silent_listen")
        scan_fresh = (t - self.last_scan_t) < 10.0
        hop_useful = False
        sc_obj = getattr(self, "_last_scan_obj", None)
        if scan_fresh and sc_obj and getattr(sc_obj, "channels", None):
            chans = sc_obj.channels
            cur = next((c for c in chans if c.ch == self.sim.channel), None)
            alts = [c for c in chans if c.ch != self.sim.channel]
            if cur and alts:
                best = min(alts, key=lambda c: c.noise_dbm)
                # hot here, or a materially quieter channel to move to
                hop_useful = (cur.noise_dbm > self.th["jam_energy_dbm"]
                              or (cur.noise_dbm - best.noise_dbm) > 6.0)
        if (self.used["hops"] < self.b["max_channel_hops"]
                and self.used["costly"] < self.b["max_costly_actions"]
                and scan_fresh and hop_useful):
            av += ["hop_channel", "channel_hop_probe"]
        if self.used["moves"] < self.b["max_moves"] and \
           self.used["costly"] < self.b["max_costly_actions"]:
            av += ["move", "mobility_test"]
        if self.sc.lora_available and self.used["costly"] < self.b["max_costly_actions"]:
            av.append("fallback_to_lora")
        av.append("declare_link_lost")
        return av

    def _failsafe_triggered(self, t: float, no_link_for: float, ctx: Context) -> str | None:
        """Dead-man rule — runs whether or not the model produced anything."""
        if no_link_for >= self.th["no_link_declare_s"]:
            return "no_usable_link_timeout"
        if self.onset_t and (t - self.onset_t) >= self.b["recovery_timeout_s"]:
            return "recovery_timeout"
        if self.used["costly"] >= self.b["max_costly_actions"] and no_link_for > 3.0:
            return "budget_exhausted"
        # All channels scanned bad + no reachable peer + no escape => unrecoverable.
        #
        # AUDIT F4.2: this used to read `sc.lora_jammed`, a ground-truth field set together
        # with `recoverable=False` for the refusal family -- so the failsafe that produces
        # the refusal-gate pass was conditioned on the refusal label. The agent could not
        # see it, but the controller ships with the agent, so it was truth leaking into the
        # deployed artefact. The escape test is now purely what the drone can know:
        # either it has no fallback radio at all, or it already spent the fallback and the
        # link did not come back. That is also better behaviour -- it must TRY the escape
        # before declaring there is none.
        scan = ctx.last_scan
        if scan and scan.channels:
            allbad = all(c.noise_dbm > self.th["jam_energy_dbm"] for c in scan.channels)
            no_escape = (not self.sc.lora_available) or (self.used["lora"] > 0)
            if allbad and no_link_for > 2.0 and no_escape:
                return "all_channels_jammed_no_escape"
        return None

    # ------------------------------------------------------------------ #
    def _apply(self, d: Decision, t: float) -> None:
        c, a = d.call, d.args
        s = self.sim
        if c == "spectrum_scan":
            res = s.scan(a.get("channels"), a.get("dwell_ms", 20))
            self.ex.note_scan(res)
            self._last_scan_obj = res
            self.used["scans"] += 1
            self.last_scan_t = t
        elif c == "silent_listen":
            self.used["silent"] += 1
        elif c == "load_test":
            s.load_test(a.get("factor", 0.5))
        elif c == "neighbor_probe":
            if hasattr(s, "neighbor_probe"):
                s.neighbor_probe(a.get("peer"))
        elif c == "hop_channel":
            s.hop_channel(a.get("channel", s.channel))
            self.used["hops"] += 1
            self.used["costly"] += 1
        elif c == "channel_hop_probe":
            s.hop_channel(a.get("channel", s.channel))
            self.used["hops"] += 1
            self.used["costly"] += 1
        elif c == "reroute":
            if hasattr(s, "reroute"):
                s.reroute(a.get("via"))
        elif c == "set_tx_power":
            s.set_tx_power(a.get("dbm", 20))
        elif c == "change_tdma_slot":
            s.change_tdma_slot(a.get("slot", 1))
        elif c == "move" or c == "mobility_test":
            s.move(a.get("dx", 30.0), a.get("dy", 0.0), a.get("dz", 0.0))
            self.used["moves"] += 1
            self.used["costly"] += 2
        elif c == "fallback_to_lora":
            s.fallback_to_lora()
            self.used["lora"] += 1
            self.used["costly"] += BUDGET_COST["fallback_to_lora"]
        elif c == "declare_link_lost":
            s.declare_link_lost()
            self.declared = True
        if c in LADDER:
            self.rung = max(self.rung, LADDER.index(c))

    # ------------------------------------------------------------------ #
    def run(self) -> EpisodeLog:
        sc, s = self.sc, self.sim
        n_steps = int(sc.duration_s / s.dt)
        no_link_for = 0.0
        self._last_scan_obj = None
        pdr_hist = []
        offered_total = delivered_total = 0.0

        for i in range(n_steps):
            t = s.t
            budget_view = {
                "hops_used": self.used["hops"], "scans_used": self.used["scans"],
                "actions_used": self.used["costly"],
                "max_channel_hops": self.b["max_channel_hops"],
                "max_spectrum_scans": self.b["max_spectrum_scans"],
                "max_costly_actions": self.b["max_costly_actions"],
            }
            obs = s.step(budget_view)
            feats = self.ex.update(obs)

            # DECISION D9 (AUDIT/G1): link health is BIDIRECTIONAL.
            #
            # This used to average the forward PDR only -- the frames WE receive. Under that
            # metric `set_tx_power` is inert by construction, because raising our own transmit
            # power cannot improve what we hear. But PLAYBOOK["fading"] = set_tx_power, so the
            # recommended remedy for the single most important family provably could not move
            # the number it was judged on. G1 caught it: the call changed nothing in any family.
            #
            # A link is only usable if traffic flows BOTH ways -- an 802.11 data frame that is
            # not ACKed is a lost frame, and the ACK depends on the peer hearing US. So health
            # is the WORSE of the two directions, using the peer's own reciprocal report
            # (TELEMETRY.md §2) when it is fresh enough to trust.
            #
            # This keeps the physics honest rather than pretending our TX power improves our
            # RX: under fading BOTH directions degrade and raising power lifts the reverse one,
            # so the action helps. Under jamming at OUR location the reverse link stays healthy
            # and only the forward one is hurt, so raising power does NOT help -- which is
            # exactly the discrimination the design wants.
            pdr = (sum(_link_health(l) for l in obs.links.values())
                   / max(1, len(obs.links)))
            pdr_hist.append(pdr)
            offered_total += 1.0
            delivered_total += pdr
            if self.baseline_pdr is None and t > 5.0:
                self.baseline_pdr = sum(pdr_hist[:int(5.0 / s.dt)]) / max(1, int(5.0 / s.dt))
            no_link_for = 0.0 if pdr > 0.2 else no_link_for + s.dt
            if self.ex.onset_t and self.onset_t is None:
                self.onset_t = self.ex.onset_t
                self.log.attack_onset_t = self.ex.onset_t

            if self.declared:
                continue
            # Once recovery is verified and holding, the incident is over: go back
            # to nominal monitoring instead of continuing to spend actions.
            if self.log.recovery_t is not None and pdr > 0.6:
                continue
            # ---- decide at the decision cadence, only once the gate has fired ----
            if self.onset_t is None or (i % self.decision_every) != 0:
                continue

            ctx = Context(t=t, channel=s.channel, n_channels=sc.n_channels,
                          peers=list(obs.links.keys()), budget=budget_view,
                          available=self._available(t, no_link_for),
                          last_scan=self._last_scan_obj,
                          tests_run=[x["test"] for x in self.log.tests_run],
                          actions_taken=[x["fn"] for x in self.log.actions],
                          hypothesis_history=[x["top"] for x in self.log.classification_trace],
                          lora_available=sc.lora_available)

            reason = self._failsafe_triggered(t, no_link_for, ctx)
            if reason:
                # Force the ACTION, never the diagnosis. Preserve whatever the agent
                # last believed; the controller has no opinion about the cause.
                last = (self.log.classification_trace[-1]["belief"]
                        if self.log.classification_trace
                        else {c: 1.0 / len(CAUSES) for c in CAUSES})
                d = Decision(belief=dict(last), call="declare_link_lost",
                             confidence=0.0, unrecoverable=1.0,
                             why=f"controller failsafe: {reason}")
                self.log.end_reason = reason
            else:
                d = self.agent.decide(feats, ctx)
                # An action already tried for this same hypothesis, which did not
                # restore the link, is not tried again -- escalate or abstain.
                # AUDIT: the repeat-suppressor used to cover EVERY call, including
                # diagnostic tests. Re-issuing a RECOVERY that already failed is pointless
                # -- that is what this is for. Re-issuing a TEST is how you get a second
                # sample, and `scan_periodicity` is defined as "the hot channel moved
                # BETWEEN scans", so it is uncomputable from one. Suppressing the second
                # scan therefore made sweep undiagnosable and, worse, did it silently: the
                # call was rewritten to no_op while `spectrum_scan` stayed in the action
                # mask, so the agent asked again, and again, and never learned why. The
                # budget (max_spectrum_scans) is what bounds tests; this is not.
                key = (d.top, d.call)
                if (d.call not in ("no_op", "declare_link_lost")
                        and d.call not in DIAGNOSE
                        and key in self.tried
                        and (t - self.tried[key]) < 900.0):
                    d = Decision(belief=d.belief, call="no_op",
                                 confidence=d.confidence, unrecoverable=d.unrecoverable,
                                 why=f"already tried {d.call} for {d.top} at "
                                     f"t={self.tried[key]:.0f}s and it did not recover")
                elif d.call not in ("no_op",):
                    self.tried[key] = t
                if d.call not in ctx.available:      # HARD mask
                    d = Decision(belief=d.belief, call="no_op",
                                 confidence=d.confidence, unrecoverable=d.unrecoverable,
                                 why=f"masked: {d.call} unavailable; " + d.why)

            self.records.append({"t": round(t, 2), "features": list(feats),
                                 "call": d.call, "args": dict(d.args or {}),
                                 "top": d.top, "available": list(ctx.available)})
            self.log.classification_trace.append(
                {"t": round(t, 2), "top": d.top, "p": round(d.top_p, 3),
                 # `declared` is what the agent COMMITS to under the cost matrix; `top` is
                 # argmax of the posterior. They differ whenever the asymmetry bites, which
                 # is the whole reason the matrix exists (AUDIT F4.3).
                 "declared": d.declared,
                 # An abstention is a positive refusal to claim, not a missing field.
                 # The verifier must not fall back to argmax for these rows.
                 "abstained": bool(getattr(d, "abstained", False)),
                 "belief": {k: round(v, 3) for k, v in d.belief.items()},
                 "unrecoverable": round(d.unrecoverable, 3)})
            if (not getattr(d, "abstained", False)
                    and d.top in ("barrage", "spot", "reactive", "sweep")
                    and d.top_p >= self.th["act_confidence"]):
                self.log.declared_jamming = True

            if d.call in ("spectrum_scan", "neighbor_probe", "silent_listen",
                          "load_test", "channel_hop_probe", "mobility_test"):
                self.log.tests_run.append({"t": round(t, 2), "test": d.call,
                                           "args": d.args, "why": d.why})
            elif d.call != "no_op":
                self.log.actions.append({"t": round(t, 2), "fn": d.call, "args": d.args,
                                         "why": d.why, "expect": d.expect,
                                         "budget_after": self.b["max_costly_actions"] - self.used["costly"]})
            self._apply(d, t)
            if d.call == "declare_link_lost":
                self.log.declared_lost_t = round(t, 2)
                if not self.log.end_reason:
                    self.log.end_reason = "agent_declared"

            # recovery verification
            if self.baseline_pdr and pdr >= self.th["recovery_pdr_frac_of_baseline"] * self.baseline_pdr:
                if self.recovery_hold_start is None:
                    self.recovery_hold_start = t
                elif (t - self.recovery_hold_start) >= self.th["recovery_hold_s"] \
                        and self.log.recovery_t is None:
                    self.log.recovery_t = round(t, 2)
            else:
                self.recovery_hold_start = None

        self.log.actions_consumed = self.used["costly"]
        self.log.hops_consumed = self.used["hops"]
        self.log.packets_lost = round(offered_total - delivered_total, 1)
        self.log.pdr_series = [[round(i * s.dt, 1), round(p, 3)]
                               for i, p in enumerate(pdr_hist) if i % 5 == 0]
        # AUDIT F4.2: `survived` used to be computed HERE, from `sc.truth.recoverable`,
        # and the verifier simply read it back -- so the component that is not allowed to
        # see ground truth was handing the scorer a truth-derived metric. The controller
        # now records only what it observed; verify/verifier.py computes survival.
        recovered = self.log.recovery_t is not None
        self.log.recovered = recovered
        self.log.survived = None
        if not self.log.end_reason:
            self.log.end_reason = "recovered" if recovered else "episode_end"
        return self.log
