"""
Teacher: a privileged expert used ONLY offline, in simulation, to label traces.

Design note (important, and stated in the report). The design's §7.2 warns that a
teacher must not act on information the student cannot have, or the student learns to
guess. We honour that by splitting where the supervision comes from:

  * the CLASSIFICATION target comes from the simulator's ground truth. That is
    ordinary supervised learning on a free, unlimited label -- not cheating, and it
    removes ~70% of the teacher cost the design budgeted for.
  * the ACTION target is a deterministic function of the BELIEF (the §7.5 playbook),
    so it is reproducible by any agent that classifies correctly.
  * the TEST-SELECTION target is the cheapest test that would actually disambiguate
    the situation the student is looking at. The oracle can compute this because it
    knows the answer; the student learns the mapping from observations.

This is privileged-expert distillation (as in DAgger with an expert), and the student
is evaluated with NO privileged access at all.
"""
from __future__ import annotations

from .api import Agent, Context, Decision, CAUSES, normalise

# feature indices (see percept.features.FEATURE_NAMES)
F_PDR, F_SPREAD, F_DEG, F_CORR = 0, 5, 6, 12
F_NOISE_D, F_NOPRE, F_TXN = 17, 20, 22
F_RETRY, F_LOAD, F_LOADCORR = 24, 28, 29
F_SCANAGE, F_SCANBAD, F_SCANSPREAD, F_ALT, F_PERIOD = 30, 31, 32, 34, 35
F_HB = 43

# The recovery playbook (design §7.5): belief -> action. Deterministic given belief.
PLAYBOOK = {
    "spot":        ("hop_channel", "a clean channel exists; move to it"),
    "sweep":       ("hop_channel", "hop ahead of the sweep"),
    "barrage":     ("fallback_to_lora", "all channels hot; trade rate for a link"),
    "reactive":    ("change_tdma_slot", "deny the jammer its trigger; do NOT hop"),
    "fading":      ("set_tx_power", "geometry, not an attacker; raise margin, NEVER hop"),
    "node_loss":   ("reroute", "the peer is gone; route around it"),
    "congestion":  ("change_tdma_slot", "back off and separate transmitters"),
    "hidden_term": ("change_tdma_slot", "separate the colliding transmitters"),
}

# Which cheap test would CONFIRM each cause, and the evidence that makes it unnecessary
CONFIRMING_TEST = {
    "barrage":     ("spectrum_scan", F_SCANAGE),
    "spot":        ("spectrum_scan", F_SCANAGE),
    "sweep":       ("spectrum_scan", F_SCANAGE),
    "reactive":    ("silent_listen", None),
    "node_loss":   ("neighbor_probe", None),
    "congestion":  ("load_test", None),
    "hidden_term": ("load_test", None),
    "fading":      ("mobility_test", None),
}


class TeacherAgent:
    """Privileged expert. Construct with the true cause; never deployed."""
    name = "teacher"

    def __init__(self, true_cause: str, recoverable: bool = True,
                 confirm_before_acting: bool = True):
        self.true = true_cause
        self.recoverable = recoverable
        self.confirm = confirm_before_acting
        self.reset()

    def reset(self) -> None:
        self._confirmed = False
        self._tests = 0

    # ------------------------------------------------------------------ #
    def _belief(self) -> dict[str, float]:
        b = {c: 0.02 for c in CAUSES}
        b[self.true] = 0.9
        return normalise(b)

    def _evidence_sufficient(self, f: list[float]) -> bool:
        """Would a competent observer already be sure, from what the STUDENT sees?"""
        t = self.true
        if t in ("barrage", "spot", "sweep"):
            fresh = f[F_SCANAGE] < 0.35
            if not fresh:
                return False
            if t == "sweep":
                return f[F_PERIOD] > 0.3
            if t == "barrage":
                return f[F_SCANBAD] > 0.0 and f[F_SCANSPREAD] < 0.25
            return f[F_SCANSPREAD] >= 0.25
        if t == "reactive":
            return f[F_TXN] > 0.35
        if t == "node_loss":
            return f[F_SPREAD] > 0.3 and f[F_HB] > 0.4
        if t == "congestion":
            return f[F_LOADCORR] > 0.25 and f[F_LOAD] > 0.2
        if t == "hidden_term":
            return f[F_RETRY] > 0.1 and f[F_LOAD] < 0.2
        if t == "fading":
            # the robust call: degraded with a demonstrably clean channel
            return f[F_NOISE_D] < 0.15 and (f[F_PDR] < -0.3 or f[F_CORR] > 0.35)
        return True

    # ------------------------------------------------------------------ #
    def decide(self, features: list[float], ctx: Context) -> Decision:
        b = self._belief()
        f = features

        # Unrecoverable: the expert recognises hopelessness early and refuses.
        if not self.recoverable:
            allbad = (f[F_SCANAGE] < 0.5 and f[F_SCANBAD] > 0.3)
            if allbad or f[F_PDR] < -0.8:
                if "declare_link_lost" in ctx.available:
                    return Decision(b, "declare_link_lost", {}, 0.95, unrecoverable=0.95,
                                    why="every channel is jammed and no peer is reachable; "
                                        "hopping cannot help - execute the failsafe")
            if "spectrum_scan" in ctx.available and f[F_SCANAGE] >= 0.35:
                return Decision(b, "spectrum_scan", {"dwell_ms": 20}, 0.5,
                                why="confirm no clean channel exists before refusing",
                                expect="all channels hot => unrecoverable")

        # Gather the confirming evidence first, when it is cheap and not yet in hand.
        if self.confirm and not self._evidence_sufficient(f) and self._tests < 3:
            test, _ = CONFIRMING_TEST.get(self.true, (None, None))
            if test and test in ctx.available:
                self._tests += 1
                return Decision(b, test, self._test_args(test), 0.5,
                                why=f"evidence for {self.true} not yet in hand; "
                                    f"{test} is the cheapest test that settles it",
                                expect=f"result consistent with {self.true}")

        # Act on the (correct) belief.
        call, why = PLAYBOOK[self.true]
        if call not in ctx.available:
            for alt in ("set_tx_power", "reroute", "change_tdma_slot", "no_op"):
                if alt in ctx.available:
                    call, why = alt, f"{call} unavailable; falling back to {alt}"
                    break
        return Decision(b, call, self._act_args(call, ctx), 0.9, why=why,
                        expect="PDR recovers to >=0.8 of baseline within 3 s")

    # ------------------------------------------------------------------ #
    def _test_args(self, test: str) -> dict:
        return {"spectrum_scan": {"dwell_ms": 20},
                "silent_listen": {"duration_ms": 200},
                "neighbor_probe": {"peer": "P0"},
                "load_test": {"factor": 0.5},
                "mobility_test": {"dx": 30.0, "dy": 0.0, "dz": 0.0}}.get(test, {})

    def _act_args(self, call: str, ctx: Context) -> dict:
        if call == "hop_channel":
            scan = ctx.last_scan
            if scan and getattr(scan, "channels", None):
                alts = [c for c in scan.channels if c.ch != ctx.channel]
                if alts:
                    return {"channel": min(alts, key=lambda c: c.noise_dbm).ch}
            return {"channel": (ctx.channel % ctx.n_channels) + 1}
        if call == "set_tx_power":
            return {"dbm": 20}
        if call == "change_tdma_slot":
            return {"slot": 3}
        if call == "reroute":
            return {"via": None}
        if call in ("move", "mobility_test"):
            return {"dx": 30.0, "dy": 0.0, "dz": 0.0}
        return {}
