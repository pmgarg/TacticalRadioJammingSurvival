"""
The on-device student.

Inference is a pure-numpy forward pass over the exported bundle -- no sklearn, no
framework. That is deliberate: it is the same arithmetic the C/ESP32 build will do,
so what runs here is what would run on the drone.

Decision rule is MINIMUM EXPECTED COST under the confusion-cost matrix, not argmax:
declaring jamming when the truth is fading costs 10, so the model must be far more
confident before it will say "jamming". Below the calibrated abstain threshold it
returns "unknown" and the controller keeps investigating instead of acting.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .api import Agent, Context, Decision, CAUSES, normalise
from .policy_table import PLAYBOOK

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# student_v9: retrained on the fixed ns-3 world (docs/FIXES.md "student_v9 -- retrained
# on the fixed world"), the canonical bundle as of the audit-fixes merge. The prior
# default pointed at data/student/student_bundle.json, a directory that does not exist
# in this tree -- StudentAgent() with no explicit bundle_path crashed on FileNotFoundError.
DEFAULT_BUNDLE = os.path.join(HERE, "..", "data", "student_v9", "student_bundle.json")


def _relu(x):
    return np.maximum(x, 0.0)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class _Net:
    def __init__(self, d):
        self.classes = d["classes"]
        if "W_int8" in d:
            # int8 weights with per-output-channel scales: dequantise once at load,
            # which is exactly what the ESP32 kernel does per layer.
            self.W = [np.asarray(q, dtype=np.float32) * np.asarray(s, dtype=np.float32)[None, :]
                      for q, s in zip(d["W_int8"], d["W_scale"])]
        else:
            self.W = [np.asarray(w, dtype=np.float32) for w in d["W"]]
        self.b = [np.asarray(b, dtype=np.float32) for b in d["b"]]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = _relu(h)
        return _softmax(h)


class StudentAgent:
    name = "student"

    def __init__(self, bundle_path: str | None = None, use_cost_rule: bool = True):
        p = bundle_path or DEFAULT_BUNDLE
        d = json.load(open(p))
        self.cause = _Net(d["cause"])
        self.call = _Net(d["call"])
        self.cost_order = d["cost_order"]
        self.COST = np.asarray(d["cost_matrix"], dtype=np.float32)
        self.abstain = float(d.get("abstain_threshold", 0.0))
        self.use_cost_rule = use_cost_rule
        self.n_features = int(d["n_features"])
        self.reset()

    def reset(self) -> None:
        self._last = None

    # ------------------------------------------------------------------ #
    def _belief(self, p: np.ndarray) -> dict[str, float]:
        b = {c: 0.0 for c in CAUSES}
        for c, v in zip(self.cause.classes, p):
            if c in b:
                b[c] = float(v)
        return normalise(b)

    def _declare(self, p: np.ndarray) -> tuple[str, float]:
        """Minimum-expected-cost decision (design §9.3)."""
        P = np.zeros(len(self.cost_order), dtype=np.float32)
        for c, v in zip(self.cause.classes, p):
            if c in self.cost_order:
                P[self.cost_order.index(c)] = v
        if not self.use_cost_rule:
            i = int(np.argmax(P))
            return self.cost_order[i], float(P[i])
        exp = P @ self.COST
        i = int(np.argmin(exp))
        return self.cost_order[i], float(P[i])

    # ------------------------------------------------------------------ #
    def decide(self, features: list[float], ctx: Context) -> Decision:
        x = np.asarray(features, dtype=np.float32)
        pc = self.cause(x)
        belief = self._belief(pc)
        top, conf = self._declare(pc)

        # calibrated abstention: below threshold we do not claim a cause
        if conf < self.abstain:
            return Decision(belief, "no_op", {}, conf,
                            why=f"posterior {conf:.2f} below the calibrated abstain "
                                f"threshold {self.abstain:.2f}; gathering more evidence "
                                f"rather than acting",
                            declared=None, abstained=True)

        # `declared` is the cost-rule decision; `belief` stays the honest posterior.
        pa = self.call(x)
        order = np.argsort(-pa)
        for i in order:
            call = self.call.classes[int(i)]
            if call in ctx.available:
                return Decision(belief, call, self._args(call, ctx), conf,
                                why=f"student: {top} (p={conf:.2f}); "
                                    f"{PLAYBOOK.get(top, ('', 'learned policy'))[1]}",
                                expect="PDR recovers to >=0.8 of baseline within 3 s",
                                declared=top)
        return Decision(belief, "no_op", {}, conf, why="no available call", declared=top)

    def _args(self, call: str, ctx: Context) -> dict:
        if call in ("hop_channel", "channel_hop_probe"):
            scan = ctx.last_scan
            if scan and getattr(scan, "channels", None):
                alts = [c for c in scan.channels if c.ch != ctx.channel]
                if alts:
                    return {"channel": min(alts, key=lambda c: c.noise_dbm).ch}
            return {"channel": (ctx.channel % ctx.n_channels) + 1}
        return {"set_tx_power": {"dbm": 20}, "change_tdma_slot": {"slot": 3},
                "reroute": {"via": None}, "spectrum_scan": {"dwell_ms": 20},
                "silent_listen": {"duration_ms": 200}, "load_test": {"factor": 0.5},
                "neighbor_probe": {"peer": "P0"},
                "mobility_test": {"dx": 30.0, "dy": 0.0, "dz": 0.0},
                "move": {"dx": 30.0, "dy": 0.0, "dz": 0.0}}.get(call, {})
