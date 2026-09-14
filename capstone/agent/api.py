"""Shared agent types. Baseline, student and teacher all implement Agent.decide()."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

CAUSES = ["barrage", "spot", "reactive", "sweep",
          "fading", "node_loss", "congestion", "hidden_term"]
UNKNOWN = "unknown"

# listen_test and transmit_probe were removed in contract 1.2.0 -- no simulator
# implemented them, so they cost budget and returned nothing (DESIGN v2.0 A13).
DIAGNOSE = ["spectrum_scan", "neighbor_probe", "silent_listen", "load_test",
            "channel_hop_probe", "mobility_test"]
ACTIONS = ["no_op", "set_tx_power", "reroute", "change_tdma_slot",
           "hop_channel", "fallback_to_lora", "move", "declare_link_lost"]
ALL_CALLS = DIAGNOSE + ACTIONS


@dataclass
class Decision:
    belief: dict[str, float]
    call: str
    args: dict = field(default_factory=dict)
    confidence: float = 0.0
    unrecoverable: float = 0.0
    why: str = ""
    expect: str = ""            # the observable change this action predicts
    declared: str | None = None
    """The cause the agent COMMITS to, which is not the same as argmax(belief).

    Under an asymmetric cost matrix the rational declaration is the minimum-expected-cost
    class, and that can differ from the most probable one -- that is the entire point of
    having the matrix. `belief` stays the honest posterior; `declared` is the decision.
    Agents that do not separate the two leave this None and the verifier falls back to
    argmax, so nothing downstream breaks."""

    @property
    def top(self) -> str:
        if not self.belief:
            return UNKNOWN
        return max(self.belief, key=self.belief.get)

    @property
    def top_p(self) -> float:
        return max(self.belief.values()) if self.belief else 0.0


@dataclass
class Context:
    """Everything the agent may see besides the feature vector. No ground truth."""
    t: float
    channel: int
    n_channels: int
    peers: list[str]
    budget: dict
    available: list[str]                    # action mask (design §7.6 mechanism 1)
    last_scan: object | None = None
    tests_run: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    hypothesis_history: list[str] = field(default_factory=list)
    recovery_attempts: list[dict] = field(default_factory=list)
    lora_available: bool = True


class Agent(Protocol):
    name: str
    def decide(self, features: list[float], ctx: Context) -> Decision: ...
    def reset(self) -> None: ...


def uniform_belief() -> dict[str, float]:
    p = 1.0 / len(CAUSES)
    return {c: p for c in CAUSES}


def normalise(b: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in b.values())
    if s <= 0:
        return uniform_belief()
    return {k: max(0.0, v) / s for k, v in b.items()}
