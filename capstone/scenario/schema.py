"""
Data-driven scenario definition.

The engine accepts an ARBITRARY mesh: any node count, any topology, any mobility,
any traffic pattern, and any composition of jammers/impairments. The graded corpus
(§10.1 of the design) is a frozen subset of what this schema can express, but the
instructor can hand us a new YAML file at demo time and it runs unchanged.

One scenario file == one reproducible episode (given its seed).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

CONTRACT_PATH = os.path.join(os.path.dirname(__file__), "..", "contract", "agent_contract.json")

CAUSES = ["barrage", "spot", "reactive", "sweep",
          "fading", "node_loss", "congestion", "hidden_term"]
JAMMER_TYPES = {"barrage", "spot", "sweep", "reactive", "none"}


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    id: str
    pos: list[float]                       # [x, y, z] metres
    role: str = "peer"                     # "agent" | "peer" | "interferer"
    tx_power_dbm: float = 16.0
    alive: bool = True

    def __post_init__(self):
        if len(self.pos) != 3:
            raise ValueError(f"node {self.id}: pos must be [x,y,z], got {self.pos}")


@dataclass
class Mobility:
    node: str
    type: str = "static"                   # static | constant_velocity | waypoint | random_walk
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    waypoints: list[list[float]] = field(default_factory=list)
    speed: float = 5.0
    bounds: list[float] | None = None       # [xmin,ymin,xmax,ymax] for random_walk


@dataclass
class Flow:
    src: str
    dst: str
    rate_kbps: float = 200.0
    packet_bytes: int = 512
    start_s: float = 1.0
    stop_s: float | None = None
    type: str = "cbr"                      # cbr | burst


@dataclass
class Jammer:
    id: str
    type: str                              # barrage | spot | sweep | reactive
    pos: list[float]
    eirp_dbm: float = 15.0
    channels: list[int] = field(default_factory=list)   # spot/barrage target set
    duty: float = 1.0                      # 0..1
    # sweep
    dwell_ms: float = 200.0
    sweep_order: str = "linear"            # linear | random
    # reactive
    threshold_dbm: float = -85.0
    delay_us: float = 10.0
    burst_ms: float = 1.5
    p_fire: float = 1.0

    def __post_init__(self):
        if self.type not in JAMMER_TYPES:
            raise ValueError(f"jammer {self.id}: unknown type {self.type!r}")


@dataclass
class ChannelModel:
    path_loss: str = "log_distance"        # friis | log_distance
    exponent: float = 2.4                  # LOS drone mesh, 2.4 GHz
    ref_loss_db: float = 40.0              # at 1 m, 2.4 GHz
    fading: str = "none"                   # none | nakagami | rayleigh
    nakagami_m: float = 1.0
    doppler_hz: float = 0.0
    shadowing_db: float = 0.0
    noise_floor_dbm: float = -96.0


@dataclass
class Event:
    t: float
    type: str                              # jammer_on|jammer_off|node_down|node_up|
                                           # load_spike|fade_enter|fade_exit
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Truth:
    """What the verifier is told. The agent NEVER sees this."""
    cause: str
    onset_t: float
    recoverable: bool = True
    notes: str = ""

    def __post_init__(self):
        if self.cause not in CAUSES:
            raise ValueError(f"truth.cause must be one of {CAUSES}, got {self.cause!r}")


@dataclass
class Scenario:
    name: str
    family: str
    truth: Truth
    nodes: list[Node]
    seed: int = 1
    duration_s: float = 60.0
    n_channels: int = 8
    channel: int = 6
    routing: str = "olsr"
    lora_available: bool = True
    lora_jammed: bool = False
    mobility: list[Mobility] = field(default_factory=list)
    traffic: list[Flow] = field(default_factory=list)
    jammers: list[Jammer] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    channel_model: ChannelModel = field(default_factory=ChannelModel)
    links: str | list[list[str]] = "auto"   # "auto" (range-based) or explicit pairs
    comm_range_m: float = 220.0
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------- derived helpers ---------------- #
    @property
    def agent_node(self) -> Node:
        for n in self.nodes:
            if n.role == "agent":
                return n
        raise ValueError(f"scenario {self.name}: no node with role 'agent'")

    @property
    def peers(self) -> list[Node]:
        return [n for n in self.nodes if n.role == "peer"]

    def node(self, nid: str) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def adjacency(self) -> list[tuple[str, str]]:
        """Explicit link list, or range-based if links == 'auto'."""
        if isinstance(self.links, list):
            return [(a, b) for a, b in self.links]
        out = []
        comm = [n for n in self.nodes if n.role in ("agent", "peer")]
        for i, a in enumerate(comm):
            for b in comm[i + 1:]:
                d = math.dist(a.pos, b.pos)
                if d <= self.comm_range_m:
                    out.append((a.id, b.id))
        return out

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the scenario is runnable."""
        errs: list[str] = []
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            errs.append("duplicate node ids")
        try:
            self.agent_node
        except ValueError as e:
            errs.append(str(e))
        if not (1 <= self.channel <= self.n_channels):
            errs.append(f"channel {self.channel} outside 1..{self.n_channels}")
        for j in self.jammers:
            for ch in j.channels:
                if not (1 <= ch <= self.n_channels):
                    errs.append(f"jammer {j.id}: channel {ch} outside 1..{self.n_channels}")
        for f in self.traffic:
            for nid in (f.src, f.dst):
                if nid not in ids:
                    errs.append(f"flow references unknown node {nid!r}")
        for m in self.mobility:
            if m.node not in ids:
                errs.append(f"mobility references unknown node {m.node!r}")
        for e in self.events:
            if e.target and e.target not in ids + [j.id for j in self.jammers]:
                errs.append(f"event at t={e.t} references unknown target {e.target!r}")
        if self.truth.onset_t >= self.duration_s:
            errs.append("truth.onset_t is after the end of the episode")
        # connectivity sanity: agent must have at least one link at t=0
        adj = self.adjacency()
        a = self.agent_node.id
        if not any(a in pair for pair in adj):
            errs.append(f"agent node {a} has no links at t=0 (comm_range too small?)")
        return errs

    # ---------------- (de)serialisation ---------------- #
    def to_dict(self) -> dict:
        d = asdict(self)
        d["links"] = self.links
        return d

    def dump(self, path: str) -> None:
        d = self.to_dict()
        with open(path, "w") as fh:
            if path.endswith((".yaml", ".yml")) and yaml:
                yaml.safe_dump(d, fh, sort_keys=False, default_flow_style=False)
            else:
                json.dump(d, fh, indent=2)


def _mk(cls, d):
    if isinstance(d, cls):
        return d
    return cls(**d)


def load_scenario(path: str) -> Scenario:
    with open(path) as fh:
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("PyYAML not installed; use a .json scenario")
            raw = yaml.safe_load(fh)
        else:
            raw = json.load(fh)
    return from_dict(raw)


def from_dict(raw: dict) -> Scenario:
    raw = dict(raw)
    raw["nodes"] = [_mk(Node, n) for n in raw.get("nodes", [])]
    raw["mobility"] = [_mk(Mobility, m) for m in raw.get("mobility", [])]
    raw["traffic"] = [_mk(Flow, f) for f in raw.get("traffic", [])]
    raw["jammers"] = [_mk(Jammer, j) for j in raw.get("jammers", [])]
    raw["events"] = [_mk(Event, e) for e in raw.get("events", [])]
    if "channel_model" in raw:
        raw["channel_model"] = _mk(ChannelModel, raw["channel_model"])
    raw["truth"] = _mk(Truth, raw["truth"])
    sc = Scenario(**raw)
    errs = sc.validate()
    if errs:
        raise ValueError(f"invalid scenario {sc.name!r}:\n  - " + "\n  - ".join(errs))
    return sc
