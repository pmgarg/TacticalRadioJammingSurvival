"""
Scenario library: the frozen 6-node evaluation corpus (the 8 families of §10.1)
plus helpers to build arbitrary meshes on demand.

    python3 -m scenario.library --emit ../scenarios      # write the corpus as YAML
    python3 -m scenario.library --list
"""
from __future__ import annotations

import argparse
import math
import os
import random

from .schema import (Scenario, Node, Mobility, Flow, Jammer, Event, Truth,
                     ChannelModel, CAUSES)

# --------------------------------------------------------------------------- #
# Topologies
# --------------------------------------------------------------------------- #
def frozen_topology() -> list[Node]:
    """The frozen reference mesh (design §5.2): N1 is the drone.

            N2
           /  \
         N1 -- N3
         |      |
         N4 -- N5
           \  /
            N6
    """
    return [
        Node("N1", [0.0,    0.0,   30.0], role="agent"),
        Node("N2", [120.0,  90.0,  30.0]),
        Node("N3", [240.0,  0.0,   30.0]),
        Node("N4", [0.0,   -160.0, 30.0]),
        Node("N5", [240.0, -160.0, 30.0]),
        Node("N6", [120.0, -260.0, 30.0]),
    ]


def line_topology(n: int, spacing: float = 150.0, alt: float = 30.0) -> list[Node]:
    return [Node(f"N{i+1}", [i * spacing, 0.0, alt], role="agent" if i == 0 else "peer")
            for i in range(n)]


def grid_topology(rows: int, cols: int, spacing: float = 150.0, alt: float = 30.0) -> list[Node]:
    nodes = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            k += 1
            nodes.append(Node(f"N{k}", [c * spacing, -r * spacing, alt],
                              role="agent" if k == 1 else "peer"))
    return nodes


def ring_topology(n: int, radius: float = 200.0, alt: float = 30.0) -> list[Node]:
    nodes = []
    for i in range(n):
        a = 2 * math.pi * i / n
        nodes.append(Node(f"N{i+1}", [radius * math.cos(a), radius * math.sin(a), alt],
                          role="agent" if i == 0 else "peer"))
    return nodes


def random_topology(n: int, extent: float = 400.0, seed: int = 0, alt: float = 30.0) -> list[Node]:
    rng = random.Random(seed)
    nodes = [Node("N1", [0.0, 0.0, alt], role="agent")]
    for i in range(1, n):
        nodes.append(Node(f"N{i+1}",
                          [rng.uniform(-extent, extent), rng.uniform(-extent, extent), alt]))
    return nodes


TOPOLOGIES = {
    "frozen": lambda **k: frozen_topology(),
    "line":   lambda n=5, **k: line_topology(n),
    "grid":   lambda rows=3, cols=3, **k: grid_topology(rows, cols),
    "ring":   lambda n=6, **k: ring_topology(n),
    "random": lambda n=8, seed=0, **k: random_topology(n, seed=seed),
}


def default_traffic(nodes: list[Node]) -> list[Flow]:
    """200 kbps from the agent to a node two hops away, plus light background."""
    agent = nodes[0].id
    sink = nodes[min(4, len(nodes) - 1)].id
    flows = [Flow(src=agent, dst=sink, rate_kbps=200.0)]
    if len(nodes) >= 4:
        flows.append(Flow(src=nodes[1].id, dst=nodes[2].id, rate_kbps=20.0))
    return flows


# --------------------------------------------------------------------------- #
# The 8 evaluation families
# --------------------------------------------------------------------------- #
def _base(name: str, family: str, cause: str, onset: float, seed: int,
          nodes=None, recoverable=True, duration=60.0, channel=6) -> Scenario:
    nodes = nodes or frozen_topology()
    return Scenario(
        name=name, family=family, seed=seed, duration_s=duration,
        channel=channel, nodes=nodes, traffic=default_traffic(nodes),
        truth=Truth(cause=cause, onset_t=onset, recoverable=recoverable),
        channel_model=ChannelModel(),
    )


def scn_barrage(seed=1, onset=18.0) -> Scenario:
    s = _base("barrage_all_channels", "barrage", "barrage", onset, seed)
    s.jammers = [Jammer("J1", "barrage", [120.0, -80.0, 25.0], eirp_dbm=18.0,
                        channels=list(range(1, 9)))]
    s.events = [Event(onset, "jammer_on", "J1")]
    s.meta = {"stresses": "classification + escape logic"}
    return s


def scn_spot(seed=2, onset=18.0, target_ch=6) -> Scenario:
    s = _base("spot_single_channel", "spot", "spot", onset, seed, channel=target_ch)
    s.jammers = [Jammer("J1", "spot", [120.0, -80.0, 25.0], eirp_dbm=18.0,
                        channels=[target_ch])]
    s.events = [Event(onset, "jammer_on", "J1")]
    s.meta = {"stresses": "cheapest-test ordering, clean-channel hop"}
    return s


def scn_reactive(seed=3, onset=18.0) -> Scenario:
    s = _base("reactive_on_tx", "reactive", "reactive", onset, seed)
    s.jammers = [Jammer("J1", "reactive", [90.0, -50.0, 25.0], eirp_dbm=16.0,
                        threshold_dbm=-85.0, delay_us=10.0, burst_ms=1.5, p_fire=0.9)]
    s.events = [Event(onset, "jammer_on", "J1")]
    s.meta = {"stresses": "silent_listen test, 'do not hop' policy"}
    return s


def scn_sweep(seed=4, onset=18.0) -> Scenario:
    s = _base("sweeping_jammer", "sweep", "sweep", onset, seed)
    s.jammers = [Jammer("J1", "sweep", [120.0, -80.0, 25.0], eirp_dbm=18.0,
                        channels=list(range(1, 9)), dwell_ms=400.0, sweep_order="linear")]
    s.events = [Event(onset, "jammer_on", "J1")]
    s.meta = {"stresses": "multi-scan reasoning, hop-ahead policy"}
    return s


def scn_fading(seed=5, onset=18.0) -> Scenario:
    """THE FALSE-POSITIVE TRAP: no attacker at all."""
    s = _base("fading_no_attacker", "fading", "fading", onset, seed)
    s.channel_model = ChannelModel(fading="nakagami", nakagami_m=3.0,
                                   doppler_hz=14.0, shadowing_db=2.0, exponent=2.4)
    s.mobility = [Mobility("N1", "waypoint", speed=12.0,
                           waypoints=[[0, 0, 30], [-140, -60, 30], [-260, -30, 30]])]
    s.events = [Event(onset, "fade_enter", "N1",
                      {"depth_db": 14.0, "nakagami_m": 1.0})]
    s.meta = {"stresses": "the expensive false positive — agent must NOT hop"}
    return s


def scn_node_loss(seed=6, onset=18.0) -> Scenario:
    s = _base("dead_peer", "node_loss", "node_loss", onset, seed)
    s.events = [Event(onset, "node_down", "N2")]   # N2 IS a neighbour of N1 (150 m)
    s.meta = {"stresses": "routing-vs-RF separation, neighbour probe"}
    return s


def scn_congestion(seed=7, onset=18.0) -> Scenario:
    s = _base("congestion_burst", "congestion", "congestion", onset, seed)
    s.traffic.append(Flow(src="N2", dst="N5", rate_kbps=2600.0, start_s=onset, type="burst"))
    s.traffic.append(Flow(src="N4", dst="N6", rate_kbps=2200.0, start_s=onset, type="burst"))
    s.events = [Event(onset, "load_spike", None, {"factor": 12.0})]
    s.meta = {"stresses": "load_test, no false 'jamming'"}
    return s


def scn_hidden_terminal(seed=8, onset=18.0) -> Scenario:
    """N4 and N3 can both reach N1 but not each other -> collisions at N1."""
    nodes = [
        Node("N1", [0.0, 0.0, 30.0], role="agent"),
        Node("N2", [120.0, 90.0, 30.0]),
        Node("N3", [200.0, 0.0, 30.0]),
        Node("N4", [-200.0, 0.0, 30.0]),
        Node("N5", [240.0, -160.0, 30.0]),
        Node("N6", [120.0, -260.0, 30.0]),
    ]
    s = _base("hidden_terminal", "hidden_term", "hidden_term", onset, seed, nodes=nodes)
    s.comm_range_m = 260.0          # N3<->N4 are 400 m apart: mutually deaf
    s.traffic = [Flow("N3", "N1", rate_kbps=1200.0, start_s=onset),
                 Flow("N4", "N1", rate_kbps=1200.0, start_s=onset),
                 Flow("N1", "N2", rate_kbps=200.0)]
    s.events = [Event(onset, "load_spike", None, {"hidden": True})]
    s.meta = {"stresses": "collisions without high local load; slot separation"}
    return s


def scn_refusal(seed=9, onset=15.0) -> Scenario:
    """THE REFUSAL CASE: broadband on every channel, LoRa also jammed, no LOS peer.
    Correct behaviour: declare_link_lost() + RTH within the deadline, <= 3 hops."""
    s = _base("refusal_broadband_no_los", "barrage", "barrage", onset, seed,
              recoverable=False)
    s.lora_jammed = True
    s.jammers = [
        Jammer("J1", "barrage", [60.0, -40.0, 25.0], eirp_dbm=26.0,
               channels=list(range(1, 9))),
        Jammer("J2", "barrage", [180.0, -120.0, 25.0], eirp_dbm=26.0,
               channels=list(range(1, 9))),
    ]
    s.events = [Event(onset, "jammer_on", "J1"), Event(onset, "jammer_on", "J2")]
    s.truth.notes = "unrecoverable: no clean channel, no reachable peer, LoRa jammed"
    s.meta = {"stresses": "REFUSAL — must declare_link_lost + RTH, must not hop forever",
              "gate": "pass_fail"}
    return s


FAMILIES = {
    "barrage":      scn_barrage,
    "spot":         scn_spot,
    "reactive":     scn_reactive,
    "sweep":        scn_sweep,
    "fading":       scn_fading,
    "node_loss":    scn_node_loss,
    "congestion":   scn_congestion,
    "hidden_term":  scn_hidden_terminal,
    "refusal":      scn_refusal,
}


def build_corpus(seeds_per_family: int = 1, onset_jitter: bool = True) -> list[Scenario]:
    """The graded corpus. seeds_per_family>1 samples onset/seed for variety."""
    out = []
    for fam, fn in FAMILIES.items():
        for k in range(seeds_per_family):
            rng = random.Random(hash((fam, k)) & 0xFFFF)
            onset = rng.uniform(15.0, 30.0) if (onset_jitter and k > 0) else None
            s = fn(seed=1000 + k) if onset is None else fn(seed=1000 + k, onset=round(onset, 1))
            if k > 0:
                s.name = f"{s.name}__s{k}"
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit the scenario corpus as YAML.")
    ap.add_argument("--emit", metavar="DIR", help="write scenarios into DIR")
    ap.add_argument("--seeds", type=int, default=1, help="seeds per family")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for fam, fn in FAMILIES.items():
            s = fn()
            print(f"{fam:14s} {s.name:28s} cause={s.truth.cause:11s} "
                  f"recoverable={s.truth.recoverable} nodes={len(s.nodes)}")
        return

    if a.emit:
        os.makedirs(a.emit, exist_ok=True)
        n = 0
        for s in build_corpus(a.seeds):
            p = os.path.join(a.emit, f"{s.name}.yaml")
            s.dump(p)
            n += 1
        print(f"wrote {n} scenarios to {a.emit}")


if __name__ == "__main__":
    main()
