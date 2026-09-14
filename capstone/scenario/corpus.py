"""
Randomised scenario corpus generator + validator.

Produces a large, parameter-randomised corpus across all families AND topologies,
split train / val / test by seed, with a HELD-OUT FAMILY that never appears in
training (design §10.3: generalising to a novel attack is the interesting result).

Every scenario is validated before it is allowed into the corpus:
  - schema valid and connected
  - healthy pre-onset baseline (>= 0.85 mean PDR) -- otherwise the metrics are noise
  - a real, detectable change after onset (except the deliberately mild families)

    python3 -m scenario.corpus --n 540 --out ../data/corpus
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

from .schema import (Scenario, Node, Mobility, Flow, Jammer, Event, Truth,
                     ChannelModel)
from . import library as L

FAMILY_LIST = ["barrage", "spot", "reactive", "sweep", "fading",
               "node_loss", "congestion", "hidden_term", "refusal"]
HELD_OUT_FAMILY = "sweep"       # never in train; tests novel-attack generalisation


def _topology(rng: random.Random) -> tuple[list[Node], float]:
    kind = rng.choice(["frozen", "frozen", "frozen", "line", "grid", "ring", "random"])
    if kind == "frozen":
        return L.frozen_topology(), 220.0
    if kind == "line":
        n = rng.randint(4, 7)
        return L.line_topology(n, spacing=rng.uniform(110, 175)), 240.0
    if kind == "grid":
        r, c = rng.randint(2, 3), rng.randint(2, 3)
        return L.grid_topology(r, c, spacing=rng.uniform(120, 170)), 240.0
    if kind == "ring":
        n = rng.randint(5, 8)
        return L.ring_topology(n, radius=rng.uniform(150, 240)), 300.0
    n = rng.randint(5, 9)
    return L.random_topology(n, extent=rng.uniform(220, 330), seed=rng.randint(0, 10**6)), 300.0


def _jammer_pos(nodes: list[Node], rng: random.Random) -> list[float]:
    ax, ay, az = nodes[0].pos
    ang = rng.uniform(0, 2 * math.pi)
    d = rng.uniform(60, 190)
    return [ax + d * math.cos(ang), ay + d * math.sin(ang), az - rng.uniform(0, 12)]


def make(family: str, seed: int) -> Scenario:
    rng = random.Random(seed)
    nodes, comm = _topology(rng)
    nch = 8
    ch = rng.randint(1, nch)
    onset = round(rng.uniform(14.0, 30.0), 1)
    dur = round(rng.uniform(55.0, 75.0), 1)
    cm = ChannelModel(exponent=rng.uniform(2.25, 2.65),
                      ref_loss_db=rng.uniform(38.0, 42.0),
                      shadowing_db=rng.uniform(0.0, 2.0))
    sc = Scenario(name=f"{family}_{seed}", family=family, seed=seed,
                  duration_s=dur, n_channels=nch, channel=ch, nodes=nodes,
                  comm_range_m=comm, traffic=L.default_traffic(nodes),
                  truth=Truth(cause=("barrage" if family == "refusal" else family),
                              onset_t=onset,
                              recoverable=(family != "refusal")),
                  channel_model=cm)
    for f in sc.traffic:
        f.rate_kbps = round(f.rate_kbps * rng.uniform(0.6, 1.5), 1)

    eirp = rng.uniform(14.0, 22.0)
    jp = _jammer_pos(nodes, rng)

    if family == "barrage":
        sc.jammers = [Jammer("J1", "barrage", jp, eirp_dbm=eirp,
                             channels=list(range(1, nch + 1)))]
        sc.events = [Event(onset, "jammer_on", "J1")]
    elif family == "spot":
        sc.jammers = [Jammer("J1", "spot", jp, eirp_dbm=eirp, channels=[ch],
                             duty=rng.choice([1.0, 1.0, 0.7]))]
        sc.events = [Event(onset, "jammer_on", "J1")]
    elif family == "reactive":
        sc.jammers = [Jammer("J1", "reactive", jp, eirp_dbm=eirp,
                             threshold_dbm=-85.0, delay_us=rng.uniform(5, 20),
                             burst_ms=rng.uniform(1.0, 2.5),
                             p_fire=rng.uniform(0.75, 1.0))]
        sc.events = [Event(onset, "jammer_on", "J1")]
    elif family == "sweep":
        sc.jammers = [Jammer("J1", "sweep", jp, eirp_dbm=eirp,
                             channels=list(range(1, nch + 1)),
                             dwell_ms=rng.choice([150.0, 250.0, 400.0, 600.0]))]
        sc.events = [Event(onset, "jammer_on", "J1")]
    elif family == "fading":
        sc.channel_model = ChannelModel(fading="nakagami", nakagami_m=rng.uniform(2.5, 3.5),
                                        doppler_hz=rng.uniform(8, 20),
                                        shadowing_db=rng.uniform(1.0, 3.0),
                                        exponent=cm.exponent, ref_loss_db=cm.ref_loss_db)
        tgt = [nodes[0].pos[0] - rng.uniform(90, 190),
               nodes[0].pos[1] - rng.uniform(30, 110), nodes[0].pos[2]]
        sc.mobility = [Mobility(nodes[0].id, "waypoint", speed=rng.uniform(6, 16),
                                waypoints=[tgt])]
        sc.events = [Event(onset, "fade_enter", nodes[0].id,
                           {"depth_db": rng.uniform(11.0, 17.0), "nakagami_m": 1.0})]
    elif family == "node_loss":
        victim = nodes[1].id
        sc.events = [Event(onset, "node_down", victim)]
    elif family == "congestion":
        for k in (1, 2):
            if len(nodes) > k + 1:
                sc.traffic.append(Flow(src=nodes[k].id, dst=nodes[k + 1].id,
                                       rate_kbps=rng.uniform(1800, 3200),
                                       start_s=onset, type="burst"))
        sc.events = [Event(onset, "load_spike", None, {"factor": rng.uniform(8, 16)})]
    elif family == "hidden_term":
        sc.events = [Event(onset, "load_spike", None,
                           {"factor": rng.uniform(6, 12), "hidden": True})]
    elif family == "refusal":
        sc.lora_jammed = True
        sc.jammers = [
            Jammer("J1", "barrage", jp, eirp_dbm=rng.uniform(24, 30),
                   channels=list(range(1, nch + 1))),
            Jammer("J2", "barrage", _jammer_pos(nodes, rng), eirp_dbm=rng.uniform(24, 30),
                   channels=list(range(1, nch + 1))),
        ]
        sc.events = [Event(onset, "jammer_on", "J1"), Event(onset, "jammer_on", "J2")]
        sc.truth.notes = "unrecoverable"
    return sc


# --------------------------------------------------------------------------- #
def validate(sc: Scenario) -> tuple[bool, str]:
    """A scenario is only usable if it is well posed: healthy before onset, and
    (for the attack families) actually degraded after it."""
    errs = sc.validate()
    if errs:
        return False, errs[0]
    from sim.refsim import RefSim
    sim = RefSim(sc)
    n = int(min(sc.duration_s, 70.0) / sim.dt)
    pre, post = [], []
    for i in range(n):
        o = sim.step()
        p = sum(l.pdr for l in o.links.values()) / max(1, len(o.links))
        (pre if o.t < sc.truth.onset_t - 1 else post).append(p)
    if not pre or not post:
        return False, "no samples"
    a = sum(pre) / len(pre)
    b = sum(post) / len(post)
    if a < 0.85:
        return False, f"unhealthy pre-onset baseline ({a:.2f})"
    if sc.family in ("barrage", "spot", "refusal", "congestion", "hidden_term",
                     "node_loss", "fading") and b > a - 0.12:
        return False, f"no detectable degradation ({a:.2f}->{b:.2f})"
    return True, f"ok pre={a:.2f} post={b:.2f}"


def build(n_target: int, seed0: int = 10000) -> dict[str, list[Scenario]]:
    per = max(1, n_target // len(FAMILY_LIST))
    out, rejected = [], 0
    seed = seed0
    for fam in FAMILY_LIST:
        made = 0
        while made < per and seed < seed0 + 100000:
            sc = make(fam, seed)
            seed += 1
            ok, why = validate(sc)
            if ok:
                out.append(sc)
                made += 1
            else:
                rejected += 1
    # split by seed: 70 / 15 / 15, and the held-out family is test-only
    train, val, test = [], [], []
    hold = os.environ.get("HOLD_OUT_FAMILY", HELD_OUT_FAMILY)
    for sc in out:
        if hold and sc.family == hold:
            test.append(sc)
            continue
        # Seeds are consecutive per family, so seed % 100 is NOT uniform -- it put
        # entire families in one split. Hash the name instead.
        import hashlib
        r = int(hashlib.md5(sc.name.encode()).hexdigest()[:8], 16) % 100
        (train if r < 70 else val if r < 85 else test).append(sc)
    return {"train": train, "val": val, "test": test, "rejected": rejected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=540)
    ap.add_argument("--out", default="../data/corpus")
    a = ap.parse_args()
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sp = build(a.n)
    os.makedirs(a.out, exist_ok=True)
    tot = 0
    for split in ("train", "val", "test"):
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        for sc in sp[split]:
            sc.dump(os.path.join(d, sc.name + ".yaml"))
        tot += len(sp[split])
        fams = {}
        for sc in sp[split]:
            fams[sc.family] = fams.get(sc.family, 0) + 1
        print(f"  {split:6s} n={len(sp[split]):4d}  {dict(sorted(fams.items()))}")
    print(f"total validated scenarios: {tot}   rejected during validation: {sp['rejected']}")
    print(f"held-out family (test only, never trained on): {HELD_OUT_FAMILY}")


if __name__ == "__main__":
    main()
