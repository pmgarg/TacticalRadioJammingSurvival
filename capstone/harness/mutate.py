"""Mutation corpus + detection-rate sweep (the brief's robustness deliverable).

A held-out split tests generalisation to *unseen instances of the same families*. That is
not robustness. Robustness is: hold the cause fixed, deform the world along an axis the
agent was never trained on, and watch where accuracy breaks.

Six axes, each with a physical meaning and a signed magnitude, so the output is a CURVE
(accuracy vs deformation) rather than a number:

  power      jammer EIRP        dB        weaker jammer -> harder to see
  onset      attack start time  s         breaks any learned timing prior
  geometry   node positions     m         changes the link budget and the topology
  duty       jammer duty cycle  fraction  intermittent jamming looks like fading
  load       offered traffic    x         changes the congestion baseline
  channel    jammer target      channels  moves the attack off the trained channel
  severity   cause intensity     relative  weakens/strengthens the NON-jammer causes
                                           (fade depth, load spike, path-loss exponent)

Every mutation carries `axis`, `magnitude` and the parent scenario id, so a failure is
always attributable to a specific deformation of a specific case.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from scenario.corpus import make, validate          # noqa: E402
from scenario.schema import Scenario                # noqa: E402

AXES = ("power", "onset", "geometry", "duty", "load", "channel", "severity")

# magnitudes per axis, signed where sign is meaningful
LEVELS = {
    "power":    [-9.0, -6.0, -3.0, 3.0, 6.0],        # dB on jammer EIRP
    "onset":    [-8.0, -4.0, 4.0, 8.0],              # seconds
    "geometry": [10.0, 25.0, 50.0, 80.0],            # metres of position jitter
    "duty":     [0.25, 0.4, 0.6, 0.8],               # absolute duty cycle
    "load":     [0.4, 0.7, 1.5, 2.5],                # multiplier on offered load
    "channel":  [1, 2, 3, 5],                        # channels shifted
    "severity": [0.5, 0.7, 1.3, 1.8],                # multiplier on the cause's intensity
}


def mutate(sc: Scenario, axis: str, mag: float, rng: random.Random) -> Scenario | None:
    m = copy.deepcopy(sc)
    m.name = f"{sc.name}__{axis}{mag:+g}"
    m.seed = (sc.seed * 7919 + hash(axis) % 1000) & 0x7FFFFFFF

    if axis == "power":
        if not m.jammers:
            return None                      # nothing to weaken; not a valid mutation
        for j in m.jammers:
            j.eirp_dbm += mag

    elif axis == "onset":
        lo, hi = 5.0, max(6.0, m.duration_s - 10.0)
        for ev in getattr(m, "events", []) or []:
            if hasattr(ev, "t"):
                ev.t = min(hi, max(lo, ev.t + mag))
        if m.truth.onset_t is not None:
            m.truth.onset_t = min(hi, max(lo, m.truth.onset_t + mag))

    elif axis == "geometry":
        for n in m.nodes:
            if n.role == "agent":
                continue                     # move the world, not the observer
            n.pos = [n.pos[0] + rng.uniform(-mag, mag),
                     n.pos[1] + rng.uniform(-mag, mag),
                     max(5.0, n.pos[2] + rng.uniform(-mag / 4, mag / 4))]

    elif axis == "duty":
        if not m.jammers:
            return None
        for j in m.jammers:
            j.duty = float(mag)

    elif axis == "load":
        if not m.traffic:
            return None
        for fl in m.traffic:
            fl.rate_kbps = max(10.0, fl.rate_kbps * mag)

    elif axis == "channel":
        if not m.jammers:
            return None
        shifted = False
        for j in m.jammers:
            if j.channels:
                j.channels = [((c - 1 + int(mag)) % m.n_channels) + 1 for c in j.channels]
                shifted = True
        if not shifted:
            return None                      # barrage covers everything; shift is a no-op
    elif axis == "severity":
        # The non-jammer causes need a deformation axis of their own, or `fading` -- the
        # family the whole false-positive result rests on -- gets almost no mutations.
        touched = False
        for ev in getattr(m, "events", []) or []:
            pr = getattr(ev, "params", None) or {}
            if ev.type == "fade_enter":
                if "depth_db" in pr:
                    pr["depth_db"] = max(3.0, pr["depth_db"] * mag)
                    touched = True
                if "nakagami_m" in pr:            # >1 shrinks the fade (Rician-like)
                    pr["nakagami_m"] = max(0.6, pr["nakagami_m"] / max(0.3, mag))
            elif ev.type == "load_spike" and "factor" in pr:
                pr["factor"] = max(1.2, pr["factor"] * mag)
                touched = True
        if not touched and m.channel_model is not None:
            # node_loss has no intensity knob; deform the link budget instead
            m.channel_model.exponent = max(1.8, min(4.0, m.channel_model.exponent * mag))
            touched = True
        if not touched:
            return None

    else:
        raise ValueError(axis)

    ok, why = validate(m)
    if not ok:
        return None
    m.meta = dict(m.meta or {}); m.meta["mutation"] = {"parent": sc.name, "axis": axis, "magnitude": mag}
    return m


def build(families: list[str], per_family: int, seed0: int, out_dir: str,
          manifest: str) -> dict:
    rng = random.Random(seed0)
    os.makedirs(out_dir, exist_ok=True)
    rows, made, skipped = [], 0, 0
    for fam in families:
        for k in range(per_family):
            base = make(fam, seed0 + k * 13)
            ok, _ = validate(base)
            if not ok:
                continue
            for axis in AXES:
                for mag in LEVELS[axis]:
                    mu = mutate(base, axis, mag, rng)
                    if mu is None:
                        skipped += 1
                        continue
                    path = os.path.join(out_dir, mu.name + ".yaml")
                    mu.dump(path)
                    rows.append({"name": mu.name, "family": fam, "parent": base.name,
                                 "axis": axis, "magnitude": mag,
                                 "true_cause": mu.truth.cause,
                                 "recoverable": mu.truth.recoverable,
                                 "path": os.path.relpath(path, HERE)})
                    made += 1
    with open(manifest, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    by_axis, by_fam = {}, {}
    for r in rows:
        by_axis[r["axis"]] = by_axis.get(r["axis"], 0) + 1
        by_fam[r["family"]] = by_fam.get(r["family"], 0) + 1
    return {"mutations": made, "skipped_invalid": skipped,
            "by_axis": by_axis, "by_family": by_fam, "manifest": manifest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="barrage,spot,reactive,sweep,fading,"
                                          "node_loss,congestion,hidden_term")
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--seed", type=int, default=40000)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "mutations"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "..", "data", "mutations.jsonl"))
    a = ap.parse_args()
    s = build([f for f in a.families.split(",") if f], a.per_family, a.seed,
              a.out, a.manifest)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
