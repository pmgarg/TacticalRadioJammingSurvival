"""Measured per-family signatures, rendered in the SAME units the prompt panel uses.

Writing this table by hand was a bug: the panel decodes twenty unit-mapped features back
to native percentages, but a hand-written table quoted them on the [-1,+1] scale. The
model was shown `offered_load 35%` and told to look for `-0.30`, which is the same number
in two notations it had no way to reconcile. So the table is now GENERATED, by running the
simulator and formatting each cell with the identical formatter the panel uses.

    python3 harness/prototypes.py --rebuild     # re-measure and cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from percept.features import FeatureExtractor, FEATURE_NAMES      # noqa: E402
from scenario.corpus import make, validate                        # noqa: E402
from sim.refsim import RefSim                                     # noqa: E402

CACHE = os.path.join(HERE, "..", "data", "prototypes.json")
FAMILIES = ["barrage", "spot", "sweep", "reactive",
            "fading", "node_loss", "congestion", "hidden_term"]

COLS = ["noise_delta_base", "energy_no_preamble", "scan_bad_frac", "scan_noise_spread",
        "scan_periodicity", "tx_shadow_loss_delta", "silent_loss_rate", "retry_ewma",
        "tx_defer_time", "offered_load", "link_asymmetry", "heartbeat_gap",
        "pdr_spread", "rssi_pdr_corr"]

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def measure(n_seeds: int = 4, seed0: int = 51000) -> dict:
    """Late-episode feature means per family, with two spectrum scans performed."""
    out = {}
    for fam in FAMILIES:
        acc, n, seed = [0.0] * len(COLS), 0, seed0
        while n < n_seeds and seed < seed0 + 3000:
            sc = make(fam, seed)
            seed += 1
            if not validate(sc)[0]:
                continue
            sim = RefSim(sc)
            ex = FeatureExtractor(sc.n_channels, dt=sim.dt)
            onset, scans, f = sc.truth.onset_t or 0.0, 0, None
            while sim.t < sc.duration_s:
                o = sim.step()
                if scans < 2 and o.t > onset + 2.0 + scans * 5.0:
                    ex.note_scan(sim.scan())
                    scans += 1
                f = ex.update(o)
            for j, c in enumerate(COLS):
                acc[j] += f[IDX[c]]
            n += 1
        out[fam] = [round(a / max(1, n), 3) for a in acc]
    return {"columns": COLS, "families": out, "n_seeds": n_seeds}


def load(rebuild: bool = False) -> dict:
    if not rebuild and os.path.exists(CACHE):
        return json.load(open(CACHE))
    d = measure()
    os.makedirs(os.path.dirname(os.path.abspath(CACHE)), exist_ok=True)
    json.dump(d, open(CACHE, "w"), indent=1)
    return d


def render(decode) -> str:
    """`decode(idx, value) -> str` must be the panel's own formatter."""
    d = load()
    cols = d["columns"]
    w = 13
    head = f"{'cause':<13}" + "".join(f"{c[:w-1]:>{w}}" for c in cols)
    lines = [head, "-" * len(head)]
    for fam in FAMILIES:
        row = d["families"][fam]
        cells = "".join(f"{decode(IDX[c], v):>{w}}" for c, v in zip(cols, row))
        lines.append(f"{fam:<13}{cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    load(rebuild=a.rebuild)
    from harness.loop import _decode
    print(render(_decode))
    print(f"\ncached: {os.path.relpath(CACHE, HERE)}")
