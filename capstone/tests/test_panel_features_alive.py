"""G7: every feature the teacher is SHOWN must actually vary in the corpus it is judged on.

WHY THIS GATE EXISTS
The panel in agent/llm_teacher.py is a promise. A row reading

    scan_periodicity   hot channel keeps moving (sweep -> high)

tells the model that this number carries the sweep signature. If that number is pinned
at a single constant across all 13,700 corpus windows, the promise is a lie: the model
is being instructed to look for a signal that cannot exist, and it spends reasoning --
and tokens -- on it every single call.

That is not hypothetical. Measured on data/traces_ns3_v2/train.jsonl, ELEVEN of 56
features had zero variance, and two of them (loss_load_corr, scan_periodicity) were on
the panel as the designated discriminators for congestion and sweep -- the two families
the system diagnoses worst. Seven of nine induced rules were dropped for support 0-10,
and three of those died on a conjunct over a dead feature.

The root cause is that percept/features.py substitutes a DEFAULT when telemetry is
absent, so a missing field is indistinguishable from a real measurement. A default is
the right engineering choice for robustness and the wrong one for honesty, so the
honesty has to be enforced here instead.

    python3 tests/test_panel_features_alive.py
    python3 tests/test_panel_features_alive.py --rows ../data/traces_bridge/train.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                        # noqa: E402
from percept.features import FEATURE_NAMES                # noqa: E402
from agent.llm_teacher import PANEL                       # noqa: E402

# Known-dead and ACCEPTED as such, with the reason. A feature may live here only if the
# world genuinely cannot produce it -- not because it is inconvenient that it is dead.
ACCEPTED_DEAD = {
    "own_speed": "no scenario moves the node; mobility is out of scope for this corpus",
    "displacement": "as own_speed",
    "pdr_motion_corr": "as own_speed",
    "hops_used": "bookkeeping, not evidence (see NOT_EVIDENCE in analyse_failures.py)",
    "scans_used": "bookkeeping",
    "actions_used": "bookkeeping",
    "t_in_episode": "bookkeeping",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="../data/traces_ns3_v2/train.jsonl")
    ap.add_argument("--min-distinct", type=int, default=3)
    a = ap.parse_args()

    # paths are given relative to capstone/ (how every other gate is invoked), not
    # relative to tests/ -- resolving against __file__ silently skipped the gate.
    path = a.rows if os.path.isabs(a.rows) else os.path.normpath(
        os.path.join(HERE, a.rows))
    if not os.path.exists(path):
        print(f"SKIP: {a.rows} not present")
        return 0

    X = []
    for line in open(path):
        r = json.loads(line)
        if r.get("features"):
            X.append(r["features"])
    X = np.asarray(X, float)
    if len(X) < 100:
        print(f"SKIP: only {len(X)} rows")
        return 0

    panel = {n: i for n, i, _d in PANEL}
    dead_on_panel, dead_off_panel = [], []
    for name, idx in panel.items():
        col = X[:, idx]
        distinct = len(np.unique(np.round(col, 4)))
        if distinct < a.min_distinct and name not in ACCEPTED_DEAD:
            dead_on_panel.append((name, idx, float(col.mean()), distinct))
    for i, name in enumerate(FEATURE_NAMES):
        if name in panel or name in ACCEPTED_DEAD:
            continue
        col = X[:, i]
        if len(np.unique(np.round(col, 4))) < a.min_distinct:
            dead_off_panel.append((name, i, float(col.mean())))

    print("=" * 72)
    print(f"G7 panel honesty — {len(X)} rows from {os.path.basename(path)}")
    print("=" * 72)
    if dead_off_panel:
        print(f"\n  note: {len(dead_off_panel)} feature(s) are constant but NOT on the panel")
        print("        (harmless to the teacher; still dead weight for the student)")
        for n, i, m in dead_off_panel:
            print(f"          [{i:>2}] {n:<22} constant at {m:+.3f}")
    if dead_on_panel:
        print(f"\n  [FAIL] {len(dead_on_panel)} PANEL feature(s) are constant in this corpus.")
        print("         The teacher is told these carry a diagnosis. They cannot.")
        for n, i, m, d in dead_on_panel:
            desc = next(dd for nn, _ii, dd in PANEL if nn == n)
            print(f"          [{i:>2}] {n:<22} constant at {m:+.3f}  ({d} distinct)")
            print(f"               panel claims: \"{desc}\"")
        print("\n         Fix one of three ways, in this order of preference:")
        print("           1. make the world emit it (the signal is missing, not the feature)")
        print("           2. drop the row from PANEL (stop paying tokens to lie)")
        print("           3. add it to ACCEPTED_DEAD with a reason the world cannot produce it")
        print("=" * 72)
        return 1
    print("\n  [PASS] every panel feature varies in this corpus")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
