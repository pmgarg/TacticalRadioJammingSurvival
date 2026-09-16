#!/usr/bin/env python3
"""
G1 (ns-3 half) -- prove the AUDIT F3 fixes took effect in the real simulator.

Reads the CSVs written by `bash sim_ns3/build_and_check.sh` and asserts, per family,
the observable consequence of each fix. Nothing here trusts the source code; every
check is a statement about measured telemetry.

This is the gate that catches the class of bug that hid in this project for weeks:
a jammer on the wrong channel, a counter that is always zero, an action that does
nothing, a pre-onset baseline that was never clean. Every one of them is invisible
to a compiler and invisible to accuracy numbers -- they just quietly make a family
unlearnable.

    python3 capstone/tests/test_world_ns3.py
    python3 capstone/tests/test_world_ns3.py --dir ../data/ns3_check
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(HERE, "..", "data", "ns3_check")

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    if cond is None:
        SKIP.append(name)
        print(f"  [SKIP] {name}" + (f"  -- {detail}" if detail else ""))
        return
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(rows, col, lo=None, hi=None):
    """Column as floats, optionally restricted to a time window."""
    out = []
    for r in rows:
        try:
            t = float(r["t"])
        except (KeyError, ValueError):
            continue
        if lo is not None and t < lo:
            continue
        if hi is not None and t > hi:
            continue
        try:
            out.append(float(r[col]))
        except (KeyError, ValueError, TypeError):
            pass
    return out


def mean(xs, default=None):
    return statistics.fmean(xs) if xs else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--onset", type=float, default=18.0,
                    help="nominal attack onset in the shipped scenarios")
    a = ap.parse_args()
    d = os.path.abspath(a.dir)

    found = sorted(glob.glob(os.path.join(d, "*.percept.csv")))
    print("=" * 76)
    print("G1(ns-3)  WORLD-CONTRACT GATE -- did the AUDIT F3 fixes take effect?")
    print(f"          reading {len(found)} percept CSVs from {d}")
    print("=" * 76)
    if not found:
        print("\n  No CSVs. Run:  bash sim_ns3/build_and_check.sh")
        sys.exit(2)

    fams = {os.path.basename(p)[:-len(".percept.csv")]: load(p) for p in found}
    for k, v in fams.items():
        print(f"    {k:32s} {len(v):5d} rows")

    def rows(name):
        return fams.get(name)

    # ---- F3a: the reactive jammer ------------------------------------------
    print("\n=== F3a  reactive jammer: right channel, right time, right length ===")
    r = rows("reactive_on_tx")
    if r is None:
        check("reactive family present", None, "reactive_on_tx.percept.csv missing")
    else:
        pre = mean(num(r, "meas_floor_dbm", hi=a.onset - 2))
        post = mean(num(r, "meas_floor_dbm", lo=a.onset + 2))
        # (b) armed at onset, not from t=0: the pre-onset floor must be quiet.
        check("pre-onset noise floor is quiet (jammer NOT armed from t=0)",
              pre is not None and pre < -90.0,
              f"pre={pre:.1f} dBm (was armed during main() before the fix)")
        # (a) A reactive jammer is INVISIBLE to the min-hold noise floor by construction:
        # min-hold takes the minimum over the sample window, so a 1.5 ms burst every
        # ~17 ms cannot lift it. That is not a failure, it is what min-hold is for -- and
        # it is precisely why the design identifies reactive with S4' (below) and not S2.
        check("reactive does NOT lift the min-hold floor (expected: S2 is blind to bursts)",
              pre is not None and post is not None and abs(post - pre) < 3.0,
              f"delta={post - pre:+.1f} dB -- so S4', not S2, must carry this family")
        pdr_pre = mean(num(r, "pdr_mean", hi=a.onset - 2))
        pdr_post = mean(num(r, "pdr_mean", lo=a.onset + 2))
        check("delivery degrades after onset",
              pdr_pre is not None and pdr_post is not None and pdr_post < pdr_pre,
              f"pdr {pdr_pre:.2f} -> {pdr_post:.2f}")
        # (c) burst length: a reactive jammer must NOT look like a barrage. Its duty
        # is low, so the post-onset floor should sit well below the barrage floor.
        b = rows("barrage_all_channels")
        if b:
            bpost = mean(num(b, "meas_floor_dbm", lo=a.onset + 2))
            check("reactive floor is well below barrage (it is NOT a de-facto barrage)",
                  bpost is not None and post is not None and post < bpost - 10.0,
                  f"reactive={post:.1f} barrage={bpost:.1f} dBm")

    # ---- F3b: TX-shadow counters -------------------------------------------
    print("\n=== F3b  TX-shadow counters are populated ===")
    for fam in ("reactive_on_tx", "barrage_all_channels", "fading_no_attacker"):
        r = rows(fam)
        if r is None:
            continue
        se = sum(num(r, "shadow_exp"))
        sm = sum(num(r, "silent_exp"))
        check(f"{fam}: shadow/silent expectations are non-zero",
              se > 0 and sm > 0, f"shadow_exp={se:.0f} silent_exp={sm:.0f}")
    def s4(rr):
        e, m = sum(num(rr, "shadow_exp", lo=a.onset)), sum(num(rr, "shadow_miss", lo=a.onset))
        e2, m2 = sum(num(rr, "silent_exp", lo=a.onset)), sum(num(rr, "silent_miss", lo=a.onset))
        if e <= 0 or e2 <= 0:
            return None
        return (m / e) - (m2 / e2)

    scores = {f: s4(v) for f, v in fams.items()}
    for f, v in sorted(scores.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        print(f"    S4' {f:30s} {'n/a' if v is None else f'{v:+.3f}'}")
    sr = scores.get("reactive_on_tx")
    others = [v for f, v in scores.items() if v is not None and f != "reactive_on_tx"]
    check("S4' is HIGHEST for reactive -- the discriminator works",
          None if (sr is None or not others) else sr > max(others),
          f"reactive={sr:+.3f} vs best other={max(others):+.3f}" if sr is not None and others
          else "not enough samples")

    # ---- F3d: flow.0.start honoured ----------------------------------------
    print("\n=== F3d  the mission flow honours flow.0.start ===")
    r = rows("hidden_terminal")
    if r is None:
        check("hidden_terminal present", None, "csv missing")
    else:
        tx_pre = sum(num(r, "data_tx", hi=a.onset - 2))
        tx_post = sum(num(r, "data_tx", lo=a.onset + 2))
        if tx_pre == 0 and tx_post == 0:
            check("no mission data before flow.0.start (clean pre-onset baseline)", None,
                  "data_tx is 0 throughout: it is written from apps[0]->DataSent() "
                  "regardless of which node the flow source actually is (AUDIT item j). "
                  "Fix that before this check can mean anything.")
        else:
            check("no mission data before flow.0.start (clean pre-onset baseline)",
                  tx_pre == 0,
                  f"data_tx pre={tx_pre:.0f} post={tx_post:.0f}")

    # ---- F3c: per-link PDR is no longer pinned at 1.0 ----------------------
    print("\n=== F3c  per-link PDR is real, not clamped by DATA counted as beacons ===")
    for fam in ("hidden_terminal", "congestion_burst"):
        r = rows(fam)
        if r is None:
            continue
        vals = []
        for row in r:
            s = (row.get("pdr_per_link") or "").replace("|", " ").replace(";", " ")
            for tok in s.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        if not vals:
            check(f"{fam}: per-link PDR parsed", None, "could not parse pdr_per_link")
            continue
        frac1 = sum(1 for v in vals if v >= 0.999) / len(vals)
        check(f"{fam}: per-link PDR is not pinned at 1.0",
              frac1 < 0.95, f"{frac1:.0%} of samples were exactly 1.0 (n={len(vals)})")

    # ---- TDMA actually on --------------------------------------------------
    print("\n=== F3  TDMA is enabled (change_tdma_slot is not a no-op) ===")
    r = rows("congestion_burst")
    if r:
        att = sum(num(r, "tx_attempts"))
        check("tx_attempts recorded with --tdmaSlots=4",
              att > 0, f"tx_attempts total={att:.0f}")

    # ---- the discrimination that everything rests on -----------------------
    print("\n=== S2  noise floor separates attacker from no-attacker ===")
    base = {}
    for fam in ("barrage_all_channels", "spot_single_channel", "sweeping_jammer",
                "fading_no_attacker", "dead_peer"):
        r = rows(fam)
        if not r:
            continue
        pre = mean(num(r, "meas_floor_dbm", hi=a.onset - 2))
        post = mean(num(r, "meas_floor_dbm", lo=a.onset + 2))
        if pre is None or post is None:
            continue
        base[fam] = post - pre
        print(f"    {fam:28s} delta = {post - pre:+6.1f} dB")
    for fam in ("barrage_all_channels", "spot_single_channel"):
        if fam in base:
            check(f"{fam}: noise floor rises (attacker present)", base[fam] > 3.0,
                  f"{base[fam]:+.1f} dB")
    for fam in ("fading_no_attacker", "dead_peer"):
        if fam in base:
            check(f"{fam}: noise floor does NOT rise (no attacker) -- the FP defence",
                  abs(base[fam]) < 3.0, f"{base[fam]:+.1f} dB")

    # ---- F3f: is the jammer's energy on the channel it was configured for? -------
    print("\n=== F3f  does the radiated block land on the CONFIGURED channel? ===")
    r = rows("spot_single_channel")
    cfgp = os.path.join(d, "spot_single_channel.cfg")
    if r and os.path.exists(cfgp):
        cfg = dict(l.split("=", 1) for l in open(cfgp).read().split("\n") if "=" in l)
        want = int(float(cfg.get("jam.0.ch.0", "0")))
        post = [row for row in r if float(row["t"]) > a.onset + 5]
        bp = [float(x) for x in (post[0].get("band_power") or "").split()] if post else []
        got = (max(range(len(bp)), key=lambda i: bp[i]) + 1) if bp else None
        check(f"spot jammer configured on ch {want} radiates hottest on ch {want}",
              got == want,
              f"hottest channel is {got}, configured {want}. "
              f"SpectrumValue5MhzFactory centres its block at 2412+5*ch MHz while channel "
              f"ch is at 2407+5*ch -- exactly one channel high. Fix needs the corpus "
              f"regenerated, so do it together, not before a demo.")

    print("\n" + "=" * 76)
    print(f"WORLD GATE: {len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("FAILED:", FAIL)
    print("=" * 76)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
