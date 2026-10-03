"""
ns-3 baseline check: is a corpus scenario actually well posed IN ns-3?

corpus.validate() only runs the reference simulator, so a scenario whose ns-3 baseline is
broken (crash, no link, links dead before onset) or whose attack does nothing in ns-3 passes
validation and then silently poisons the teacher corpus -- the hidden_term bug was exactly
that. This runs the real binary STANDALONE (no agent, no LLM, CPU only) and checks, from
the percept CSV:

  crash        non-zero exit, or the log stops before 80% of the duration
  no_link      no usable link rows
  unhealthy    mean pdr over [5, onset-1] below `--min-pre` (default 0.80)
  no_effect    attack/stress family whose pdr never drops >= `--min-drop` after onset

    python3 -m scenario.ns3_baseline_check --ns3 <jamming-sim> -n 12 --workers 3

Library use (from corpus validation):  check_scenario(sc, ns3_bin) -> (ok, reason, stats)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FAMILIES = ["barrage", "spot", "reactive", "sweep", "fading",
            "node_loss", "congestion", "hidden_term", "refusal"]
# families whose onset must visibly hurt delivery (fading/sweep are gradual or bursty)
EXPECT_DROP = {"barrage", "spot", "refusal", "congestion", "hidden_term", "node_loss",
               "reactive", "sweep", "fading"}
# per-family override of the minimum post-onset drop in mean pdr
MIN_DROP = {"reactive": 0.08, "sweep": 0.08, "fading": 0.08}


def _read_pdr(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                if int(float(r.get("n_links", 0) or 0)) <= 0:
                    continue
                rows.append({"t": float(r["t"]), "pdr": float(r["pdr_mean"]),
                             "rx_err": float(r.get("rx_err") or 0),
                             "defer": float(r.get("tx_defer_ms") or 0),
                             "retry": float(r.get("tx_retry_depth") or 0),
                             "sh_e": float(r.get("shadow_exp") or 0),
                             "sh_m": float(r.get("shadow_miss") or 0)})
            except (KeyError, ValueError):
                continue
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def check_scenario(sc, ns3_bin: str, min_pre: float = 0.80, min_drop: float = 0.12,
                   timeout_s: float = 900.0, keep_dir: str | None = None):
    """Run one scenario standalone in ns-3. Returns (ok, reason, stats)."""
    from sim.export_ns3 import to_config
    with tempfile.TemporaryDirectory(prefix="ns3chk_", dir=keep_dir) as d:
        cfg, out = os.path.join(d, "s.cfg"), os.path.join(d, "s")
        with open(cfg, "w") as fh:
            fh.write(to_config(sc))
        try:
            p = subprocess.run([ns3_bin, f"--config={cfg}", f"--out={out}", "--tdmaSlots=4"],
                               capture_output=True, text=True, timeout=timeout_s)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            return False, "crash", {"detail": "timeout"}
        csv_path = out + ".percept.csv"
        rows = _read_pdr(csv_path) if os.path.exists(csv_path) else []
        dur, onset = sc.duration_s, sc.truth.onset_t
        stats = {"rc": rc, "rows": len(rows), "duration": dur, "onset": onset}
        if rc == 0 and not rows:
            return False, "no_link", stats          # ran to completion, never had a link
        if rc != 0 or rows[-1]["t"] < 0.8 * dur:
            stats["last_t"] = rows[-1]["t"] if rows else 0.0
            return False, "crash", stats
        before = [r for r in rows if 5.0 <= r["t"] < onset - 1.0]
        after = [r for r in rows if r["t"] >= onset + 3.0]
        if not before or not after:
            return False, "no_link", stats

        def avg(rs, k):
            return _mean([r[k] for r in rs])

        a, b = avg(before, "pdr"), avg(after, "pdr")
        worst = min(_mean([r["pdr"] for r in rows if w <= r["t"] < w + 5.0] or [1.0])
                    for w in range(int(onset) + 2, int(dur) - 4))
        stats.update(pre=round(a, 3), post=round(b, 3), drop=round(a - b, 3),
                     worst5s=round(a - worst, 3))
        if a < min_pre:
            return False, "unhealthy", stats
        need = MIN_DROP.get(sc.family, min_drop)
        delivery_hit = max(a - b, a - worst) >= need
        # Congestion, hidden terminals and reactive jamming hurt the MAC, not delivery:
        # retransmissions keep pdr near 1.0 while the stress shows up underneath. So for
        # those families the stress itself is the effect to look for --
        #   congestion / hidden_term: rx_err x2, or tx deferral x3, over the pre-onset level
        #   reactive: retry depth up >= 0.1, or >= 15% of the beacons due while the agent
        #             transmits are lost (measured on reactive_50000: retries 0 -> 0.21,
        #             26% TX-shadow loss, pdr 1.00 -> 0.98)
        pe, qe = avg(before, "rx_err"), avg(after, "rx_err")
        pd, qd = avg(before, "defer"), avg(after, "defer")
        pr, qr = avg(before, "retry"), avg(after, "retry")
        sh_e = sum(r["sh_e"] for r in after)
        sh_loss = sum(r["sh_m"] for r in after) / sh_e if sh_e > 0 else 0.0
        stats.update(rx_err_ratio=round(qe / max(pe, 1.0), 2),
                     defer_ratio=round(qd / max(pd, 0.01), 2),
                     retry_rise=round(qr - pr, 3), shadow_loss=round(sh_loss, 3))
        if sc.family in ("congestion", "hidden_term"):
            mac_hit = (qe >= 2.0 * max(pe, 1.0)) or (qd >= 3.0 * max(pd, 0.01))
        elif sc.family == "reactive":
            mac_hit = (qr - pr >= 0.1) or (sh_loss >= 0.15)
        else:
            mac_hit = False
        if sc.family in EXPECT_DROP and not (delivery_hit or mac_hit):
            return False, "no_effect", stats
        return True, "ok", stats


def _check_one(args):
    sc, ns3_bin = args
    return check_scenario(sc, ns3_bin)


def check_many(scs, ns3_bin: str, workers: int = 3) -> list:
    """check_scenario over many scenarios in parallel; results in input order."""
    if not scs:
        return []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_check_one, [(sc, ns3_bin) for sc in scs]))


def _job(args):
    fam, seed, ns3, min_pre, min_drop = args
    from scenario import corpus
    sc = corpus.make(fam, seed)
    ok, why = corpus.validate(sc)
    if not ok:
        return {"family": fam, "seed": seed, "verdict": "refsim_rejected", "detail": why}
    ok, reason, st = check_scenario(sc, ns3, min_pre, min_drop)
    return {"family": fam, "seed": seed, "verdict": reason, **st}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3", default=os.environ.get("NS3_BIN"))
    ap.add_argument("-n", "--per-family", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-pre", type=float, default=0.80)
    ap.add_argument("--min-drop", type=float, default=0.12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.ns3 or not os.path.exists(a.ns3):
        sys.exit("need --ns3 <path to jamming-sim> (or NS3_BIN)")
    jobs = [(f, a.seed0 + i, a.ns3, a.min_pre, a.min_drop)
            for f in a.families.split(",") for i in range(a.per_family)]
    print(f"{len(jobs)} scenarios, {a.workers} workers", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_job, jobs):
            res.append(r)
            print(f"  {r['family']:<12} seed={r['seed']}  {r['verdict']:<14}"
                  f" pre={r.get('pre', '-')} post={r.get('post', '-')}", flush=True)
    print("\nper-family verdicts (ok / total):")
    summary = {}
    for f in a.families.split(","):
        rs = [r for r in res if r["family"] == f]
        cnt = {}
        for r in rs:
            cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
        summary[f] = cnt
        bad = ", ".join(f"{k}={v}" for k, v in sorted(cnt.items()) if k != "ok")
        print(f"  {f:<12} {cnt.get('ok', 0):>2}/{len(rs):<2}  {bad}")
    tot_ok = sum(1 for r in res if r["verdict"] == "ok")
    print(f"\nTOTAL ok {tot_ok}/{len(res)}")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"summary": summary, "results": res}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
