"""Policy induction: the LLM proposes rules, the corpus decides which survive.

This is the piece the brief means by "the LLM helps define the tools, rules and policies
for the student". It is deliberately NOT "ask the model for rules and write them down".
The loop is:

  1. sample labelled states from the corpus (features + true cause), stratified by family
  2. show the LLM a compact table and ask for candidate rules in the DSL
  3. reject anything that does not parse or references an unknown feature
  4. MEASURE every surviving candidate on a held-out split: support and precision
  5. keep a rule only if precision clears the bar and support is non-trivial
  6. for the fading rows, apply a stricter bar, because a bad rule there is the
     expensive error the whole project exists to avoid

The LLM proposes; the data disposes. A rule that sounds right and measures badly is
discarded, and the discard is recorded so the failure is visible rather than silently
dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from percept.features import FeatureExtractor, FEATURE_NAMES     # noqa: E402
from scenario.corpus import make, validate                        # noqa: E402
from sim.refsim import RefSim                                     # noqa: E402
from gateway.provider import ClaudeCliProvider                    # noqa: E402
from harness.parser import parse_decision                         # noqa: E402
from harness.policy import RuleSet                                # noqa: E402
from harness.loop import UNIT_MAPPED                              # noqa: E402

FAMILIES = ["barrage", "spot", "reactive", "sweep", "fading",
            "node_loss", "congestion", "hidden_term"]

# The features a rule may reference. Restricting this is the main defence against the
# model inventing a rule on a feature that is an artefact rather than physics.
ALLOWED = ["pdr_fast", "pdr_slow", "pdr_spread", "frac_links_degraded",
           "rssi_mean", "rssi_slope", "rssi_pdr_corr", "sinr", "sinr_delta_base",
           "noise_now", "noise_delta_base", "cca_busy", "energy_no_preamble",
           "foreign_fps", "tx_noise_delta", "tx_loss_delta", "retry_ewma",
           "offered_load", "loss_load_corr", "scan_bad_frac", "scan_noise_spread",
           "scan_best_alt_margin", "scan_periodicity", "fade_runlen_mean",
           "outage_duty", "pdr_motion_corr", "heartbeat_gap", "pdr_reverse",
           "link_asymmetry", "frac_peers_report_me_bad", "tx_success_ratio",
           "tx_defer_time", "tx_shadow_loss_delta", "silent_loss_rate"]

PROMPT_HEAD = """You are deriving the decision rules for a small on-device agent that must
work out why a drone's mesh radio link is failing. You are shown labelled examples: each
row is the agent's telemetry at one instant, with the TRUE cause in the first column.

Your job: write rules that separate these causes, in the restricted grammar below. The
rules will be MEASURED on data you have not seen, so a rule that sounds plausible but does
not hold will be discarded. Prefer few, strong, physically-motivated rules over many weak
ones.

GRAMMAR
  {"id": "<short name>",
   "if": [[<feature>, <op>, <number>], ...],      // ALL clauses must hold; op: > >= < <= == !=
   "then": {"hypothesis": "<cause>"},             // and/or {"call": "<tool>"}
   "note": "<the physical reason, one line>"}

Values are normalised to [-1, +1] (-1 low, 0 nominal, +1 high), EXCEPT the columns marked
with % below, which are shown as native percentages but must be written in the rule on the
[-1,+1] scale (so 50% is 0.0, 100% is +1.0, 0% is -1.0).

CAUSES: barrage, spot, reactive, sweep, fading, node_loss, congestion, hidden_term

THE ASYMMETRY THAT MATTERS: declaring jamming when the truth is fading costs 8-10; the
reverse costs about 1. So a rule that concludes a JAMMING cause must rest on positive
evidence of an emitter (raised noise floor, busy energy with no decodable packets, hot
channels in a scan) and never on symptoms alone.

FEATURES YOU MAY USE (anything else is rejected):
"""

PROMPT_TAIL = """
Reply with ONLY a JSON object:
{"rules": [ ... ]}
"""


def rows_from_corpus(path, per_family, families):
    """Labelled (cause, features) rows from an ns-3 trace file instead of a refsim rollout.

    DESIGN 9.2 requires every headline number to come from ns-3, and that has to include
    the evidence the rules are induced from: a rule fitted on refsim thresholds is a rule
    fitted on the wrong world. These rows were produced by train/gen_ns3_corpus.py from the
    authoritative simulator, so the thresholds the LLM proposes are measured against the
    same physics the agent will meet."""
    import collections
    per = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        fam = r.get("family")
        if fam in families and r.get("features"):
            per[fam].append((r["cause"], r["features"]))
    rows = []
    rng = random.Random(0)
    for fam in families:
        pool = per.get(fam, [])
        rng.shuffle(pool)
        rows.extend(pool[:per_family * 4])
    rng.shuffle(rows)
    return rows


def sample_states(families, per_family, seed0, per_episode=4):
    """Labelled (features, true_cause) rows, taken after onset, from valid scenarios."""
    rng = random.Random(seed0)
    rows = []
    for fam in families:
        got, seed = 0, seed0
        while got < per_family and seed < seed0 + 3000:
            sc = make(fam, seed)
            seed += 1
            ok, _ = validate(sc)
            if not ok:
                continue
            got += 1
            sim = RefSim(sc)
            ex = FeatureExtractor(sc.n_channels, dt=sim.dt)
            kept, f, scans = [], None, 0
            onset = sc.truth.onset_t or 0.0
            while sim.t < sc.duration_s:
                o = sim.step()
                # The agent would have scanned by now, so the induction table must
                # contain live spectral columns. Without this the scan features are
                # identically zero in every row and the model cannot use them --
                # which silently removes the only evidence that separates barrage,
                # spot and sweep.
                # TWO scans, 5 s apart -- scan_periodicity is undefined until the second
                # one, and it is the only feature that separates sweep from spot. One
                # scan leaves that column dead and the model has to invent a proxy.
                if scans < 2 and o.t > onset + 2.0 + scans * 5.0:
                    ex.note_scan(sim.scan())
                    scans += 1
                f = ex.update(o)
                if o.t > onset + 8.0:
                    kept.append(list(f))
            if kept:
                step = max(1, len(kept) // per_episode)
                for i in range(0, len(kept), step):
                    rows.append((sc.truth.cause, kept[i]))
    rng.shuffle(rows)
    return rows


def render_table(rows, cols) -> str:
    hdr = f"{'TRUE_CAUSE':<13}" + "".join(f"{c[:11]:>12}" for c in cols)
    out = [hdr, "-" * len(hdr)]
    for cause, f in rows:
        cells = []
        for c in cols:
            i = FEATURE_NAMES.index(c)
            v = f[i]
            cells.append(f"{(v+1)*50:>11.0f}%" if i in UNIT_MAPPED else f"{v:>12.2f}")
        out.append(f"{cause:<13}" + "".join(cells))
    return "\n".join(out)


def measure(rs: RuleSet, rows) -> None:
    """Support and precision of each rule on held-out rows."""
    for r in rs.rules:
        hyp = r.then.get("hypothesis")
        fired = [c for c, f in rows if r.matches(f)]
        r.support = len(fired)
        r.precision = (sum(1 for c in fired if c == hyp) / len(fired)) if fired and hyp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=21000)
    ap.add_argument("--holdout-seed0", type=int, default=31000)
    ap.add_argument("--cols", default=("noise_delta_base,energy_no_preamble,scan_bad_frac,"
                                       "scan_noise_spread,scan_periodicity,rssi_pdr_corr,"
                                       "tx_shadow_loss_delta,silent_loss_rate,"
                                       "loss_load_corr,heartbeat_gap,link_asymmetry,"
                                       "tx_defer_time,fade_runlen_mean,foreign_fps"))
    ap.add_argument("--min-precision", type=float, default=0.75)
    ap.add_argument("--min-precision-jamming", type=float, default=0.90)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--rows", type=int, default=56)
    ap.add_argument("--from-rows", default=None,
                    help="induce from an ns-3 trace jsonl instead of simulating refsim "
                         "(DESIGN 9.2: the rules must be fitted on the authoritative world)")
    ap.add_argument("--holdout-rows", default=None,
                    help="held-out ns-3 trace jsonl used to MEASURE every candidate rule")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "policy_v1.json"))
    a = ap.parse_args()

    cols = [c for c in a.cols.split(",") if c in FEATURE_NAMES]
    print("sampling training rows ...")
    if a.from_rows:
        train = rows_from_corpus(a.from_rows, a.per_family, FAMILIES)[:a.rows]
        print(f"induction table: {len(train)} rows from {a.from_rows} (ns-3)")
    else:
        train = sample_states(FAMILIES, a.per_family, a.seed0)[:a.rows]
    print("sampling held-out rows ...")
    if a.holdout_rows:
        hold = rows_from_corpus(a.holdout_rows, a.per_family * 3, FAMILIES)
        print(f"held-out measurement set: {len(hold)} rows from {a.holdout_rows} (ns-3)")
    else:
        hold = sample_states(FAMILIES, a.per_family, a.holdout_seed0)
    print(f"train rows={len(train)}  holdout rows={len(hold)}\n")

    prompt = (PROMPT_HEAD + "  " + ", ".join(ALLOWED) + "\n\n"
              + "LABELLED EXAMPLES (% columns are native percentages)\n"
              + render_table(train, cols) + "\n" + PROMPT_TAIL)

    p = ClaudeCliProvider(timeout_s=900)
    print("asking the model for candidate rules ...")
    raw = p.complete(prompt)
    out = parse_decision(raw)
    if not out.ok:
        print("FAILED to parse rule proposal:", out.detail)
        print(raw[:800])
        return 1
    cands = out.data.get("rules", [])
    print(f"proposed: {len(cands)} rules ({out.mode})\n")

    rs, rejected = RuleSet.from_dicts(cands, "v1", "claude-cli induction")
    bad_feature = [r for r in rejected]
    # a proposal may only reference the allowed list
    keep = []
    for r in rs.rules:
        off = [c[0] for c in r.clauses if c[0] not in ALLOWED]
        (bad_feature.append({"index": -1, "reason": f"feature not allowed: {off}",
                             "rule": r.id}) if off else keep.append(r))
    rs.rules = keep

    measure(rs, hold)
    JAM = {"barrage", "spot", "reactive", "sweep"}
    accepted, dropped = [], []
    for r in rs.rules:
        bar = a.min_precision_jamming if r.then.get("hypothesis") in JAM else a.min_precision
        if r.support >= a.min_support and r.precision >= bar:
            accepted.append(r)
        else:
            dropped.append({"id": r.id, "support": r.support,
                            "precision": round(r.precision, 3), "bar": bar,
                            "then": r.then})
    final = RuleSet(accepted, "v1", "claude-cli induction, held-out validated")
    final.save(a.out)

    print(final.render())
    print(f"\n{'='*70}")
    print(f"proposed {len(cands)}  |  malformed/disallowed {len(bad_feature)}  "
          f"|  measured {len(rs.rules)}  |  ACCEPTED {len(accepted)}  "
          f"dropped {len(dropped)}")
    for d in dropped:
        print(f"  dropped {d['id']:<16} support={d['support']:<4} "
              f"precision={d['precision']:.2f} < {d['bar']}  -> {d['then']}")
    for b in bad_feature:
        print(f"  rejected {str(b.get('rule'))[:60]}  ({b['reason']})")
    json.dump({"proposed": cands, "accepted": [r.id for r in accepted],
               "dropped": dropped, "rejected": bad_feature},
              open(a.out.replace(".json", "_audit.json"), "w"), indent=1)
    print(f"\nwritten: {os.path.relpath(a.out, HERE)} (+ _audit.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
