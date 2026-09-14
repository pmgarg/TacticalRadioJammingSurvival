"""
Fidelity gate (design §10.1): does the fast refsim behave like authoritative ns-3?

Runs matched scenarios in both, extracts features with the SAME extractor, and
compares the post-onset marginals of the features the diagnosis actually depends on.
refsim is only allowed to be a training accelerator if it agrees with ns-3 here.
"""
from __future__ import annotations
import os, sys, csv, math, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenario.library import FAMILIES
from scenario.schema import load_scenario
from sim.refsim import RefSim
from sim.ns3_adapter import features_from_ns3
from percept.features import FeatureExtractor, FEATURE_NAMES

KEY = {"pdr_fast":0, "frac_links_degraded":6, "rssi_pdr_corr":12,
       "noise_delta_base":17, "energy_no_preamble":20, "scan_bad_frac":31,
       "scan_noise_spread":32, "pdr_spread":5}

def refsim_features(sc, n=600):
    sim = RefSim(sc); ex = FeatureExtractor(sc.n_channels, dt=sim.dt)
    out=[]
    for i in range(n):
        o = sim.step()
        if i % 20 == 0: o.scan = sim.scan()
        out.append(ex.update(o))
    return out

def window(feats, lo, hi, dt=0.1, t0=0.0):
    return [f for i,f in enumerate(feats) if lo <= t0 + i*dt < hi]

def stats(feats, idx):
    v=[f[idx] for f in feats]
    if not v: return (float('nan'), float('nan'))
    m=sum(v)/len(v); s=math.sqrt(sum((x-m)**2 for x in v)/max(1,len(v)-1))
    return (m,s)

def main():
    ns3dir = sys.argv[1] if len(sys.argv)>1 else "/tmp/ns3runs"
    prefix = sys.argv[2] if len(sys.argv)>2 else "h_"
    rows=[]
    print(f"{'family':<22}{'feature':<22}{'ns3(mean)':>10}{'ref(mean)':>10}{'|diff|':>8}  verdict")
    print("-"*82)
    nbad=0; ntot=0
    for fam, mk in FAMILIES.items():
        sc = mk()
        p = os.path.join(ns3dir, f"{prefix}{sc.name}.percept.csv")
        if not os.path.exists(p):
            continue
        nf = features_from_ns3(p, sc.n_channels)
        rf = refsim_features(sc)
        # post-onset window
        nf_w = window(nf, sc.truth.onset_t+4, sc.truth.onset_t+30, t0=float(next(csv.DictReader(open(p)))['t']))
        rf_w = window(rf, sc.truth.onset_t+4, sc.truth.onset_t+30)
        for name, idx in KEY.items():
            a,_ = stats(nf_w, idx); b,_ = stats(rf_w, idx)
            d = abs(a-b) if (a==a and b==b) else float('nan')
            ok = (d == d) and d <= 0.5
            ntot += 1
            if not ok: nbad += 1
            rows.append((fam,name,a,b,d,ok))
            print(f"{fam:<22}{name:<22}{a:>10.2f}{b:>10.2f}{d:>8.2f}  {'ok' if ok else 'DIVERGES'}")
    print("-"*82)
    print(f"agreement: {ntot-nbad}/{ntot} feature-family pairs within 0.5 (normalised units)")
    json.dump([{"family":r[0],"feature":r[1],"ns3":r[2],"refsim":r[3],"absdiff":r[4],"ok":r[5]} for r in rows],
              open("/tmp/xval.json","w"), indent=1)

if __name__=="__main__":
    main()
