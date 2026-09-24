"""
Python side of the AgentBridge: drives a live ns-3 episode at 1 Hz.

ns-3 connects to a UNIX socket, sends one JSON state per decision epoch, and BLOCKS
until we reply with a function call. Because ns-3 is single-threaded, the block stalls
wall-clock time while simulation time stays frozen -- the agent may take as long as it
needs without distorting the experiment.

    python3 -m sim.bridge_server --scenario ../scenarios/spot_single_channel.yaml \
        --ns3 <path-to-jamming-sim-binary> --agent baseline
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.api import Context
from agent.baseline import BaselineAgent
from agent.student import StudentAgent
from agent.api import DIAGNOSE
from agent.controller import LADDER

# mirrors agent/controller.py::_available -- the calls that are always legal
ACT_ALWAYS = ["set_tx_power", "reroute", "change_tdma_slot"]
DIAG_ALWAYS = ["neighbor_probe", "load_test"]
from percept.features import FeatureExtractor, RawObs, LinkObs, ChannelObs, ScanResult
from scenario.schema import load_scenario
from sim.export_ns3 import to_config

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
AGENTS = {"baseline": BaselineAgent, "student": StudentAgent}


def state_to_obs(st: dict, ex_prev_scan_t, t) -> tuple[RawObs, ScanResult | None]:
    pdrs = st.get("pdr", []) or []
    rssis = st.get("rssi", []) or []
    rev_r = st.get("rev_rssi", []) or []
    rev_p = st.get("rev_pdr", []) or []
    rev_a = st.get("rev_age", []) or []
    hbs = st.get("hb", []) or []
    band = st.get("band", []) or []
    links = {}
    for i in range(len(pdrs)):
        pdr = float(pdrs[i])
        rp = float(rev_p[i]) if i < len(rev_p) else -1.0
        links[f"P{i}"] = LinkObs(
            pdr=pdr,
            rssi_dbm=float(rssis[i]) if i < len(rssis) else -120.0,
            retry_rate=float(st.get("retry", min(1.0, 1.0 - pdr))),
            frames_rx=int(round(pdr * 10)),
            heartbeat_age_s=float(hbs[i]) if i < len(hbs) else 0.0,
            reachable=pdr > 0.1,
            rssi_reverse_dbm=(float(rev_r[i]) if i < len(rev_r) else None),
            pdr_reverse=(rp if rp >= 0 else None),
            report_age_s=(float(rev_a[i]) if i < len(rev_a) else 999.0))
    ch = int(st.get("channel", 6))
    floor = band[ch - 1] if 0 < ch <= len(band) else -96.0
    if floor < -199:
        floor = -96.0
    jam = floor > -80.0
    # ROBUSTNESS (capstone): decod used to be 10*mean(pdr_per_link) -- the MESH's own
    # delivery ratio, which degrades under jamming AND congestion alike and so cannot
    # separate them (that was the whole point of this feature). ns-3 now reports the
    # raw decoded-802.11-frame rate from its PHY sniffer (mesh peer or not) as
    # decod_fps. /60 (calibrated against a smoke-test scenario) crushed real corpus
    # congestion traffic (measured late-episode: ~30-225 raw fps) back near the -1
    # floor; /8 keeps jamming pinned at -1 while giving congestion's low end real
    # separation (crosses into positive territory). hidden_term's real corpus traffic
    # (~4.5-7.5 raw fps) is ~10-40x below congestion's and stays low under ANY
    # reasonable linear divisor -- that confusion still needs the tx_defer_time rule,
    # not this feature alone.
    decod = float(st.get("decod_fps", 0.0)) / 8.0
    busy = min(1.0, 0.05 + (0.9 if jam else 0.0) + 0.3 * min(1.0, decod / 20.0))
    scan = None
    if band and (ex_prev_scan_t is None or t - ex_prev_scan_t >= 2.0):
        scan = ScanResult(t=t, channels=[
            ChannelObs(ch=i + 1, noise_dbm=float(band[i]),
                       busy_frac=1.0 if float(band[i]) > -80 else 0.05,
                       decodable_fps=decod if i == ch - 1 else 0.0)
            for i in range(len(band))])
    obs = RawObs(t=t, channel=ch, links=links, noise_dbm=floor,
                 cca_busy_frac=busy, decodable_fps=decod,
                 # ns-3 does NOT provide a TX-conditioned noise measurement: the
                 # spectrum analyzer is gated OFF while we transmit (otherwise it just
                 # measures our own PA). So the S4 statistic is unavailable here and we
                 # must not fabricate it -- we alternate the flag so both of the
                 # extractor's accumulators see the same floor and the delta goes to
                 # zero. Reactive jamming is then found by the silent_listen TEST,
                 # which is exactly why that action exists in the API.
                 own_tx_active=(int(t) % 2 == 0),
                 own_tx_duty=float(st.get("tx_duty", 0.35)),
                 offered_load_norm=float(st.get("load", 0.4)),
                 queue_occupancy=min(1.0, 1.0 - (sum(float(x) for x in pdrs) / len(pdrs) if pdrs else 0)),
                 consecutive_tx_fail=0, pos=(0.0, 0.0, 30.0), vel=(0.0, 0.0, 0.0),
                 scan=scan, budget={"hops_used": int(st.get("hops_used", 0)),
                                    "max_channel_hops": CONTRACT["budgets"]["max_channel_hops"]},
                 tx_attempts=int(st.get("tx_att", 0)), tx_acked=int(st.get("tx_ack", 0)),
                 tx_defer_ms=float(st.get("tx_defer_ms", 0.0)),
                 shadow_expected=int(st.get("sh_e", 0)), shadow_missed=int(st.get("sh_m", 0)),
                 silent_expected=int(st.get("si_e", 0)), silent_missed=int(st.get("si_m", 0)))
    return obs, scan


def run(scenario_path, ns3_bin: str, agent_name: str = "baseline",
        out_prefix: str | None = None, verbose: bool = True,
        bundle: str | None = None, agent_obj=None) -> dict:
    """`scenario_path` may be a path or an already-loaded Scenario.

    `agent_obj` lets a CALLER supply the agent instead of naming one from the registry --
    which is what makes it possible to drive the ns-3 bridge with the LLM teacher (whose
    construction needs a provider and a trace store) rather than only with the agents this
    module happens to know about.
    """
    sc = scenario_path if hasattr(scenario_path, "truth") else load_scenario(scenario_path)
    tmpd = tempfile.mkdtemp(prefix="bridge_")
    cfg = os.path.join(tmpd, "s.cfg")
    open(cfg, "w").write(to_config(sc))
    sock_path = os.path.join(tmpd, "agent.sock")
    out_prefix = out_prefix or os.path.join(tmpd, "run")

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    # AUDIT F3: --tdmaSlots was never passed, so jamming-sim.cc kept its default of 0
    # (CSMA only), MeshNodeApp::SetTdma() was never called and change_tdma_slot() was a
    # no-op in every run -- three of the eight playbook entries depend on it.
    proc = subprocess.Popen(
        [ns3_bin, f"--config={cfg}", f"--out={out_prefix}", f"--sock={sock_path}",
         "--tdmaSlots=4"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    srv.settimeout(60)
    conn, _ = srv.accept()
    f = conn.makefile("rwb")

    agent = agent_obj if agent_obj is not None else (
        StudentAgent(bundle) if agent_name == "student" else AGENTS[agent_name]())
    agent.reset()
    ex = FeatureExtractor(sc.n_channels, dt=0.1)   # percept runs at 10 Hz
    trace, actions, prev_scan_t = [], [], None
    records: list[dict] = []      # (features, available) at each live decision
    pending_scan = False          # a spectrum_scan action is awaiting its result
    last_scan_obj = None
    used = {"hops": 0, "scans": 0, "silent": 0, "moves": 0, "costly": 0}
    tried: dict[tuple, float] = {}
    B = CONTRACT["budgets"]
    TH = CONTRACT["thresholds"]
    no_link_for = 0.0
    onset_t = None
    # Recovery + delivery history, mirroring agent/controller.py so the two produce the
    # same episode_log fields and the one verifier can score both.
    pdr_series: list[list[float]] = []
    baseline_pdr = None
    recovery_hold_start = None
    recovery_t = None
    declared = False
    last_t = 0.0
    WARMUP_S = 5.0     # OLSR convergence + beacon ramp; no diagnosis before this

    while True:
        line = f.readline()
        if not line:
            break
        try:
            st = json.loads(line.decode().strip())
        except Exception:
            continue
        t = float(st.get("t", 0.0))
        # A spectrum scan is an ACTION that costs budget and makes the drone deaf for
        # ~260 ms. The band data happens to be in every state message, but handing the
        # agent a full per-channel view on every 100 ms tick would give it free scans
        # it never paid for -- and it made scan_age permanently "fresh" and the sweep
        # periodicity feature meaningless. The percept layer only learns the spectrum
        # when a spectrum_scan was actually performed.
        obs, scan = state_to_obs(st, None, t)
        obs.scan = None
        if pending_scan:
            ex.note_scan(scan)      # the result of the scan the agent just paid for
            last_scan_obj = scan
            pending_scan = False
        feats = ex.update(obs)

        # The always-available set is taken from the CONTRACT, not retyped here. This
        # list had drifted: it still offered listen_test and transmit_probe after contract
        # 1.2.0 removed them, so the bridge and the controller disagreed about what the
        # agent was allowed to do -- two masks, two truths, and the live path was the one
        # nobody was checking.
        avail = ["no_op"] + [c for c in ACT_ALWAYS if c in CONTRACT["act"]] \
                          + [c for c in DIAG_ALWAYS if c in CONTRACT["diagnose"]]
        if (used["scans"] < B["max_spectrum_scans"]
                and (prev_scan_t is None or (t - prev_scan_t) >= B["min_scan_interval_s"])):
            avail.append("spectrum_scan")
        if used["silent"] < B["max_silent_listens"]:
            avail.append("silent_listen")
        # same hard interlock as the Controller: a hop must be both EVIDENCED (a
        # fresh scan) and USEFUL (this channel is hot, or a much quieter one exists).
        scan_fresh = (prev_scan_t is not None and (t - prev_scan_t) < 10.0)
        hop_useful = False
        if scan_fresh and last_scan_obj and getattr(last_scan_obj, "channels", None):
            ch_now = int(st.get("channel", 6))
            cur = next((c for c in last_scan_obj.channels if c.ch == ch_now), None)
            alts = [c for c in last_scan_obj.channels if c.ch != ch_now]
            if cur and alts:
                best = min(alts, key=lambda c: c.noise_dbm)
                hop_useful = (cur.noise_dbm > TH["jam_energy_dbm"]
                              or (cur.noise_dbm - best.noise_dbm) > 6.0)
        if (used["hops"] < B["max_channel_hops"]
                and used["costly"] < B["max_costly_actions"]
                and scan_fresh and hop_useful):
            avail += ["hop_channel", "channel_hop_probe"]
        if used["moves"] < B["max_moves"] and used["costly"] < B["max_costly_actions"]:
            avail += ["move", "mobility_test"]
        # fallback_to_lora is deliberately NOT offered here. ns-3 models one 2.4 GHz
        # mesh; there is no second radio to fall back to, so offering the call would let
        # the agent "recover" by invoking something the world cannot implement. The refsim
        # controller DOES offer it when the scenario sets lora_available, so barrage
        # outcomes are not comparable between the two worlds -- recorded, not hidden.
        avail.append("declare_link_lost")

        # Percept at 10 Hz, DECISIONS at 1 Hz (design §3). Feed every sample to the
        # feature layer, but only let the agent act on every 10th.
        tick = int(round(t * 10))
        if tick % 10 != 0:
            f.write((json.dumps({"call": "no_op"}) + "\n").encode()); f.flush()
            continue
        # ---- dead-man failsafe (design §7.6, mechanism 3) ----
        pdr_now = (sum(float(x) for x in st.get("pdr", [])) /
                   max(1, len(st.get("pdr", []) or [1])))
        dt_s = max(0.0, t - last_t); last_t = t
        no_link_for = 0.0 if pdr_now > 0.2 else no_link_for + dt_s
        pdr_series.append([round(t, 1), round(pdr_now, 3)])
        if baseline_pdr is None and t > 5.0:
            early = [p for tt, p in pdr_series if tt <= 5.0]
            baseline_pdr = (sum(early) / len(early)) if early else None
        if baseline_pdr and pdr_now >= TH["recovery_pdr_frac_of_baseline"] * baseline_pdr:
            if recovery_hold_start is None:
                recovery_hold_start = t
            elif (t - recovery_hold_start) >= TH["recovery_hold_s"] and recovery_t is None:
                recovery_t = round(t, 2)
        else:
            recovery_hold_start = None
        if onset_t is None and ex.onset_t is not None:
            onset_t = ex.onset_t
        reason = None
        if no_link_for >= TH["no_link_declare_s"]:
            reason = "no_usable_link_timeout"
        elif onset_t is not None and (t - onset_t) >= B["recovery_timeout_s"]:
            reason = "recovery_timeout"
        elif used["costly"] >= B["max_costly_actions"] and no_link_for > 3.0:
            reason = "budget_exhausted"
        if reason and not declared:
            declared = True
            actions.append({"t": round(t, 2), "fn": "declare_link_lost", "args": {},
                            "why": f"controller failsafe: {reason}"})
            if verbose:
                print(f"  t={t:5.1f} FAILSAFE -> declare_link_lost ({reason})")
            f.write((json.dumps({"call": "declare_link_lost"}) + "\n").encode())
            f.flush()
            break

        # ---- anomaly gate (design §7.2) ----
        if t < WARMUP_S or ex.onset_t is None:
            f.write((json.dumps({"call": "no_op"}) + "\n").encode())
            f.flush()
            if verbose and t >= WARMUP_S:
                pass
            continue

        ctx = Context(t=t, channel=int(st.get("channel", 6)), n_channels=sc.n_channels,
                      peers=[f"P{i}" for i in range(int(st.get("n_links", 0)))],
                      budget={"hops_used": used["hops"]}, available=avail,
                      last_scan=last_scan_obj, lora_available=sc.lora_available)
        d = agent.decide(feats, ctx)
        if d.call not in avail:
            d.call, d.args = "no_op", {}
        # do not re-issue a remedy that already failed for this same hypothesis
        # Same fix as agent/controller.py: diagnostics are exempt from the
        # repeat-suppressor. A second scan is evidence, not a repeat.
        key = (d.top, d.call)
        if (d.call not in ("no_op", "declare_link_lost") and d.call not in DIAGNOSE
                and key in tried):
            d.call, d.args = "no_op", {}
        elif d.call != "no_op":
            tried[key] = t
        if d.call in ("hop_channel", "channel_hop_probe"):
            used["hops"] += 1
            used["costly"] += 1
        elif d.call == "spectrum_scan":
            used["scans"] += 1
            prev_scan_t = t
            pending_scan = True     # result arrives on the next state message
        elif d.call == "silent_listen":
            used["silent"] += 1
        elif d.call in ("move", "mobility_test"):
            used["moves"] += 1
            used["costly"] += 2

        records.append({"t": round(t, 2), "features": list(feats),
                        "call": d.call, "available": list(avail)})
        trace.append({"t": round(t, 2), "top": d.top, "p": round(d.top_p, 3),
                      # `declared` and `abstained` MUST be here. verify/verifier.py keys on
                      # them: without `declared` it falls back to argmax, discarding the
                      # minimum-expected-cost decision the agent actually acted on (AUDIT
                      # F4.3), and without `abstained` a refusal to classify is scored as a
                      # claim (F5.1b). agent/controller.py has logged both since those
                      # fixes; the bridge did not, so every ns-3 episode was scored by a
                      # different rule than every refsim episode. One verifier, two inputs,
                      # two truths.
                      "declared": getattr(d, "declared", None),
                      "abstained": bool(getattr(d, "abstained", False)),
                      "belief": {k: round(v, 3) for k, v in d.belief.items()},
                      "unrecoverable": round(d.unrecoverable, 3)})
        if d.call != "no_op":
            actions.append({"t": round(t, 2), "fn": d.call, "args": d.args, "why": d.why})
        if verbose:
            print(f"  t={t:5.1f} links={st.get('n_links')} -> {d.top:<11s} p={d.top_p:.2f} "
                  f"call={d.call}{'' if not d.args else ' ' + json.dumps(d.args)}")

        reply = {"call": d.call}
        reply.update({k: v for k, v in (d.args or {}).items()
                      if isinstance(v, (int, float, str))})
        f.write((json.dumps(reply) + "\n").encode())
        f.flush()
        if d.call == "declare_link_lost":
            break

    try:
        conn.close()
        srv.close()
    except Exception:
        pass
    proc.wait(timeout=120)
    # An episode_log in the SAME shape agent/controller.py produces, so verify/verifier.py
    # scores a live ns-3 episode with exactly the code that scores a refsim one. Two scorers
    # would be two truths.
    hops = sum(1 for a in actions if a.get("fn") in ("hop_channel", "channel_hop_probe"))
    tests = {"spectrum_scan", "neighbor_probe", "silent_listen", "load_test",
             "channel_hop_probe", "mobility_test"}
    declared_lost = next((a["t"] for a in actions if a.get("fn") == "declare_link_lost"), None)
    episode_log = {
        "episode_id": f"{sc.name}#{sc.seed}", "scenario": sc.name,
        "agent": getattr(agent, "name", agent_name),
        "attack_onset_t": onset_t,
        "first_correct_classification_t": None,
        "classification_trace": trace,
        "tests_run": [{"t": a["t"], "test": a["fn"], "args": a.get("args", {}),
                       "why": a.get("why", "")} for a in actions if a.get("fn") in tests],
        "actions": [a for a in actions if a.get("fn") not in tests],
        "recovery_t": recovery_t, "packets_lost": 0.0,
        "actions_consumed": used["costly"], "hops_consumed": hops,
        "tests_before_correct_classification": None,
        "recovered": recovery_t is not None, "survived": None,
        "declared_jamming": any(r.get("top") in ("barrage", "spot", "reactive", "sweep")
                                for r in trace),
        "declared_lost_t": declared_lost, "end_reason": "bridge_end", "pdr_series": pdr_series,
    }
    return {"scenario": sc.name, "truth": sc.truth.cause, "trace": trace,
            "actions": actions, "out_prefix": out_prefix, "records": records,
            "recoverable": sc.truth.recoverable, "family": sc.family,
            "episode_log": episode_log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--ns3", required=True, help="path to the jamming-sim binary")
    ap.add_argument("--agent", default="baseline")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = run(a.scenario, a.ns3, a.agent, a.out, bundle=a.bundle)
    print(f"\n[bridge] scenario={r['scenario']} truth={r['truth']} "
          f"decisions={len(r['trace'])} actions={[x['fn'] for x in r['actions']]}")


if __name__ == "__main__":
    main()
