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
import time

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


def _f(v, default: float = 0.0) -> float:
    """Coerce a JSON field to float for the UI feed. A viewer must never raise."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def run(scenario_path, ns3_bin: str, agent_name: str = "baseline",
        out_prefix: str | None = None, verbose: bool = True,
        bundle: str | None = None, agent_obj=None, on_tick=None,
        on_state=None, on_log=None) -> dict:
    """`scenario_path` may be a path or an already-loaded Scenario.

    `on_state(event: dict)` is called on EVERY state message from ns-3 (10 Hz), before
    and between decisions. It exists because `on_tick` alone makes the simulator look
    dead: no decision happens during the 5 s warm-up, and with the LLM teacher the first
    decision then waits on a ~40 s model call. A viewer watching a blank screen for a
    minute cannot tell a running simulation from a hung one. This is the pulse.

    `on_tick(event: dict)` is an optional observer called once per decision tick with the
    live state -- time, delivery, per-channel band, the belief and the call just chosen.
    It exists so a UI can animate a run AS IT HAPPENS instead of replaying a trace
    afterwards. It is passive: it cannot change the decision, and any exception it raises
    is swallowed, because a viewer must never be able to alter or break a scored run.

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

    # FAIL FAST, LOUDLY. A missing or unbuildable ns-3 binary used to show up as the
    # bridge blocking forever on accept() with no message at all -- the caller saw
    # "starting..." and nothing else, indefinitely. Check before launching anything.
    if not ns3_bin:
        raise RuntimeError("no ns-3 binary given (set $NS3_BIN or pass --ns3)")
    if not os.path.exists(ns3_bin):
        raise RuntimeError(f"ns-3 binary not found: {ns3_bin}")
    if not os.access(ns3_bin, os.X_OK):
        raise RuntimeError(f"ns-3 binary is not executable: {ns3_bin}")

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

    # Poll in short slices instead of one long blocking accept, so a simulator that
    # exits immediately (the usual symptom of a broken build) is reported in about a
    # second rather than after the full timeout.
    # DRAIN ns-3's STDOUT. It is a pipe, and until now nobody read it: a chatty run
    # could fill the OS buffer and block the simulator forever with no error anywhere.
    # Draining it on a thread fixes that AND gives the UI the simulator's own log.
    ns3_log: list[str] = []

    def _drain():
        try:
            for ln in iter(proc.stdout.readline, ""):
                ln = ln.rstrip("\n")
                if not ln:
                    continue
                ns3_log.append(ln)
                if len(ns3_log) > 500:
                    ns3_log.pop(0)
                if on_log is not None:
                    try:
                        on_log(ln)
                    except Exception:                                # noqa: BLE001
                        pass
        except Exception:                                            # noqa: BLE001
            pass

    threading.Thread(target=_drain, daemon=True).start()

    srv.settimeout(0.5)
    conn = None
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            conn, _ = srv.accept()
            break
        except socket.timeout:
            if proc.poll() is not None:
                break          # ns-3 is gone; stop waiting for a connection
    if conn is None:
        # ns-3 was started but never connected back. Almost always it died on launch --
        # a broken or half-linked build, a missing dylib, a bad config. Its own output
        # says which, so surface that instead of a bare timeout.
        proc.kill()
        time.sleep(0.3)                     # let the drain thread catch the last lines
        out = "\n".join(ns3_log[-25:])[:1200]
        rc = proc.poll()
        raise RuntimeError(
            f"ns-3 exited without connecting to the bridge (exit code {rc}). "
            f"The binary started but never reached the bridge -- usually a broken build. "
            f"ns-3 said:\n{out.strip() or '(no output)'}") from None
    # Once connected, ns-3 streams state at 10 Hz and only pauses while WE decide, so a
    # long silence means it died mid-run. Without this, readline() blocked forever.
    conn.settimeout(float(os.environ.get("BRIDGE_READ_TIMEOUT_S", "120")))
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
    # EPISODE MEMORY. Without this the agent re-reasons from the current panel on every
    # tick with no idea what it already did: it cannot notice that a remedy failed, and
    # it cannot stop repeating a diagnosis the evidence has already contradicted. The
    # renderers in harness/loop.py and agent/llm_teacher.py have always been able to show
    # `tests_run` and `actions_taken` -- the live bridge simply never filled them in, so
    # every prompt claimed the agent had done nothing.
    tests_run: list[str] = []
    actions_taken: list[str] = []
    hypothesis_history: list[str] = []
    recovery_attempts: list[dict] = []
    pending_effect: dict | None = None      # a remedy whose outcome we have not seen yet
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
        try:
            line = f.readline()
        except socket.timeout:
            proc.kill()
            raise RuntimeError(
                "ns-3 stopped sending state (no message for "
                f"{os.environ.get('BRIDGE_READ_TIMEOUT_S', '120')}s). The simulator "
                "died mid-episode; check its output above.") from None
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

        if on_state is not None:
            try:
                on_state({
                    "t": round(t, 2),
                    "pdr": _f(feats[0]),
                    "channel": int(st.get("channel", 0)),
                    "link_pdr": [_f(x) for x in (st.get("pdr") or [])],
                    "link_hb": [_f(x) for x in (st.get("hb") or [])],
                    "warmup": bool(t < WARMUP_S),
                    "onset_t": ex.onset_t,
                })
            except Exception:                                        # noqa: BLE001
                pass          # a viewer must never break a scored run

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

        # Score the outcome of the previous remedy BEFORE asking again, so the agent is
        # told whether what it did actually helped. This is the whole self-correction
        # loop: "I tried X believing Y, delivery did not improve" is the only signal that
        # can make the next answer different from the last one.
        if pending_effect is not None and t >= pending_effect["t"] + 2.0:
            after = _f(feats[0])
            gain = after - pending_effect["pdr_before"]
            pending_effect["outcome"] = ("helped" if gain >= 0.15 else
                                         "no change" if gain > -0.15 else "made it worse")
            pending_effect["delta_pct"] = round(gain * 50.0, 1)
            recovery_attempts.append(pending_effect)
            pending_effect = None

        ctx = Context(t=t, channel=int(st.get("channel", 6)), n_channels=sc.n_channels,
                      peers=[f"P{i}" for i in range(int(st.get("n_links", 0)))],
                      budget={"hops_used": used["hops"]}, available=avail,
                      last_scan=last_scan_obj, lora_available=sc.lora_available,
                      tests_run=list(tests_run), actions_taken=list(actions_taken),
                      hypothesis_history=list(hypothesis_history),
                      recovery_attempts=list(recovery_attempts))
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

        if d.top and (not hypothesis_history or hypothesis_history[-1] != d.top):
            hypothesis_history.append(d.top)
        if d.call in DIAGNOSE:
            if d.call not in tests_run:
                tests_run.append(d.call)
        elif d.call not in ("no_op", "declare_link_lost"):
            actions_taken.append(f"{d.call}@{t:.0f}s")
            pending_effect = {"t": t, "call": d.call, "believed": d.top,
                              "pdr_before": _f(feats[0])}

        records.append({"t": round(t, 2), "features": list(feats),
                        "call": d.call, "available": list(avail)})

        if on_tick is not None:
            try:
                on_tick({
                    "t": round(t, 2),
                    # feats[0] is pdr_fast as the AGENT sees it (-1..+1), not a raw
                    # field name that may drift; the UI is showing the agent's view.
                    "pdr": _f(feats[0]),
                    "rssi": _f(feats[8]),
                    "channel": int(st.get("channel", 0)),
                    "n_channels": int(sc.n_channels),
                    # The spectrum shown is the SCAN THE AGENT PAID FOR, not the
                    # simulator's God-view. Before the first scan there is nothing to
                    # draw -- which is the point: information has to be bought, and a
                    # viewer should see the band appear at the moment it is.
                    "band": ([ _f(c.noise_dbm) for c in
                               sorted(last_scan_obj.channels, key=lambda c: c.ch) ]
                             if (last_scan_obj and getattr(last_scan_obj, "channels", None))
                             else []),
                    "scan_t": (round(float(last_scan_obj.t), 1)
                               if last_scan_obj is not None else None),
                    "n_links": int(st.get("n_links", 0)),
                    # PER-LINK telemetry, so a topology view can colour each edge
                    # independently instead of showing one aggregate number. This is
                    # what makes hidden_terminal and node_loss visible on screen: one
                    # edge dies while the rest stay healthy.
                    "link_pdr": [_f(x) for x in (st.get("pdr") or [])],
                    "link_rssi": [_f(x) for x in (st.get("rssi") or [])],
                    "link_hb": [_f(x) for x in (st.get("hb") or [])],
                    "belief": dict(d.belief or {}),
                    "top": d.top, "p": round(d.top_p, 3),
                    "call": d.call, "args": dict(d.args or {}),
                    "why": str(getattr(d, "why", ""))[:240],
                    "scans_used": used["scans"], "hops_used": used["hops"],
                    "onset_t": ex.onset_t,
                })
            except Exception:                                        # noqa: BLE001
                pass          # a viewer must never break a scored run
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
