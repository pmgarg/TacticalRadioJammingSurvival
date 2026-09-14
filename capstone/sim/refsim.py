"""
Reference channel simulator.

Implements the same physics ns-3 will produce (path loss, correlated Rayleigh/
Nakagami fading, jammer PSD raising the noise floor, SINR->PDR, reactive triggering,
congestion collisions), but in pure Python at ~10k x real time. Its role (design A2):
a fast approximation used for development, unit tests and bulk training rollouts,
whose behaviour is re-validated against ns-3 — ns-3 remains authoritative for all
reported numbers.

It consumes a Scenario (any topology) and emits a stream of RawObs, plus the
ground-truth timeline the verifier reads. The AGENT NEVER SEES the truth fields.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from percept.features import RawObs, LinkObs, ChannelObs, ScanResult
from scenario.schema import Scenario, Jammer

THERMAL_NOISE_DBM = -96.0
SINR_50 = 8.0          # dB at which PDR = 0.5
SINR_SLOPE = 2.6       # dB per logit
C_LIGHT = 3e8
FREQ_HZ = 2.437e9


def _path_loss_db(d: float, model: str, exponent: float, ref_loss: float) -> float:
    d = max(1.0, d)
    if model == "friis":
        return 20 * math.log10(4 * math.pi * d * FREQ_HZ / C_LIGHT)
    return ref_loss + 10.0 * exponent * math.log10(d)


# 802.11 adjacent-channel leakage, indexed by |channel offset| (5 MHz spacing,
# 20 MHz occupied bandwidth). Matches the transmit-mask shape ns-3's
# SpectrumValue5MhzFactory produces.
ADJ_LEAK_DB = {0: 0.0, 1: -3.0, 2: -8.0, 3: -15.0, 4: -28.0, 5: -40.0}


def _leak_db(offset: int) -> float:
    return ADJ_LEAK_DB.get(abs(offset), -50.0)


def _lin(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0)


def _db(lin: float) -> float:
    return 10.0 * math.log10(max(lin, 1e-20))


class _Fader:
    """Time-correlated Rayleigh/Nakagami fading via an AR(1) complex Gaussian."""

    def __init__(self, rng: random.Random, doppler_hz: float, m: float, dt: float):
        self.rng, self.m = rng, max(0.2, m)
        tau_c = 0.4 / doppler_hz if doppler_hz > 0 else 1e9
        self.rho = math.exp(-dt / tau_c) if tau_c < 1e8 else 1.0
        self.i, self.q = rng.gauss(0, 1), rng.gauss(0, 1)

    def sample_db(self) -> float:
        s = math.sqrt(max(0.0, 1.0 - self.rho ** 2))
        self.i = self.rho * self.i + s * self.rng.gauss(0, 1)
        self.q = self.rho * self.q + s * self.rng.gauss(0, 1)
        p = (self.i ** 2 + self.q ** 2) / 2.0            # unit-mean exponential (m=1)
        if self.m != 1.0:                                 # crude Nakagami shaping
            p = p ** (1.0 / self.m)
        return _db(max(p, 1e-6))


@dataclass
class TruthRow:
    t: float
    cause: str
    jammer_on: bool
    jammer_channels: list[int]
    peers_alive: list[str]
    true_pdr: dict[str, float]


@dataclass
class RefSim:
    scenario: Scenario
    dt: float = 0.1

    def __post_init__(self):
        sc = self.scenario
        self.rng = random.Random(sc.seed)
        self.t = 0.0
        self.channel = sc.channel
        self.tx_power = sc.agent_node.tx_power_dbm
        self.agent = sc.agent_node
        self.pos = list(self.agent.pos)
        self.vel = [0.0, 0.0, 0.0]
        self.alive = {n.id: n.alive for n in sc.nodes}
        self.jammer_on = {j.id: False for j in sc.jammers}
        self.jammers = {j.id: j for j in sc.jammers}
        self.fade_extra_db = 0.0
        self.load_factor = 1.0
        self.hidden = False
        self.lora_active = False
        self.declared_lost = False
        self.tdma_slot = 0
        self.faders: dict[str, _Fader] = {}
        cm = sc.channel_model
        for p in sc.peers:
            self.faders[p.id] = _Fader(self.rng, cm.doppler_hz, cm.nakagami_m, self.dt)
        self.neighbors = self._neighbors()
        self.heartbeat_age = {p: 0.0 for p in self.neighbors}
        self.last_rssi = {p: -60.0 for p in self.neighbors}
        self.own_tx_active = False
        self.consec_fail = 0
        # TELEMETRY.md: TX-shadow bookkeeping + reciprocal reports
        self._tx_intervals: list[tuple[float, float]] = []   # recent (start, end)
        self._beacon_period = 0.1                            # peers beacon at 10 Hz
        self._peer_reports: dict[str, tuple[float, float, float]] = {}  # id->(rssi,pdr,t)
        self._last_tx_end = -1e9
        self.truth: list[TruthRow] = []
        self._events = sorted(sc.events, key=lambda e: e.t)
        self._ev_i = 0
        self._pending_reactive_until = -1.0
        self._mobility = {m.node: m for m in sc.mobility}
        self._wp_i = 0

    # ------------------------------------------------------------------ #
    def _neighbors(self) -> list[str]:
        a = self.agent.id
        out = []
        for x, y in self.scenario.adjacency():
            if x == a:
                out.append(y)
            elif y == a:
                out.append(x)
        return out or [p.id for p in self.scenario.peers[:2]]

    def _apply_events(self) -> None:
        while self._ev_i < len(self._events) and self._events[self._ev_i].t <= self.t:
            e = self._events[self._ev_i]
            self._ev_i += 1
            if e.type == "jammer_on" and e.target in self.jammer_on:
                self.jammer_on[e.target] = True
            elif e.type == "jammer_off" and e.target in self.jammer_on:
                self.jammer_on[e.target] = False
            elif e.type == "node_down":
                self.alive[e.target] = False
            elif e.type == "node_up":
                self.alive[e.target] = True
            elif e.type == "fade_enter":
                # Flying into shadow: extra path loss AND the LOS component is lost,
                # so the channel degrades from Rician-like (m>1) to Rayleigh (m=1).
                self.fade_extra_db = float(e.params.get("depth_db", 15.0))
                new_m = float(e.params.get("nakagami_m", 1.0))
                for pid, fd in self.faders.items():
                    fd.m = max(0.2, new_m)
            elif e.type == "fade_exit":
                self.fade_extra_db = 0.0
                for pid, fd in self.faders.items():
                    fd.m = max(0.2, self.scenario.channel_model.nakagami_m)
            elif e.type == "load_spike":
                self.load_factor = float(e.params.get("factor", 8.0))
                self.hidden = bool(e.params.get("hidden", False))

    def _move(self) -> None:
        m = self._mobility.get(self.agent.id)
        if not m:
            return
        if m.type == "constant_velocity":
            self.vel = list(m.velocity)
        elif m.type == "waypoint" and m.waypoints:
            tgt = m.waypoints[min(self._wp_i, len(m.waypoints) - 1)]
            d = [tgt[i] - self.pos[i] for i in range(3)]
            dist = math.sqrt(sum(x * x for x in d))
            if dist < 5.0:
                self._wp_i = min(self._wp_i + 1, len(m.waypoints) - 1)
                self.vel = [0.0, 0.0, 0.0]
            else:
                self.vel = [x / dist * m.speed for x in d]
        for i in range(3):
            self.pos[i] += self.vel[i] * self.dt

    # ------------------------------------------------------------------ #
    def _jammer_power_at_agent(self, ch: int) -> float:
        """Total interference power (dBm) landing on `ch` at the agent, including
        adjacent-channel leakage from jammers centred on nearby channels."""
        cm = self.scenario.channel_model
        tot = 0.0
        for jid, j in self.jammers.items():
            if not self.jammer_on[jid]:
                continue
            gain = self._jammer_leak_lin(j, ch)
            if gain <= 0.0:
                continue
            d = math.dist(self.pos, j.pos)
            pl = _path_loss_db(d, cm.path_loss, cm.exponent, cm.ref_loss_db)
            tot += _lin(j.eirp_dbm - pl) * gain
        return _db(tot) if tot > 0 else -200.0

    def _jammer_power_at_peer(self, peer_id: str) -> float:
        """Interference landing on the current channel at a PEER's location -- what
        determines how well that peer hears US (the reverse link)."""
        cm = self.scenario.channel_model
        try:
            peer = self.scenario.node(peer_id)
        except KeyError:
            return -200.0
        tot = 0.0
        for jid, j in self.jammers.items():
            if not self.jammer_on[jid]:
                continue
            gain = self._jammer_leak_lin(j, self.channel)
            if gain <= 0.0:
                continue
            d = math.dist(peer.pos, j.pos)
            pl = _path_loss_db(d, cm.path_loss, cm.exponent, cm.ref_loss_db)
            tot += _lin(j.eirp_dbm - pl) * gain
        return _db(tot) if tot > 0 else -200.0

    def _jammer_leak_lin(self, j: Jammer, ch: int) -> float:
        """Fraction (linear) of this jammer's EIRP landing on `ch`. 0 if inactive."""
        n = self.scenario.n_channels
        if j.type == "barrage":
            chans = j.channels or list(range(1, n + 1))
            per = 1.0 / max(1, len(chans))
            return sum(per * _lin(_leak_db(c - ch)) for c in chans)
        if j.type == "spot":
            if not j.channels:
                return 0.0
            return sum(_lin(_leak_db(c - ch)) for c in j.channels)
        if j.type == "sweep":
            chans = j.channels or list(range(1, n + 1))
            idx = int((self.t * 1000.0 / max(1.0, j.dwell_ms))) % len(chans)
            return _lin(_leak_db(chans[idx] - ch))
        if j.type == "reactive":
            if self.t > self._pending_reactive_until:
                return 0.0
            chans = j.channels or [self.channel]
            return sum(_lin(_leak_db(c - ch)) for c in chans)
        return 0.0

    def _jammer_hits(self, j: Jammer, ch: int) -> bool:
        if j.type == "barrage":
            return (not j.channels) or ch in j.channels
        if j.type == "spot":
            return ch in j.channels
        if j.type == "sweep":
            chans = j.channels or list(range(1, self.scenario.n_channels + 1))
            idx = int((self.t * 1000.0 / max(1.0, j.dwell_ms))) % len(chans)
            return chans[idx] == ch
        if j.type == "reactive":
            # fires only in the window after the agent transmits
            return self.t <= self._pending_reactive_until and \
                   ((not j.channels) or ch in j.channels or ch == self.channel)
        return False

    def _update_reactive(self) -> None:
        for jid, j in self.jammers.items():
            if j.type != "reactive" or not self.jammer_on[jid]:
                continue
            if self.own_tx_active and self.rng.random() < j.p_fire:
                self._pending_reactive_until = max(
                    self._pending_reactive_until,
                    self.t + j.delay_us * 1e-6 + j.burst_ms * 1e-3)

    # ------------------------------------------------------------------ #
    def _link_state(self, peer_id: str) -> LinkObs:
        sc = self.scenario
        cm = sc.channel_model
        if not self.alive.get(peer_id, True):
            self.heartbeat_age[peer_id] += self.dt
            return LinkObs(pdr=0.0, rssi_dbm=-120.0, retry_rate=1.0, frames_rx=0,
                           heartbeat_age_s=self.heartbeat_age[peer_id], reachable=False)
        peer = sc.node(peer_id)
        d = math.dist(self.pos, peer.pos)
        pl = _path_loss_db(d, cm.path_loss, cm.exponent, cm.ref_loss_db)
        fade = self.faders[peer_id].sample_db() if cm.fading != "none" else 0.0
        shadow = self.rng.gauss(0, cm.shadowing_db) if cm.shadowing_db > 0 else 0.0
        rssi = peer.tx_power_dbm - pl + fade - shadow - self.fade_extra_db

        interf = self._jammer_power_at_agent(self.channel)
        noise_lin = _lin(cm.noise_floor_dbm) + _lin(interf)
        sinr = rssi - _db(noise_lin)

        # congestion / hidden terminal: collisions independent of noise
        coll = 0.0
        if self.load_factor > 1.0:
            offered = min(0.95, 0.06 * self.load_factor)
            coll = offered if not self.hidden else min(0.9, offered * 1.6)
        pdr_sinr = 1.0 / (1.0 + math.exp(-(sinr - SINR_50) / SINR_SLOPE))
        pdr = max(0.0, pdr_sinr * (1.0 - coll))
        if self.lora_active:
            pdr = max(pdr, 0.85 if not self.scenario.lora_jammed else 0.02)

        got = pdr > 0.05 and self.rng.random() < max(pdr, 0.05)
        if got:
            self.heartbeat_age[peer_id] = 0.0
        else:
            self.heartbeat_age[peer_id] += self.dt
        retry = min(1.0, (1.0 - pdr) * (1.3 if coll > 0 else 1.0))
        frames = int(round(pdr * 20 * self.dt * 10))
        if frames > 0:
            # SURVIVOR BIAS. RSSI is only observable on frames that decoded, and a
            # deeply-faded frame does not decode. So the measured RSSI under fading is
            # systematically better than the true mean -- which WEAKENS the S1
            # correlation on real hardware. Overstating S1 here would train the student
            # to lean on a signal ns-3 and the ESP32 cannot give it.
            if cm.fading != "none" and fade < 0.0:
                rssi = rssi - fade * 0.7   # calibrated against ns-3 (see FIDELITY.md)
            self.last_rssi[peer_id] = rssi
        else:
            rssi = self.last_rssi.get(peer_id, rssi)   # stale, like ns-3/hardware
        return LinkObs(pdr=pdr, rssi_dbm=rssi, retry_rate=retry, frames_rx=frames,
                       heartbeat_age_s=self.heartbeat_age[peer_id],
                       reachable=pdr > 0.1)

    # ------------------------------------------------------------------ #
    def scan(self, channels: list[int] | None = None, dwell_ms: float = 20.0) -> ScanResult:
        chans = channels or list(range(1, self.scenario.n_channels + 1))
        out = []
        for ch in chans:
            interf = self._jammer_power_at_agent(ch)
            nl = _lin(self.scenario.channel_model.noise_floor_dbm) + _lin(interf)
            noise = _db(nl)
            jam = interf > -95.0
            busy = min(1.0, 0.05 + (0.9 if jam else 0.0) +
                       (0.5 if self.load_factor > 1 and ch == self.channel else 0.0))
            decod = (25.0 * min(1.0, self.load_factor / 8.0)) if ch == self.channel else 0.0
            if self.load_factor <= 1.0:
                decod = 2.0 if ch == self.channel else 0.0
            out.append(ChannelObs(ch=ch, noise_dbm=noise, busy_frac=busy,
                                  decodable_fps=decod))
        return ScanResult(t=self.t, channels=out)

    def step(self, budget: dict | None = None) -> RawObs:
        self._apply_events()
        self._move()
        # own transmit pattern (duty-cycled by offered load and tdma slot)
        duty = 0.35 if self.hidden else min(0.9, 0.35 * self.load_factor)
        self.own_tx_active = (not self.declared_lost) and (self.rng.random() < duty)
        self._update_reactive()

        links = {p: self._link_state(p) for p in self.neighbors}
        interf = self._jammer_power_at_agent(self.channel)
        noise = _db(_lin(self.scenario.channel_model.noise_floor_dbm) + _lin(interf))
        jam_here = interf > -95.0
        busy = min(1.0, 0.05 + (0.9 if jam_here else 0.0) +
                   (0.55 if self.load_factor > 1 else 0.0))
        decod = 25.0 * min(1.0, self.load_factor / 8.0) if self.load_factor > 1 else 2.0

        pdr_mean = sum(l.pdr for l in links.values()) / max(1, len(links))
        self.consec_fail = 0 if pdr_mean > 0.5 else self.consec_fail + 1

        self.truth.append(TruthRow(
            t=self.t, cause=self.scenario.truth.cause,
            jammer_on=any(self.jammer_on.values()),
            jammer_channels=[c for c in range(1, self.scenario.n_channels + 1)
                             if self._jammer_power_at_agent(c) > -95.0],
            peers_alive=[k for k, v in self.alive.items() if v],
            true_pdr={k: round(v.pdr, 4) for k, v in links.items()}))

        # ---- TX-side KPIs (TELEMETRY.md §3) ----
        # Attempts follow our duty cycle; ACK success follows the link. Deferral is the
        # time CSMA spends waiting for DECODABLE carrier -- high under congestion or a
        # hidden terminal, low under barrage (noisy, but not busy with real packets).
        tx_attempts = 1 if self.own_tx_active else 0
        tx_acked = 1 if (self.own_tx_active and self.rng.random() < pdr_mean) else 0
        decodable_busy = min(0.95, 0.05 + 0.11 * max(0.0, self.load_factor - 1.0))
        if self.hidden:
            decodable_busy = min(0.95, decodable_busy + 0.35)
        defer_ms = (0.06 + 2.2 * decodable_busy * decodable_busy) if tx_attempts else 0.0
        retry_depth = (1.0 - pdr_mean) * 6.0

        # ---- TX-shadow loss (TELEMETRY.md §1) ----
        # Beacons due inside the window after our own TX vs beacons due while silent.
        # A reactive jammer damages only the former.
        n_due = max(1, int(round(self.dt / self._beacon_period))) * max(1, len(links))
        react_extra = 0.0
        for jid, j in self.jammers.items():
            if j.type == "reactive" and self.jammer_on[jid]:
                react_extra = 0.8 * j.p_fire
        # fraction of the tick spent inside the post-TX reaction window
        shadow_share = min(0.85, max(0.05, duty))
        sh_exp = int(round(n_due * shadow_share))
        si_exp = n_due - sh_exp
        base_miss = 1.0 - pdr_mean
        # a reactive jammer damages ONLY beacons arriving in the shadow; everything else
        # (barrage, spot, fading, congestion) hits both buckets equally.
        miss_shadow = min(1.0, base_miss + react_extra * (1.0 - base_miss))
        miss_silent = base_miss
        sh_miss = int(round(sh_exp * miss_shadow))
        si_miss = int(round(si_exp * miss_silent))

        # ---- reciprocal reports (TELEMETRY.md §2) ----
        # Each peer reports how well it hears US. Our TX power and the interference at
        # THEIR location drive it, so it differs from our own view of the link.
        for pid, l in links.items():
            if not self.alive.get(pid, True):
                continue
            rev_rssi = l.rssi_dbm + (self.tx_power - self.agent.tx_power_dbm)
            interf_at_peer = self._jammer_power_at_peer(pid)
            nl = _lin(self.scenario.channel_model.noise_floor_dbm) + _lin(interf_at_peer)
            sinr_rev = rev_rssi - _db(nl)
            rev_pdr = 1.0 / (1.0 + math.exp(-(sinr_rev - SINR_50) / SINR_SLOPE))
            if self.hidden or self.load_factor > 1.0:
                rev_pdr *= (1.0 - min(0.9, 0.06 * self.load_factor))
            if self.rng.random() < 0.25:      # reports arrive with the beacon, not every tick
                self._peer_reports[pid] = (rev_rssi, rev_pdr, self.t)
            rep = self._peer_reports.get(pid)
            if rep:
                l.rssi_reverse_dbm, l.pdr_reverse, l.report_age_s = rep[0], rep[1], self.t - rep[2]

        obs = RawObs(
            t=self.t, channel=self.channel, links=links, noise_dbm=noise,
            cca_busy_frac=busy, decodable_fps=decod,
            own_tx_active=self.own_tx_active, own_tx_duty=duty,
            # A hidden terminal collides at the receiver without us offering more
            # traffic -- our own load must stay low or it reads as congestion.
            offered_load_norm=(0.35 if self.hidden
                               else min(1.0, 0.4 * self.load_factor)),
            queue_occupancy=min(1.0, (1.0 - pdr_mean) * 0.9),
            consecutive_tx_fail=self.consec_fail,
            pos=tuple(self.pos), vel=tuple(self.vel), scan=None,
            budget=budget or {},
            tx_attempts=tx_attempts, tx_acked=tx_acked,
            tx_retry_depth=retry_depth, tx_defer_ms=defer_ms,
            shadow_expected=sh_exp, shadow_missed=sh_miss,
            silent_expected=si_exp, silent_missed=si_miss)
        self.t += self.dt
        return obs

    # -------------------- actuation (called by the controller) ------------- #
    def hop_channel(self, ch: int) -> None:
        self.channel = max(1, min(self.scenario.n_channels, int(ch)))

    def set_tx_power(self, dbm: float) -> None:
        self.tx_power = max(-4.0, min(20.0, float(dbm)))

    def change_tdma_slot(self, slot: int) -> None:
        """Slot separation + duty-cycle backoff: decisively removes contention for
        congestion and hidden-terminal, and denies a reactive jammer its trigger."""
        self.tdma_slot = int(slot) % 8
        self.load_factor = max(1.0, self.load_factor * 0.25)
        self.hidden = False              # transmitters are now separated in time

    def load_test(self, factor: float = 0.5) -> None:
        self.load_factor = max(1.0, self.load_factor * factor)

    def fallback_to_lora(self) -> None:
        self.lora_active = True

    def reroute(self, via=None) -> None:
        """Drop unreachable neighbours from the active set (what OLSR does when a
        peer stops answering) so traffic rides the links that still work."""
        live = [p for p in self.neighbors if self.alive.get(p, True)]
        if live:
            self.neighbors = live

    def move(self, dx: float, dy: float, dz: float) -> None:
        self.pos = [self.pos[0] + dx, self.pos[1] + dy, self.pos[2] + dz]

    def declare_link_lost(self) -> None:
        self.declared_lost = True
