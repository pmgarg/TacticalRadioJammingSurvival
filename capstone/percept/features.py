"""
L1 percept layer — raw radio counters -> the frozen 48-dim feature vector.

This is the reference implementation. It is deliberately written in plain Python
with fixed-size buffers and no numpy so that it maps 1:1 onto the C99 build that
runs inside ns-3 and on the ESP32 (design §9.5: one implementation, three call
sites). Every feature is annotated with the discrimination statistic (S1..S10,
design §7.4) it serves.

    ex = FeatureExtractor(n_channels=8)
    for obs in stream:            # 10 Hz
        feats = ex.update(obs)    # list[float], len == 48, each in [-1, 1]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .norm import (N_FEATURES, GATE_WARMUP_S, RSSI_OFFSET, RSSI_SCALE, NOISE_OFFSET, NOISE_SCALE,
                   SINR_OFFSET, SINR_SCALE, DBM_DELTA_SCALE, RATE_SCALE,
                   SPEED_SCALE, DIST_SCALE, SLOPE_SCALE, PERCEPT_HZ, TAU_FAST,
                   TAU_SLOW, TAU_BASE, CORR_WINDOW_S, FADE_PDR_THRESH,
                   CUSUM_K, CUSUM_H, JAM_ENERGY_DBM, DEFER_SCALE_MS,
                   NOMINAL_NOISE_DBM, HOT_MARGIN_DB, HOT_PEAK_MARGIN_DB,
                   TX_SHADOW_TAU_MS, clamp, nz, unit)
from .ring import Ring, Ewma, Cusum, RunLength, pearson

_CORR_N = int(CORR_WINDOW_S * PERCEPT_HZ)      # 50 samples
_FAST_N = max(4, int(1.0 * PERCEPT_HZ))        # 10 samples


# --------------------------------------------------------------------------- #
# World -> percept contract
# --------------------------------------------------------------------------- #
@dataclass
class LinkObs:
    pdr: float = 1.0
    rssi_dbm: float = -60.0
    retry_rate: float = 0.0
    frames_rx: int = 0
    heartbeat_age_s: float = 0.0
    reachable: bool = True
    # --- reciprocal report: what THIS peer said about hearing US (TELEMETRY.md §2) ---
    rssi_reverse_dbm: float | None = None   # RSSI they measured for our frames
    pdr_reverse: float | None = None        # their delivery ratio from us
    report_age_s: float = 999.0             # how stale their report is


@dataclass
class ChannelObs:
    ch: int
    noise_dbm: float = -96.0
    busy_frac: float = 0.0
    decodable_fps: float = 0.0


@dataclass
class ScanResult:
    t: float
    channels: list[ChannelObs] = field(default_factory=list)


@dataclass
class RawObs:
    """One 10 Hz sample of everything the radio can tell us. No ground truth."""
    t: float
    channel: int
    links: dict[str, LinkObs] = field(default_factory=dict)
    noise_dbm: float = -96.0
    cca_busy_frac: float = 0.0
    decodable_fps: float = 0.0            # foreign, decodable 802.11 frames/s
    own_tx_active: bool = False
    own_tx_duty: float = 0.0
    offered_load_norm: float = 0.5
    queue_occupancy: float = 0.0
    consecutive_tx_fail: int = 0
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scan: ScanResult | None = None
    budget: dict = field(default_factory=dict)
    # --- TX-side KPIs (TELEMETRY.md §3): measurable even when RX has collapsed ---
    tx_attempts: int = 0            # transmissions started this tick
    tx_acked: int = 0               # of those, acknowledged
    tx_retry_depth: float = 0.0     # mean retries before success/drop
    tx_defer_ms: float = 0.0        # enqueue -> on-air delay (channel occupancy)
    # --- TX-shadow loss (TELEMETRY.md §1): recovers S4 from TIMING ---
    shadow_expected: int = 0        # peer beacons due inside our TX shadow
    shadow_missed: int = 0
    silent_expected: int = 0        # peer beacons due while we were silent
    silent_missed: int = 0


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    def __init__(self, n_channels: int = 8, dt: float = 1.0 / PERCEPT_HZ):
        self.n_channels = n_channels
        self.dt = dt
        self.t0: float | None = None
        self.ticks = 0

        # delivery
        self.pdr_fast = Ewma(TAU_FAST)
        self.pdr_slow = Ewma(TAU_SLOW)
        self.pdr_base = Ewma(TAU_BASE)
        self.cusum = Cusum(CUSUM_K, CUSUM_H)
        self.onset_t: float | None = None
        self._gate_reason: str | None = None
        self.onset_pos: tuple[float, float, float] | None = None
        self.pdr_hist = Ring(_CORR_N)

        # signal
        self.rssi_fast = Ring(_FAST_N)
        self.rssi_hist = Ring(_CORR_N)
        self.rssi_base = Ewma(TAU_BASE)
        self.sinr_base = Ewma(TAU_BASE)

        # interference
        self.noise_fast = Ring(_FAST_N)
        self.noise_base = Ewma(TAU_BASE)
        self.noise_tx = Ewma(TAU_SLOW)        # S4: noise while/just after we TX
        self.noise_silent = Ewma(TAU_SLOW)    # S4: noise while silent
        self.loss_tx = Ewma(TAU_SLOW)
        self.loss_silent = Ewma(TAU_SLOW)

        # mac
        self.retry_ewma = Ewma(TAU_SLOW)
        self.load_hist = Ring(_CORR_N)
        self.loss_hist = Ring(_CORR_N)
        self.max_consec_fail = 0

        # spectral memory
        self.last_scan: ScanResult | None = None
        self.hot_channel_hist: list[int] = []

        # temporal
        self.fade_runs = RunLength(FADE_PDR_THRESH)
        self.outage_ticks = 0

        # self
        self.speed_hist = Ring(_CORR_N)
        self.disp_hist = Ring(_CORR_N)

        # --- TELEMETRY.md additions ---
        self.tx_success = Ewma(TAU_SLOW)      # ACKed / attempted
        self.tx_defer = Ewma(TAU_SLOW)        # enqueue -> on-air (channel occupancy)
        self.tx_retry_depth = Ewma(TAU_SLOW)
        self.shadow_exp = 0                   # beacons due inside our TX shadow
        self.shadow_miss = 0
        self.silent_exp = 0                   # beacons due while we were silent
        self.silent_miss = 0

    # ------------------------------------------------------------------ #
    def note_scan(self, scan: ScanResult) -> None:
        """Record a spectrum scan result (called when a scan action completes)."""
        self.last_scan = scan
        if scan.channels:
            # Only record a hot channel when it is MEANINGFULLY hotter than the rest.
            # Under a barrage jammer every channel sits at the same level, so the argmax
            # is decided by noise and jitters from scan to scan -- which would read as a
            # moving hot channel, i.e. a sweep. Requiring a margin over the median makes
            # the sweep signature specific, and lets two scans be enough instead of three.
            hot = max(scan.channels, key=lambda c: c.noise_dbm)
            lv = sorted(c.noise_dbm for c in scan.channels)
            med = lv[len(lv) // 2]
            if (hot.noise_dbm - med) >= HOT_PEAK_MARGIN_DB:
                self.hot_channel_hist.append(hot.ch)
            if len(self.hot_channel_hist) > 12:
                self.hot_channel_hist.pop(0)

    # ------------------------------------------------------------------ #
    def update(self, o: RawObs) -> list[float]:
        dt = self.dt
        if self.t0 is None:
            self.t0 = o.t
        self.ticks += 1
        if o.scan is not None:
            self.note_scan(o.scan)

        links = list(o.links.values()) or [LinkObs()]
        n_links = len(links)

        # ---------------- delivery ---------------- #
        pdrs = [l.pdr for l in links]
        pdr_mean = sum(pdrs) / n_links
        pdr_worst = min(pdrs)
        pdr_spread = (math.sqrt(sum((p - pdr_mean) ** 2 for p in pdrs) / n_links)
                      if n_links > 1 else 0.0)
        degraded = sum(1 for p in pdrs if p < 0.6) / n_links

        f_fast = self.pdr_fast.update(pdr_mean, dt)
        f_slow = self.pdr_slow.update(pdr_mean, dt)
        base = self.pdr_base.value if self.pdr_base.init else pdr_mean
        fired = self.cusum.update(pdr_mean, base)
        # baseline only tracks NOMINAL conditions (freeze it once anomalous)
        if self.onset_t is None:
            self.pdr_base.update(pdr_mean, dt)
        self.pdr_hist.push(pdr_mean)

        # ---------------- signal ---------------- #
        rssi_vals = [l.rssi_dbm for l in links if l.frames_rx > 0] or \
                    [l.rssi_dbm for l in links]
        rssi_mean = sum(rssi_vals) / len(rssi_vals)
        self.rssi_fast.push(rssi_mean)
        self.rssi_hist.push(rssi_mean)
        frames_rx = sum(l.frames_rx for l in links)
        sinr = rssi_mean - o.noise_dbm
        if self.onset_t is None:
            self.rssi_base.update(rssi_mean, dt)
            self.sinr_base.update(sinr, dt)
            self.noise_base.update(o.noise_dbm, dt)

        # S1 — the false-positive killer
        rssi_pdr_corr = pearson(self.rssi_hist.values(), self.pdr_hist.values())

        # ---------------- interference ---------------- #
        self.noise_fast.push(o.noise_dbm)
        # S4 — split noise/loss by whether we were transmitting
        if o.own_tx_active:
            self.noise_tx.update(o.noise_dbm, dt)
            self.loss_tx.update(1.0 - pdr_mean, dt)
        else:
            self.noise_silent.update(o.noise_dbm, dt)
            self.loss_silent.update(1.0 - pdr_mean, dt)
        tx_noise_delta = ((self.noise_tx.value - self.noise_silent.value)
                          if (self.noise_tx.init and self.noise_silent.init) else 0.0)
        tx_loss_delta = ((self.loss_tx.value - self.loss_silent.value)
                         if (self.loss_tx.init and self.loss_silent.init) else 0.0)
        # S3 — busy time carrying no decodable preamble
        decod_sat = min(1.0, o.decodable_fps / 30.0)
        energy_no_preamble = o.cca_busy_frac * (1.0 - decod_sat)

        # ---------------- mac ---------------- #
        retry_mean = sum(l.retry_rate for l in links) / n_links
        # The anomaly gate is MULTI-SIGNAL (design §7.2): PDR drop, retry-rate rise,
        # noise-floor rise, or sustained zero-RX. A PDR-only CUSUM never fires for
        # congestion or reactive jamming -- those degrade the link without killing it,
        # so the agent was never allowed to decide at all.
        hb_now = max((l.heartbeat_age_s for l in links), default=0.0)
        noise_up = (self.noise_base.init and
                    (o.noise_dbm - self.noise_base.value) > 6.0)
        retry_up = retry_mean > 0.35
        zero_rx = hb_now > 3.0

        # S4' as a GATE trigger, not just a feature.
        #
        # Measured on the fixed simulator: a correctly-modelled reactive jammer drops
        # delivery by only 0.994 -> 0.946 -- about 5 points -- while barrage and spot
        # collapse it to zero. That is not a weak jammer, it is what the brief describes:
        # "delivery fine, but everything is a retransmission". A gate that watches PDR
        # therefore never fires, the agent is never allowed to decide, and the episode
        # ends with no diagnosis at all -- which is exactly what happened to the LLM
        # teacher on ns-3 reactive: 26 steps, 0 decisions, declared=None.
        #
        # The loss is not absent, it is CONCENTRATED right after our own transmissions.
        # S4' measures precisely that, and on the fixed world it reads +0.43 for reactive
        # against <=+0.27 for every other family. So it belongs in the gate.
        self._gate_shadow_exp = getattr(self, "_gate_shadow_exp", 0) + o.shadow_expected
        self._gate_shadow_miss = getattr(self, "_gate_shadow_miss", 0) + o.shadow_missed
        self._gate_silent_exp = getattr(self, "_gate_silent_exp", 0) + o.silent_expected
        self._gate_silent_miss = getattr(self, "_gate_silent_miss", 0) + o.silent_missed
        shadow_up = False
        if self._gate_shadow_exp >= 10 and self._gate_silent_exp >= 10:
            shadow_up = ((self._gate_shadow_miss / self._gate_shadow_exp)
                         - (self._gate_silent_miss / self._gate_silent_exp)) > 0.20

        # WARMUP. Every trigger above is "this departs from our baseline", and for the
        # first couple of seconds there is no baseline to depart from: a handful of frames
        # have been exchanged, so retry_mean and heartbeat_age are noisy and large. Without
        # this guard the gate fired at t=1.0 s on EVERY family, including fading and a
        # perfectly healthy link -- which makes detection latency meaningless and hands the
        # agent a decision before it has seen anything. Corpus onsets are >= 14 s, so a
        # short warmup costs no real detection latency.
        self._gate_samples = getattr(self, "_gate_samples", 0) + 1
        warm = (self._gate_samples * dt) >= GATE_WARMUP_S and self.pdr_base.init

        if warm and (fired or noise_up or retry_up or zero_rx or shadow_up) \
                and self.onset_t is None:
            self.onset_t, self.onset_pos = o.t, o.pos
            self._gate_reason = ("pdr" if fired else "noise" if noise_up
                                 else "retry" if retry_up else "zero_rx" if zero_rx
                                 else "tx_shadow")
        r_ewma = self.retry_ewma.update(retry_mean, dt)
        retries_per_success = retry_mean / max(1e-3, pdr_mean)
        self.max_consec_fail = max(self.max_consec_fail, o.consecutive_tx_fail)
        self.load_hist.push(o.offered_load_norm)
        self.loss_hist.push(1.0 - pdr_mean)
        loss_load_corr = pearson(self.load_hist.values(), self.loss_hist.values())  # S10

        # ---------------- spectral memory ---------------- #
        scan_age = (o.t - self.last_scan.t) if self.last_scan else 999.0
        if self.last_scan and self.last_scan.channels:
            chs = self.last_scan.channels
            noises = [c.noise_dbm for c in chs]
            nm = sum(noises) / len(noises)
            # "Hot" must be measured against OUR OWN quiet floor, not a hard-coded dBm.
            # A fixed JAM_ENERGY_DBM is a knife edge: a barrage jammer that parks the
            # whole band at exactly the threshold makes bad_frac flip between 0.0 and 1.0
            # on a 1 dB change in its power, which is the difference between "all clear"
            # and "everything jammed". Referencing the pre-onset floor (which the ESP32
            # also has, from its boot-time noise_floor reading) removes the cliff and
            # keeps the absolute threshold only as a floor for pathological baselines.
            ref = self.noise_base.value if self.noise_base.init else NOMINAL_NOISE_DBM
            hot_dbm = min(JAM_ENERGY_DBM, ref + HOT_MARGIN_DB)
            bad = sum(1 for c in chs if c.noise_dbm >= hot_dbm) / len(chs)
            spread = math.sqrt(sum((x - nm) ** 2 for x in noises) / len(noises))
            cur = next((c for c in chs if c.ch == o.channel), None)
            rank = (sorted(noises).index(cur.noise_dbm) / max(1, len(chs) - 1)
                    if cur else 0.5)
            alts = [c for c in chs if c.ch != o.channel]
            best_alt = min(alts, key=lambda c: c.noise_dbm) if alts else None
            margin = ((cur.noise_dbm - best_alt.noise_dbm)
                      if (cur and best_alt) else 0.0)
        else:
            bad = spread = rank = margin = 0.0
        # sweep signature: does the hot channel keep moving between scans? (S5)
        hh = self.hot_channel_hist
        periodicity = (sum(1 for i in range(1, len(hh)) if hh[i] != hh[i - 1]) /
                       max(1, len(hh) - 1)) if len(hh) >= 2 else 0.0

        # ---------------- temporal ---------------- #
        self.fade_runs.update(pdr_mean, dt)                                    # S7
        if pdr_mean < FADE_PDR_THRESH:
            self.outage_ticks += 1
        duty = self.outage_ticks / max(1, self.ticks)
        runs = self.fade_runs.runs
        if len(runs) >= 3:
            rm = sum(runs) / len(runs)
            rv = math.sqrt(sum((x - rm) ** 2 for x in runs) / len(runs))
            period_est = rm if rv < 0.35 * max(1e-6, rm) else 0.0   # regular => periodic
        else:
            period_est = 0.0

        # ---------------- self / platform ---------------- #
        speed = math.sqrt(sum(v * v for v in o.vel))
        self.speed_hist.push(speed)
        disp = (math.dist(o.pos, self.onset_pos) if self.onset_pos else 0.0)
        self.disp_hist.push(disp)
        pdr_motion_corr = pearson(self.disp_hist.values(), self.pdr_hist.values())  # S8
        hb_gap = max((l.heartbeat_age_s for l in links), default=0.0)               # S9

        # ---------------- TX-side KPIs (TELEMETRY.md §3) ---------------- #
        if o.tx_attempts > 0:
            self.tx_success.update(o.tx_acked / o.tx_attempts, dt)
            self.tx_retry_depth.update(o.tx_retry_depth, dt)
        if o.tx_defer_ms > 0.0 or o.tx_attempts > 0:
            self.tx_defer.update(o.tx_defer_ms, dt)

        # ---------------- TX-shadow loss, S4' (TELEMETRY.md §1) ---------------- #
        # A reactive jammer concentrates its damage in the window right after OUR
        # transmissions. Compare loss for peer beacons due inside that shadow against
        # beacons due while we were silent. No noise measurement needed -- only our own
        # TX timestamps and the fact that beacons are periodic.
        self.shadow_exp += o.shadow_expected
        self.shadow_miss += o.shadow_missed
        self.silent_exp += o.silent_expected
        self.silent_miss += o.silent_missed
        if self.shadow_exp >= 5 and self.silent_exp >= 5:
            p_shadow = self.shadow_miss / self.shadow_exp
            p_silent = self.silent_miss / self.silent_exp
            shadow_delta = p_shadow - p_silent
        else:
            p_shadow = p_silent = 0.0
            shadow_delta = 0.0

        # ---------------- reciprocal link quality (TELEMETRY.md §2) ------------- #
        rev_r, rev_p, asym, bad_rep = [], [], [], 0
        n_rep = 0
        for l in links:
            if l.pdr_reverse is None or l.report_age_s > 8.0:
                continue
            n_rep += 1
            rev_p.append(l.pdr_reverse)
            asym.append(l.pdr - l.pdr_reverse)
            if l.rssi_reverse_dbm is not None:
                rev_r.append(l.rssi_reverse_dbm)
            if l.pdr_reverse < 0.6:
                bad_rep += 1
        rssi_rev = min(rev_r) if rev_r else -120.0
        pdr_rev = (sum(rev_p) / len(rev_p)) if rev_p else 1.0
        asym_mean = (sum(asym) / len(asym)) if asym else 0.0
        frac_bad_rep = (bad_rep / n_rep) if n_rep else 0.0

        # ---------------- bookkeeping ---------------- #
        b = o.budget or {}
        def frac(used, cap):
            return min(1.0, used / cap) if cap else 0.0
        t_since_onset = (o.t - self.onset_t) if self.onset_t else 0.0
        t_in_ep = o.t - self.t0

        # ------------------------------------------------------------------ #
        f = [0.0] * N_FEATURES
        # --- delivery (0-7) ---
        f[0] = unit(f_fast)
        f[1] = unit(f_slow)
        f[2] = clamp((f_fast - f_slow) * 2.0)
        f[3] = clamp(self.cusum.s / CUSUM_H)
        f[4] = unit(pdr_worst)
        f[5] = clamp(pdr_spread * 4.0)
        f[6] = unit(degraded)
        f[7] = clamp(math.log1p(max(0.0, t_since_onset)) / math.log(31.0))
        # --- signal (8-15) ---
        f[8]  = nz(rssi_mean, RSSI_OFFSET, RSSI_SCALE)
        f[9]  = clamp(self.rssi_fast.std() / 12.0)
        f[10] = clamp(self.rssi_fast.slope_per_s(dt) / SLOPE_SCALE)
        f[11] = nz(self.rssi_fast.min(rssi_mean), RSSI_OFFSET, RSSI_SCALE)
        f[12] = clamp(rssi_pdr_corr)                                    # S1 *
        f[13] = clamp(frames_rx / (RATE_SCALE * dt) * 2.0 - 1.0)
        f[14] = nz(sinr, SINR_OFFSET, SINR_SCALE)                       # S2
        f[15] = clamp((sinr - self.sinr_base.value) / DBM_DELTA_SCALE)  # S2
        # --- interference (16-23) ---
        f[16] = nz(o.noise_dbm, NOISE_OFFSET, NOISE_SCALE)
        f[17] = clamp((o.noise_dbm - self.noise_base.value) / DBM_DELTA_SCALE)  # S2 *
        f[18] = clamp(self.noise_fast.std() / 8.0)
        f[19] = unit(o.cca_busy_frac)
        f[20] = unit(energy_no_preamble)                                # S3 *
        f[21] = clamp(o.decodable_fps / RATE_SCALE * 2.0 - 1.0)         # S3
        f[22] = clamp(tx_noise_delta / 12.0)                            # S4 *
        f[23] = clamp(tx_loss_delta * 2.0)                              # S4
        # --- mac / arq (24-29) ---
        f[24] = unit(r_ewma)
        f[25] = clamp(retries_per_success - 1.0)
        f[26] = clamp(self.max_consec_fail / 20.0 * 2.0 - 1.0)
        f[27] = unit(o.queue_occupancy)
        f[28] = unit(o.offered_load_norm)
        f[29] = clamp(loss_load_corr)                                   # S10 *
        # --- spectral memory (30-35) ---
        f[30] = clamp(math.log1p(min(scan_age, 120.0)) / math.log(121.0) * 2.0 - 1.0)
        f[31] = unit(bad)
        f[32] = clamp(spread / 12.0)                                    # S5 *
        f[33] = unit(rank)
        f[34] = clamp(margin / DBM_DELTA_SCALE)
        f[35] = unit(periodicity)                                       # sweep
        # --- temporal (36-39) ---
        f[36] = clamp(self.fade_runs.mean() / 3.0)                      # S7 *
        f[37] = clamp(self.fade_runs.p95() / 5.0)                       # S7
        f[38] = unit(duty)
        f[39] = clamp(period_est / 5.0)
        # --- self / platform (40-43) ---
        f[40] = clamp(speed / SPEED_SCALE)
        f[41] = clamp(disp / DIST_SCALE)
        f[42] = clamp(pdr_motion_corr)                                  # S8 *
        f[43] = clamp(math.log1p(hb_gap) / math.log(21.0) * 2.0 - 1.0)  # S9 *
        # --- bookkeeping (44-47) ---
        f[44] = unit(frac(b.get("hops_used", 0), b.get("max_channel_hops", 3)))
        f[45] = unit(frac(b.get("scans_used", 0), b.get("max_spectrum_scans", 4)))
        f[46] = unit(frac(b.get("actions_used", 0), b.get("max_costly_actions", 6)))
        f[47] = clamp(t_in_ep / 60.0 * 2.0 - 1.0)
        # --- reciprocal link quality (48-51) --- #
        f[48] = nz(rssi_rev, RSSI_OFFSET, RSSI_SCALE)         # how they hear US
        f[49] = unit(pdr_rev)                                  # reverse delivery
        f[50] = clamp(asym_mean * 2.0)                         # forward - reverse
        f[51] = unit(frac_bad_rep)                             # my TX chain vs one link
        # --- TX-side KPIs (52-55) --- #
        f[52] = unit(self.tx_success.value if self.tx_success.init else 1.0)
        f[53] = clamp((self.tx_defer.value if self.tx_defer.init else 0.0)
                      / DEFER_SCALE_MS * 2.0 - 1.0)            # channel OCCUPANCY
        f[54] = clamp(shadow_delta * 3.0)                      # S4' REACTIVE
        # loss while we were SILENT: the baseline the shadow delta must be read against
        f[55] = unit(p_silent)
        return f


FEATURE_NAMES = [
    "pdr_fast", "pdr_slow", "pdr_delta", "pdr_cusum", "pdr_worst", "pdr_spread",
    "frac_links_degraded", "t_since_onset",
    "rssi_mean", "rssi_std", "rssi_slope", "rssi_min", "rssi_pdr_corr",
    "frames_rx_rate", "sinr", "sinr_delta_base",
    "noise_now", "noise_delta_base", "noise_std", "cca_busy",
    "energy_no_preamble", "foreign_fps", "tx_noise_delta", "tx_loss_delta",
    "retry_ewma", "retries_per_success", "consec_fail", "queue_occ",
    "offered_load", "loss_load_corr",
    "scan_age", "scan_bad_frac", "scan_noise_spread", "scan_cur_rank",
    "scan_best_alt_margin", "scan_periodicity",
    "fade_runlen_mean", "fade_runlen_p95", "outage_duty", "outage_period",
    "own_speed", "displacement", "pdr_motion_corr", "heartbeat_gap",
    "hops_used", "scans_used", "actions_used", "t_in_episode",
    "rssi_reverse", "pdr_reverse", "link_asymmetry", "frac_peers_report_me_bad",
    "tx_success_ratio", "tx_defer_time", "tx_shadow_loss_delta", "silent_loss_rate",
]
assert len(FEATURE_NAMES) == N_FEATURES, (len(FEATURE_NAMES), N_FEATURES)

# The features that carry the discrimination physics (design §7.4)
KEY_FEATURES = {
    "S1_rssi_pdr_corr": 12, "S2_noise_delta": 17, "S2_sinr_delta": 15,
    "S3_energy_no_preamble": 20, "S4_tx_noise_delta": 22, "S5_scan_spread": 32,
    "S6_frac_degraded": 6, "S7_fade_runlen": 36, "S8_pdr_motion_corr": 42,
    "S9_heartbeat_gap": 43, "S10_loss_load_corr": 29,
    # TELEMETRY.md additions
    "S4p_tx_shadow_loss": 54, "S11_link_asymmetry": 50,
    "S12_tx_defer_occupancy": 53, "S13_peers_report_me_bad": 51,
    "S4p_silent_baseline": 55,
}
