"""
ns-3 percept CSV -> RawObs stream.

This is what makes "one feature layer, three call sites" real: the SAME
FeatureExtractor consumes ns-3 output and refsim output, so a feature can never
mean one thing in training and another in the authoritative simulator.
"""
from __future__ import annotations

import csv

from percept.features import RawObs, LinkObs, ChannelObs, ScanResult


def _floats(s: str) -> list[float]:
    s = (s or "").strip()
    return [float(x) for x in s.split()] if s else []


def load_ns3_percept(path: str, n_channels: int = 8) -> list[RawObs]:
    """Read <out>.percept.csv into the same RawObs the refsim emits."""
    out: list[RawObs] = []
    prev_scan_t = None
    prev_rx_ok = None
    prev_t = None
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t = float(row["t"])
            pdrs = _floats(row.get("pdr_per_link", ""))
            rssis = _floats(row.get("rssi_per_link", ""))
            rev_r = _floats(row.get("rev_rssi_per_link", ""))
            rev_p = _floats(row.get("rev_pdr_per_link", ""))
            rev_a = _floats(row.get("rev_age_per_link", ""))
            band = _floats(row.get("band_power", ""))
            n = max(len(pdrs), len(rssis))
            links = {}
            hb = float(row.get("hb_gap_max", 0.0) or 0.0)
            for i in range(n):
                pdr = pdrs[i] if i < len(pdrs) else 0.0
                rssi = rssis[i] if i < len(rssis) else -120.0
                rp = rev_p[i] if i < len(rev_p) else -1.0
                links[f"P{i}"] = LinkObs(
                    pdr=pdr, rssi_dbm=rssi,
                    retry_rate=min(1.0, 1.0 - pdr),
                    frames_rx=int(round(pdr * 10)),
                    heartbeat_age_s=(hb if pdr < 0.05 else 0.0),
                    reachable=pdr > 0.1,
                    rssi_reverse_dbm=(rev_r[i] if i < len(rev_r) else None),
                    pdr_reverse=(rp if rp >= 0 else None),
                    report_age_s=(rev_a[i] if i < len(rev_a) else 999.0))
            floor = float(row.get("meas_floor_dbm", -96.0) or -96.0)
            if floor < -199:
                floor = -96.0
            # busy fraction is not directly exposed by ns-3; derive it the SAME coarse
            # way the ESP32 will (design §11.2) so the model never trains on
            # information the hardware cannot provide. decodable rate IS available,
            # from rx_ok below.
            jam = floor > -80.0
            # ROBUSTNESS (capstone): see the matching comment in bridge_server.py --
            # decod used to be 10*mean(pdr_per_link), the mesh's own delivery ratio,
            # which degrades under jamming AND congestion alike. rx_ok is a cumulative
            # count of every 802.11 frame the sniffer decoded (mesh peer or not); the
            # windowed delta is the real foreign-traffic-capable signal. /8 (not /60,
            # which was calibrated against an unrepresentative smoke-test scenario)
            # keeps jamming at the -1 floor while giving real corpus congestion traffic
            # (~30-225 raw fps late-episode) genuine separation.
            rx_ok = float(row.get("rx_ok", 0.0) or 0.0)
            if prev_rx_ok is not None and t > prev_t:
                decod = max(0.0, (rx_ok - prev_rx_ok) / (t - prev_t)) / 8.0
            else:
                decod = 0.0
            prev_rx_ok, prev_t = rx_ok, t
            busy = min(1.0, 0.05 + (0.9 if jam else 0.0) + 0.3 * min(1.0, decod / 20.0))
            scan = None
            if band and (prev_scan_t is None or t - prev_scan_t >= 2.0):
                scan = ScanResult(t=t, channels=[
                    ChannelObs(ch=i + 1, noise_dbm=band[i],
                               busy_frac=1.0 if band[i] > -80 else 0.05,
                               decodable_fps=decod if i == int(row.get("channel", 6)) - 1 else 0.0)
                    for i in range(min(len(band), n_channels))])
                prev_scan_t = t
            out.append(RawObs(
                t=t, channel=int(float(row.get("channel", 6))), links=links,
                noise_dbm=floor, cca_busy_frac=busy, decodable_fps=decod,
                own_tx_active=(t * 10) % 3 < 1, own_tx_duty=0.35,
                offered_load_norm=0.4,
                queue_occupancy=min(1.0, 1.0 - (sum(pdrs) / len(pdrs) if pdrs else 0.0)),
                consecutive_tx_fail=0, pos=(0.0, 0.0, 30.0), vel=(0.0, 0.0, 0.0),
                scan=scan, budget={},
                tx_attempts=int(float(row.get("tx_attempts", 0) or 0)),
                tx_acked=int(float(row.get("tx_acked", 0) or 0)),
                tx_retry_depth=float(row.get("tx_retry_depth", 0) or 0),
                tx_defer_ms=float(row.get("tx_defer_ms", 0) or 0),
                shadow_expected=int(float(row.get("shadow_exp", 0) or 0)),
                shadow_missed=int(float(row.get("shadow_miss", 0) or 0)),
                silent_expected=int(float(row.get("silent_exp", 0) or 0)),
                silent_missed=int(float(row.get("silent_miss", 0) or 0))))
    return out


def features_from_ns3(path: str, n_channels: int = 8) -> list[list[float]]:
    from percept.features import FeatureExtractor
    obs = load_ns3_percept(path, n_channels)
    dt = (obs[1].t - obs[0].t) if len(obs) > 1 else 0.1
    ex = FeatureExtractor(n_channels, dt=dt)
    return [ex.update(o) for o in obs]
