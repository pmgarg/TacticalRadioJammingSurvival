"""
Flatten a Scenario into the simple key=value config the ns-3 program reads.

Python owns the scenario schema (arbitrary topology, arbitrary jammers). The C++
side only has to parse `key=value` lines, so there is no JSON dependency in ns-3
and the two stay in sync through this one function.

    python3 -m sim.export_ns3 ../scenarios/spot_single_channel.yaml -o /tmp/s.cfg
"""
from __future__ import annotations

import argparse
import os
import sys

from scenario.schema import Scenario, load_scenario

CH_2GHZ = {1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432, 6: 2437,
           7: 2442, 8: 2447, 9: 2452, 10: 2457, 11: 2462, 12: 2467, 13: 2472}


def to_config(sc: Scenario) -> str:
    L: list[str] = []
    a = L.append
    a(f"name={sc.name}")
    a(f"family={sc.family}")
    a(f"seed={sc.seed}")
    a(f"duration={sc.duration_s}")
    a(f"n_channels={sc.n_channels}")
    a(f"channel={sc.channel}")
    a(f"routing={sc.routing}")
    a(f"comm_range={sc.comm_range_m}")
    a(f"lora_available={int(sc.lora_available)}")
    a(f"lora_jammed={int(sc.lora_jammed)}")
    cm = sc.channel_model
    a(f"pathloss={cm.path_loss}")
    a(f"exponent={cm.exponent}")
    a(f"ref_loss={cm.ref_loss_db}")
    a(f"fading={cm.fading}")
    a(f"nakagami_m={cm.nakagami_m}")
    a(f"doppler={cm.doppler_hz}")
    a(f"shadowing={cm.shadowing_db}")
    a(f"noise_floor={cm.noise_floor_dbm}")
    a(f"truth_cause={sc.truth.cause}")
    a(f"truth_onset={sc.truth.onset_t}")
    a(f"truth_recoverable={int(sc.truth.recoverable)}")

    a(f"n_nodes={len(sc.nodes)}")
    for i, n in enumerate(sc.nodes):
        a(f"node.{i}.id={n.id}")
        a(f"node.{i}.x={n.pos[0]}")
        a(f"node.{i}.y={n.pos[1]}")
        a(f"node.{i}.z={n.pos[2]}")
        a(f"node.{i}.role={n.role}")
        a(f"node.{i}.txpower={n.tx_power_dbm}")

    a(f"n_mobility={len(sc.mobility)}")
    for i, m in enumerate(sc.mobility):
        a(f"mob.{i}.node={m.node}")
        a(f"mob.{i}.type={m.type}")
        a(f"mob.{i}.speed={m.speed}")
        a(f"mob.{i}.vx={m.velocity[0]}")
        a(f"mob.{i}.vy={m.velocity[1]}")
        a(f"mob.{i}.vz={m.velocity[2]}")
        a(f"mob.{i}.n_wp={len(m.waypoints)}")
        for k, w in enumerate(m.waypoints):
            a(f"mob.{i}.wp.{k}.x={w[0]}")
            a(f"mob.{i}.wp.{k}.y={w[1]}")
            a(f"mob.{i}.wp.{k}.z={w[2]}")

    a(f"n_flows={len(sc.traffic)}")
    for i, f in enumerate(sc.traffic):
        a(f"flow.{i}.src={f.src}")
        a(f"flow.{i}.dst={f.dst}")
        a(f"flow.{i}.kbps={f.rate_kbps}")
        a(f"flow.{i}.bytes={f.packet_bytes}")
        a(f"flow.{i}.start={f.start_s}")
        a(f"flow.{i}.stop={f.stop_s if f.stop_s is not None else sc.duration_s}")

    a(f"n_jammers={len(sc.jammers)}")
    for i, j in enumerate(sc.jammers):
        a(f"jam.{i}.id={j.id}")
        a(f"jam.{i}.type={j.type}")
        a(f"jam.{i}.x={j.pos[0]}")
        a(f"jam.{i}.y={j.pos[1]}")
        a(f"jam.{i}.z={j.pos[2]}")
        a(f"jam.{i}.eirp={j.eirp_dbm}")
        a(f"jam.{i}.duty={j.duty}")
        a(f"jam.{i}.dwell_ms={j.dwell_ms}")
        a(f"jam.{i}.delay_us={j.delay_us}")
        a(f"jam.{i}.burst_ms={j.burst_ms}")
        a(f"jam.{i}.p_fire={j.p_fire}")
        chans = j.channels or list(range(1, sc.n_channels + 1))
        a(f"jam.{i}.n_ch={len(chans)}")
        for k, ch in enumerate(chans):
            a(f"jam.{i}.ch.{k}={ch}")

    a(f"n_events={len(sc.events)}")
    for i, e in enumerate(sc.events):
        a(f"ev.{i}.t={e.t}")
        a(f"ev.{i}.type={e.type}")
        a(f"ev.{i}.target={e.target or ''}")
        a(f"ev.{i}.depth_db={e.params.get('depth_db', 0.0)}")
        a(f"ev.{i}.nakagami_m={e.params.get('nakagami_m', 0.0)}")
        a(f"ev.{i}.factor={e.params.get('factor', 1.0)}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    sc = load_scenario(a.scenario)
    with open(a.out, "w") as fh:
        fh.write(to_config(sc))
    print(f"wrote {a.out} ({len(sc.nodes)} nodes, {len(sc.jammers)} jammers)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
