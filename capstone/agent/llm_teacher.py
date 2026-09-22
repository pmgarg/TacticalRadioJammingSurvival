"""
LLM teacher (the brief's actual teacher).

"Run a large model in simulation as a teacher, capture the situation-to-action traces,
and distil them into a small model that fits the drone. Then measure the gap."

This is that model. It sees exactly what the student sees -- the 56-dim feature vector,
rendered as a readable table with baselines and units -- and NO ground truth. Its whole
advantage must be reasoning, not information; a teacher given the answer would teach the
student to guess.

It implements the same Agent protocol as the oracle and the student, so it drops into
gen_traces / dagger / bridge_dagger / evaluate unchanged.
"""
from __future__ import annotations

import json
import os

from .api import Agent, Context, Decision, CAUSES, normalise
from gateway.provider import ClaudeCliProvider, extract_json, LlmError
from percept.features import FEATURE_NAMES

DIAG = ["spectrum_scan", "silent_listen", "neighbor_probe", "load_test",
        "listen_test", "transmit_probe", "channel_hop_probe", "mobility_test"]
ACTS = ["no_op", "set_tx_power", "reroute", "change_tdma_slot", "hop_channel",
        "fallback_to_lora", "move", "declare_link_lost"]

# Which features to show, and how to describe them. Showing all 56 raw floats buries the
# signal; these are the ones the physics says carry the diagnosis (DESIGN §7.4 + TELEMETRY).
PANEL = [
    ("pdr_fast", 0, "delivery ratio now"),
    ("pdr_slow", 1, "delivery ratio (5 s)"),
    ("pdr_spread", 5, "spread across links (high = one link differs)"),
    ("frac_links_degraded", 6, "fraction of peers degraded"),
    ("rssi_mean", 8, "received signal level"),
    ("rssi_pdr_corr", 12, "RSSI/PDR correlation (fading -> high)"),
    ("noise_delta_base", 17, "noise floor rise vs nominal (jamming -> high)"),
    ("energy_no_preamble", 20, "busy energy with NO decodable packet (jamming -> high)"),
    ("foreign_fps", 21, "decodable foreign packets (congestion -> high)"),
    ("tx_noise_delta", 22, "noise while WE transmit (reactive -> high)"),
    ("retry_ewma", 24, "link-layer retry rate"),
    ("offered_load", 28, "our own offered load"),
    ("loss_load_corr", 29, "loss vs our load (congestion -> high)"),
    ("scan_age", 30, "staleness of last spectrum scan (-1 = fresh, +1 = never)"),
    ("scan_bad_frac", 31, "fraction of channels hot in last scan"),
    ("scan_noise_spread", 32, "spread across channels (spot -> high, barrage -> low)"),
    ("scan_periodicity", 35, "hot channel keeps moving (sweep -> high)"),
    ("heartbeat_gap", 43, "longest silence from a peer (dead peer -> high)"),
    ("pdr_reverse", 49, "how well peers say they hear US"),
    ("link_asymmetry", 50, "forward minus reverse (my RX vs my TX fault)"),
    ("tx_defer_time", 53, "wait for a clear channel (congestion/hidden -> high)"),
    ("tx_shadow_loss_delta", 54, "extra loss right after OUR transmissions (reactive -> high)"),
    ("silent_loss_rate", 55, "loss while we were silent (read shadow delta against this)"),
    # --- added by train/analyse_failures.py, 2026-09-19 -------------------------------
    # These six were NOT shown to the teacher, yet each separates one of its most common
    # confusions on the labelled ns-3 corpus (AUC-separability in brackets). The teacher
    # was being asked to distinguish classes using evidence it had never been given.
    ("noise_now", 16, "absolute noise floor now (barrage vs congestion: 1.00)"),
    ("noise_std", 18, "how much the floor MOVES (steady emitter vs bursty traffic: 0.92)"),
    ("retries_per_success", 25, "retries per delivered frame (fading vs reactive: 1.00)"),
    ("outage_duty", 38, "fraction of time fully out (hidden_term vs reactive)"),
    ("outage_period", 39, "spacing of outages (periodic = sweep/reactive: 0.96)"),
    ("scan_best_alt_margin", 34, "how much quieter the best other channel is (spot vs barrage: 1.00)"),
]

SYSTEM = """You are the diagnostic agent on a drone whose mesh radio link is failing.
You must work out WHY and act. You have NO ground truth.

Several very different causes look almost identical from inside the radio, so the
expensive mistake is acting on a guess. Above all: declaring jamming when the real cause
is FADING is the worst error, because the response (hopping channel) destroys the mesh you
still had. Never declare jamming on symptoms alone - require positive evidence
(a raised noise floor, or a spectrum scan) first.

All feature values are normalised to [-1, +1]; -1 is low, +1 is high, 0 is nominal.

HYPOTHESES
  barrage      wideband jamming, every channel hot
  spot         narrowband jamming, one channel hot
  reactive     jammer fires only when WE transmit
  sweep        jammer walks across channels over time
  fading       propagation/multipath, NO attacker  <- the false-positive trap
  node_loss    a peer died; channel is clean
  congestion   real traffic from others; decodable packets, channel BUSY
  hidden_term  collisions at the receiver while our own load is low

DIAGNOSTIC TESTS (cost in brackets) - these buy information:
  spectrum_scan[1] energy vs channel: jamming vs congestion, and barrage/spot/sweep
  silent_listen[2] stop transmitting, re-measure: confirms REACTIVE
  neighbor_probe[2] is a specific peer alive on any channel: confirms NODE_LOSS
  load_test[2]     drop our own load: confirms CONGESTION
  listen_test[2]   a neighbour transmits while we listen: our RX vs the channel
  transmit_probe[2] we transmit, neighbour reports: our TX vs the channel
  channel_hop_probe[4] recover on a clean channel, or prove none exists
  mobility_test[8] move ~30 m: confirms FADING vs a fixed attacker

RECOVERY ACTIONS:
  no_op, set_tx_power, reroute, change_tdma_slot, hop_channel,
  fallback_to_lora, move, declare_link_lost

POLICY
  - If unsure, run the CHEAPEST test that separates your leading hypotheses.
  - Act only when you are confident; abstain (no_op) rather than act on noise.
  - fading   -> raise power / small move / reroute. NEVER hop_channel.
  - spot/sweep -> hop_channel to a scan-clean channel.
  - reactive -> change_tdma_slot, cut duty. Do NOT hop; it follows your transmissions.
  - A tx_shadow_loss_delta/tx_defer_time spike right after YOUR OWN hop_channel or
    change_tdma_slot is the resync gap, not a new jammer - it does not count as reactive
    evidence until silent_listen clearly confirms it (loss visibly drops while silent).
  - barrage  -> fallback_to_lora, or declare_link_lost if nothing is reachable.
  - node_loss -> reroute around the dead peer.
  - congestion/hidden_term -> change_tdma_slot / back off.
  - If every channel is jammed AND no peer is reachable, the honest action is
    declare_link_lost (return-to-home). Hopping forever is a failure.

Reply with ONLY a JSON object and no other text:
{"belief":{"barrage":0.0,"spot":0.0,"reactive":0.0,"sweep":0.0,"fading":0.0,
           "node_loss":0.0,"congestion":0.0,"hidden_term":0.0},
 "call":"<one function name>","args":{},"why":"<one sentence>",
 "confidence":0.0,"unrecoverable":0.0}"""


def render_state(f: list[float], ctx: Context) -> str:
    lines = [f"OBSERVATION  t={ctx.t:.1f}s   channel={ctx.channel}/{ctx.n_channels}"]
    b = ctx.budget or {}
    lines.append(f"budget: hops_used={b.get('hops_used', 0)}  "
                 f"actions_used={b.get('actions_used', 0)}")
    lines.append(f"available calls: {', '.join(ctx.available)}")
    if ctx.tests_run:
        lines.append(f"tests already run: {', '.join(ctx.tests_run)}")
    if ctx.actions_taken:
        lines.append(f"actions already taken: {', '.join(ctx.actions_taken)}")
    lines.append("")
    lines.append(f"{'feature':<24}{'value':>7}   meaning")
    for name, idx, desc in PANEL:
        if idx < len(f):
            lines.append(f"{name:<24}{f[idx]:>+7.2f}   {desc}")
    return "\n".join(lines)


class LlmOracleLabeller:
    """The brief's teacher: a large model, offline, in simulation, no ground truth."""
    name = "llm_teacher"

    def __init__(self, provider=None, k: int = 1, verbose: bool = False):
        self.p = provider or ClaudeCliProvider()
        self.k = k                      # self-consistency samples
        self.verbose = verbose
        self.failures = 0
        self.reset()

    def reset(self) -> None:
        self._last = None

    def decide(self, features: list[float], ctx: Context) -> Decision:
        prompt = SYSTEM + "\n\n" + render_state(features, ctx)
        try:
            raw = self.p.complete(prompt)
            d = extract_json(raw)
        except (LlmError, json.JSONDecodeError, ValueError):
            self.failures += 1
            # An unparseable teacher must not poison the trace: abstain.
            return Decision({c: 1.0 / len(CAUSES) for c in CAUSES}, "no_op", {}, 0.0,
                            why="LLM teacher reply unusable; abstaining")
        belief = normalise({c: float(d.get("belief", {}).get(c, 0.0)) for c in CAUSES})
        call = str(d.get("call", "no_op"))
        if call not in ctx.available:
            call = "no_op"
        args = d.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}
        return Decision(belief, call, args,
                        confidence=float(d.get("confidence", 0.0) or 0.0),
                        unrecoverable=float(d.get("unrecoverable", 0.0) or 0.0),
                        why=str(d.get("why", ""))[:400])
