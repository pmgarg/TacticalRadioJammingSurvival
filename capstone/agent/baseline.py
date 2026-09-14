"""
Classical deterministic baseline (design §8).

A hand-written rule set over the same features the learned student sees, using the
same Sense/Diagnose/Act API and emitting the same episode log. Built BEFORE the
agent so it is an honest control: if the student cannot beat this, nothing was
learned.

Feature indices it keys on (see percept.features.FEATURE_NAMES):
   0 pdr_fast          6 frac_links_degraded   12 rssi_pdr_corr  (S1)
  17 noise_delta_base  20 energy_no_preamble   22 tx_noise_delta (S4)
  24 retry_ewma        28 offered_load         29 loss_load_corr (S10)
  30 scan_age          31 scan_bad_frac        32 scan_noise_spread (S5)
  34 scan_best_alt     35 scan_periodicity     42 pdr_motion_corr (S8)
  43 heartbeat_gap (S9)
"""
from __future__ import annotations

from .api import Agent, Context, Decision, CAUSES, normalise

F_PDR, F_SPREAD, F_DEG, F_CORR = 0, 5, 6, 12
F_NOISE_D, F_NOPRE, F_TXN = 17, 20, 22
F_RETRY, F_LOAD, F_LOADCORR = 24, 28, 29
F_SCANAGE, F_SCANBAD, F_SCANSPREAD, F_ALTMARGIN, F_PERIOD = 30, 31, 32, 34, 35
F_MOTIONCORR, F_HB = 42, 43


class BaselineAgent:
    name = "baseline"

    def __init__(self, act_threshold: float = 0.6):
        self.act_threshold = act_threshold
        self.reset()

    def reset(self) -> None:
        self._acted_for = None

    # ------------------------------------------------------------------ #
    def _belief(self, f: list[float], ctx: Context) -> dict[str, float]:
        b = {c: 0.02 for c in CAUSES}
        jam_energy = f[F_NOISE_D] > 0.25 and f[F_NOPRE] > 0.10
        scan_fresh = f[F_SCANAGE] < 0.35        # log-scaled; small == recent

        # --- reactive first: it is the only cause keyed to our OWN transmissions (S4),
        #     and it can be caught with or without a scan.
        if f[F_TXN] > 0.35:
            b["reactive"] = 0.85
        # --- a hot channel that keeps MOVING between scans is a sweep (S5/periodicity).
        #     Checked before the energy test because a sweep is often off our channel
        #     at the instant we look, so noise_delta can read zero.
        elif scan_fresh and f[F_PERIOD] > 0.3:
            b["sweep"] = 0.80
        elif jam_energy:
            if scan_fresh:
                # THE barrage/spot discriminator is the per-channel SPREAD (S5):
                # barrage lifts every channel roughly equally (low spread); a spot
                # jammer makes one peak (high spread).
                if f[F_SCANSPREAD] < 0.25 and f[F_SCANBAD] > 0.0:
                    b["barrage"] = 0.85
                elif f[F_SCANSPREAD] >= 0.25:
                    b["spot"] = 0.85
                else:
                    b["barrage"] = b["spot"] = 0.30
            else:
                b["barrage"] = b["spot"] = 0.30   # energy seen, flavour unknown
                b["sweep"] = 0.15
        # --- no excess energy: fading / node loss / congestion / hidden terminal ---
        else:
            if (f[F_SPREAD] > 0.3 and f[F_HB] > 0.4 and f[F_NOISE_D] < 0.15
                    and f[F_DEG] < 0.6):
                # one link fully gone AND that peer has gone silent (S6 + S9)
                b["node_loss"] = 0.80
            elif f[F_LOADCORR] > 0.25 and f[F_LOAD] > 0.2:
                b["congestion"] = 0.80
            elif f[F_RETRY] > 0.1 and f[F_LOAD] < 0.2:
                b["hidden_term"] = 0.70
            elif f[F_CORR] > 0.35 and f[F_NOISE_D] < 0.15:
                b["fading"] = 0.85              # S1, damped: see FIDELITY.md
            elif f[F_PDR] < -0.3 and f[F_NOISE_D] < 0.15:
                # Degraded with a demonstrably CLEAN channel: not an attacker.
                # This is the robust fading call that does not depend on S1.
                b["fading"] = 0.75
                b["node_loss"] = 0.10
        return normalise(b)

    # ------------------------------------------------------------------ #
    def _cheapest_test(self, b: dict, f: list[float], ctx: Context) -> Decision | None:
        """Run the cheapest test that separates the leading hypotheses."""
        top = max(b, key=b.get)
        p = b[top]
        scan_stale = f[F_SCANAGE] >= 0.35

        # jamming flavour unresolved -> a scan is the cheapest thing that helps
        if scan_stale and "spectrum_scan" in ctx.available and \
           (f[F_NOISE_D] > 0.25 or p < self.act_threshold):
            return Decision(b, "spectrum_scan", {"dwell_ms": 20}, p,
                            why="energy present but flavour unresolved; "
                                "one scan separates barrage / spot / sweep",
                            expect="all channels hot => barrage; one hot => spot; "
                                   "hot channel moved => sweep")
        # reactive suspected but not confirmed -> silence test
        if 0.2 < b["reactive"] < self.act_threshold and "silent_listen" in ctx.available:
            return Decision(b, "silent_listen", {"duration_ms": 200}, p,
                            why="retries high and interference tracks our own TX",
                            expect="noise/retries fall to nominal while silent => reactive")
        # one peer quiet, channel clean -> is that peer alive anywhere?
        if 0.2 < b["node_loss"] < self.act_threshold and "neighbor_probe" in ctx.available:
            peer = ctx.peers[0] if ctx.peers else "N2"
            return Decision(b, "neighbor_probe", {"peer": peer}, p,
                            why="one link silent, channel clean",
                            expect="no answer on any channel => dead peer")
        # congestion vs jamming -> drop our own load
        if 0.2 < b["congestion"] < self.act_threshold and "load_test" in ctx.available:
            return Decision(b, "load_test", {"factor": 0.5}, p,
                            why="loss tracks offered load",
                            expect="PDR recovers when load drops => congestion")
        # fading vs a fixed attacker -> the expensive one, last
        if 0.2 < b["fading"] < self.act_threshold and "mobility_test" in ctx.available:
            return Decision(b, "mobility_test", {"dx": 30.0, "dy": 0.0, "dz": 0.0}, p,
                            why="intermittent loss, noise normal; geometry or attacker?",
                            expect="link changes markedly after 30 m => fading")
        return None

    # ------------------------------------------------------------------ #
    def _act(self, b: dict, f: list[float], ctx: Context) -> Decision:
        top = max(b, key=b.get)
        p = b[top]
        av = ctx.available

        if top in ("spot", "sweep") and "hop_channel" in av:
            ch = self._best_channel(ctx)
            return Decision(b, "hop_channel", {"channel": ch}, p,
                            why=f"{top} jamming; a clean channel exists",
                            expect=f"PDR on ch{ch} > 0.8 within 3 s")
        if top == "reactive":
            if "change_tdma_slot" in av:
                return Decision(b, "change_tdma_slot", {"slot": 3}, p,
                                why="reactive jammer keys off our TX; deny the trigger "
                                    "(do NOT hop - it follows us)",
                                expect="retries fall without changing channel")
        if top == "barrage":
            if ctx.lora_available and "fallback_to_lora" in av and f[F_SCANBAD] > 0.5:
                return Decision(b, "fallback_to_lora", {}, p,
                                why="all channels hot; trade rate for a surviving link",
                                expect="low-rate link comes up on the sub-GHz band")
            if "hop_channel" in av:
                ch = self._best_channel(ctx)
                return Decision(b, "hop_channel", {"channel": ch}, p,
                                why="barrage suspected; try the least-bad channel",
                                expect="partial recovery if the jammer is not truly wideband")
            # No hop left. Only claim unrecoverable if the evidence is strong;
            # otherwise let the controller's dead-man rule own that call.
            if f[F_SCANBAD] > 0.5 and p >= 0.7:
                return Decision(b, "declare_link_lost", {}, p, unrecoverable=0.9,
                                why="all channels hot, no hop budget, no LoRa: no escape")
            return Decision(b, "no_op", {}, p,
                            why="barrage suspected but evidence weak and hops spent")
        if top == "fading":
            # THE point of the whole project: do not hop.
            if "set_tx_power" in av:
                return Decision(b, "set_tx_power", {"dbm": 20}, p,
                                why="fading, not an attacker - raise margin, DO NOT hop",
                                expect="PDR improves without touching the channel")
        if top == "node_loss" and "reroute" in av:
            return Decision(b, "reroute", {"via": None}, p,
                            why="peer is gone; route around it",
                            expect="route re-forms via an alternate neighbour")
        if top in ("congestion", "hidden_term") and "change_tdma_slot" in av:
            return Decision(b, "change_tdma_slot", {"slot": 5}, p,
                            why=f"{top}: separate the transmitters, back off",
                            expect="collisions fall, PDR recovers without a hop")
        return Decision(b, "no_op", {}, p, why="no confident action available")

    def _best_channel(self, ctx: Context) -> int:
        scan = ctx.last_scan
        if scan and scan.channels:
            alts = [c for c in scan.channels if c.ch != ctx.channel]
            if alts:
                return min(alts, key=lambda c: c.noise_dbm).ch
        return (ctx.channel % ctx.n_channels) + 1

    # ------------------------------------------------------------------ #
    def decide(self, features: list[float], ctx: Context) -> Decision:
        b = self._belief(features, ctx)
        top, p = max(b, key=b.get), max(b.values())
        if p < self.act_threshold:
            t = self._cheapest_test(b, features, ctx)
            if t is not None:
                return t
            # No hypothesis is credible and no test is left that would help.
            # ABSTAIN. Acting on a flat belief is how an agent burns its budget
            # and declares a recoverable link lost (design §9.4).
            return Decision(b, "no_op", {}, p,
                            why="belief is uninformative and no distinguishing test "
                                "remains; abstaining rather than acting on noise")
        d = self._act(b, features, ctx)
        if d.call not in ctx.available:
            return Decision(b, "no_op", {}, p, why=f"{d.call} unavailable")
        return d
