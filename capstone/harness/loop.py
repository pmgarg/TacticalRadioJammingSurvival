"""The agent loop. Ours, in one file, synchronous and single-threaded on purpose.

    observe -> render -> ask -> parse -> validate -> decide -> record

There is no framework underneath this. Every control-flow branch is visible here, which
is the point: a grader can read the whole decision path in about eighty lines.

`HarnessAgent` implements the same `Agent` protocol as the privileged oracle and the
student, so it drops into controller / gen_traces / dagger / evaluate unchanged.
"""
from __future__ import annotations

import time

from agent.api import Agent, Context, Decision, CAUSES, normalise, uniform_belief
from agent.llm_teacher import PANEL, SYSTEM
from gateway.provider import ClaudeCliProvider, LlmError
from harness.registry import ToolRegistry
from harness.parser import parse_decision
from harness.trace import TraceStore, DecisionRecord

DECISION_RULE = """HOW YOUR ANSWER IS SCORED - read this before answering.

`belief` is a probability distribution over the eight hypotheses; it must sum to ~1.
You are treated as having COMMITTED to a diagnosis only when one hypothesis exceeds
0.75. Below that you are considered still undecided, and an episode that ends undecided
scores as a failure to diagnose. So do not hedge out of politeness: if the evidence is
in, put the mass where it belongs and say so.

But commit for the right reason. The cost of being wrong is NOT symmetric:

  declaring JAMMING when the truth is FADING   costs 8-10  (you hop away from a working
                                                            mesh and lose it)
  declaring FADING when the truth is JAMMING   costs ~1    (you wait, then re-diagnose)

So the rule you are applying is minimum EXPECTED cost, not most-likely cause.

STEP 1 - IS THERE AN EMITTER?  Look only at noise_delta_base, energy_no_preamble and a
spectrum scan. If the noise floor is well above baseline, OR busy energy is high with no
decodable packets, OR a scan shows hot channels -> an attacker is present, go to step 2.
If noise is nominal, there is NO attacker: go to step 3. Symptoms alone (delivery down,
RSSI down, retries up) are NOT evidence of an attacker.

STEP 2 - WHICH ATTACKER?  These are four different things; do not default to barrage.
  barrage   scan_bad_frac near 100% AND scan_noise_spread LOW - the whole band is lifted
            evenly. If only part of the band is hot, it is not barrage.
  spot      scan_bad_frac partial AND scan_noise_spread HIGH - one channel and its
            neighbours are hot, the rest of the band is quiet. (Adjacent channels rise
            too because of the transmit mask; that is still spot, not barrage.)
  sweep     the hot channel MOVES between scans - scan_periodicity high. It reads 0%
            until you have TWO scans, so before that it tells you nothing. If the band
            looks jammed and you have scanned only once, SCAN AGAIN.
  reactive  noise and loss appear only when WE transmit: tx_noise_delta high, or
            tx_shadow_loss_delta high while silent_loss_rate is LOW. The band looks clean
            when you are quiet, which is why silent_listen settles it.
  If you cannot separate them and have a scan left, SCAN. One scan is cheap (cost 1) and
  it is what distinguishes all four.

STEP 3 - WHICH BENIGN CAUSE?  There are FOUR, not one. `fading` is not the default answer
for "no attacker"; it is one of four and it has its own positive signature. Separate them
before committing:
  node_loss    ONE peer went silent while the others are fine: pdr_spread HIGH is the
               EARLY signal, heartbeat_gap HIGH is the confirming one. heartbeat_gap
               accumulates in real time, so a few seconds after the peer dies it is still
               low and tells you nothing - do not read a low heartbeat_gap as evidence
               AGAINST node_loss early in an episode. If pdr_spread is high (one link
               differs sharply from the rest), run neighbor_probe before committing to
               anything else; that is what it is for, and it costs 2.
  congestion   loss tracks OUR OWN load: loss_load_corr high, tx_defer_time high,
               foreign_fps high (other radios ARE decodable - real traffic, not noise).
               Confirm with load_test.
  hidden_term  collisions at the receiver while our own offered load is LOW:
               tx_defer_time high but offered_load low, retries high, foreign_fps
               moderate. Peers cannot hear each other.
  fading       propagation only: RSSI DOWN, noise nominal, rssi_pdr_corr HIGH (delivery
               follows signal level), fade_runlen short and bursty, heartbeat_gap normal,
               tx_defer_time low, loss_load_corr low. Confirm with mobility_test.
  Committing to `fading` when heartbeat_gap is high, or when tx_defer_time or
  loss_load_corr is high, is WRONG - those are the other three.
  Fading is BROAD: it degrades your links together. If pdr_spread is HIGH, one link is
  behaving differently from the rest, and that is node_loss (or a local obstruction),
  NOT fading. Do not commit to fading while pdr_spread is high and neighbor_probe is
  still unspent.

EVIDENCE MATURITY. Several statistics need time or repetition to separate: heartbeat_gap
has to accumulate, scan_periodicity needs a SECOND scan, and the reactive shadow
statistics need enough of our own transmissions to average. Committing at the first
plausible reading is how you end up confidently wrong. If a cheap unspent test would
settle it, spend it; an unused budget at the end of a misdiagnosed episode is not thrift.

STEP 4 - ACT OR TEST.
  - Evidence in and one hypothesis clearly ahead -> commit above 0.75 and take its action.
  - Torn between two with a test left that separates them -> run the CHEAPEST such test.
  - Torn with no test left and no evidence -> keep the belief spread and choose no_op.
    Abstaining is a valid, scored outcome. Guessing is not.

"""

REPLY_SPEC = DECISION_RULE + """Reply with ONLY a JSON object, no other text:
{"belief":{"barrage":0.0,"spot":0.0,"reactive":0.0,"sweep":0.0,"fading":0.0,
           "node_loss":0.0,"congestion":0.0,"hidden_term":0.0},
 "call":"<one function name from the lists above>","args":{},
 "why":"<one sentence>","confidence":0.0,"unrecoverable":0.0}"""


# Indices whose normalisation is unit(x): a 0..1 quantity mapped onto [-1,+1]. For these,
# a displayed "+0.00" means FIFTY PERCENT, not zero -- and a model (or a human) reading the
# raw number will call it "nominal". Every one of them is decoded back to its native
# percentage before it reaches the prompt. This was a real defect: an LLM diagnosing a
# barrage jammer read "scan_bad_frac +0.00" as "scan came back clean" and walked away
# from the correct diagnosis.
UNIT_MAPPED = {0, 1, 4, 6, 19, 20, 24, 27, 28, 31, 33, 35, 38, 44, 45, 46, 49, 51, 52, 55}

_BARS = "▁▂▃▄▅▆▇█"


def _decode(idx: int, v: float) -> str:
    if idx in UNIT_MAPPED:
        return f"{(v + 1.0) * 50.0:5.0f}%"
    return f"{v:+6.2f}"


def _bar(v: float) -> str:
    i = int(round((max(-1.0, min(1.0, v)) + 1.0) / 2.0 * (len(_BARS) - 1)))
    return _BARS[i]


def _level(v: float) -> str:
    return "HIGH" if v >= 0.45 else "low " if v <= -0.45 else "mid "


def render_panel(f: list[float], ctx: Context, reg: ToolRegistry) -> str:
    """Features as a readable table, plus the live tool catalogue with prices.

    Handing a language model a bare float vector throws away the only thing it is good
    at. Names, units, decoded percentages and an explicit scale are the whole reason an
    LLM teacher can beat a threshold rule on a family it has never seen.
    """
    b = ctx.budget or {}
    head = [
        f"OBSERVATION   t={ctx.t:.1f}s   channel={ctx.channel}/{ctx.n_channels}",
        f"budget        hops_used={b.get('hops_used', 0)}/{b.get('max_channel_hops', '?')}  "
        f"scans_used={b.get('scans_used', 0)}/{b.get('max_spectrum_scans', '?')}  "
        f"actions_used={b.get('actions_used', 0)}/{b.get('max_costly_actions', '?')}",
    ]
    if ctx.tests_run:
        head.append(f"tests already run:     {', '.join(ctx.tests_run)}")
    if ctx.actions_taken:
        head.append(f"actions already taken: {', '.join(ctx.actions_taken)}")
    head.append("")
    head.append(reg.render_catalogue(ctx.available))
    head.append("")
    head.append("TELEMETRY   percentages are native; signed values are normalised "
                "(-1 = low, 0 = nominal, +1 = high)")
    head.append(f"{'feature':<24}{'value':>7}  lvl    meaning")
    for name, idx, desc in PANEL:
        if idx < len(f):
            v = f[idx]
            head.append(f"{name:<24}{_decode(idx, v):>7} {_bar(v)}{_level(v)}  {desc}")
    return "\n".join(head)


class HarnessAgent:
    """LLM-driven agent running inside our own harness."""

    name = "harness_llm"

    def __init__(self, provider=None, registry: ToolRegistry | None = None,
                 trace: TraceStore | None = None, episode: str = "ep",
                 family: str = "", verbose: bool = False,
                 novelty_gate: float = 0.15, quiet_pdr: float = 0.55):
        self.p = provider or ClaudeCliProvider()
        self.reg = registry or ToolRegistry()
        self.trace = trace
        self.episode = episode
        self.family = family
        self.verbose = verbose
        # Event gate: re-reason when the situation CHANGES, not on a metronome.
        # A 1 Hz poll of a 60 s episode is ~60 model calls to answer the same question
        # sixty times. `novelty_gate` is the L-inf distance on the panel that counts as
        # a new situation; `quiet_pdr` is the delivery ratio above which nothing is wrong
        # and there is nothing to diagnose.
        self.novelty_gate = novelty_gate
        self.quiet_pdr = quiet_pdr
        self.skipped_quiet = 0
        self.skipped_stable = 0
        self.parse_failures = 0
        self.repairs = 0
        self.provider_errors = 0
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self._last_panel: list[float] | None = None
        self._last_decision: Decision | None = None
        self._last_available: tuple = ()

    def set_episode(self, episode: str, family: str = "") -> None:
        self.episode = episode
        self.family = family
        self.reset()

    # ------------------------------------------------------------ event gate
    def _panel_of(self, f: list[float]) -> list[float]:
        return [f[i] if i < len(f) else 0.0 for _, i, _ in PANEL]

    def _should_ask(self, f: list[float], ctx: Context) -> tuple[bool, str]:
        panel = self._panel_of(f)
        # 1. Nothing is wrong: delivery is healthy and no diagnosis is pending.
        if f[0] >= self.quiet_pdr and self._last_decision is None:
            return False, "quiet"
        avail = tuple(sorted(ctx.available))
        # 2. Nothing changed: same panel (to the gate), same legal moves.
        if self._last_panel is not None and avail == self._last_available:
            drift = max(abs(a - b) for a, b in zip(panel, self._last_panel))
            if drift < self.novelty_gate:
                return False, "stable"
        return True, ""

    # ---------------------------------------------------------------- loop
    def decide(self, features: list[float], ctx: Context) -> Decision:
        self.step += 1

        ask, why_not = self._should_ask(features, ctx)
        if not ask:
            if why_not == "quiet":
                self.skipped_quiet += 1
                return Decision(uniform_belief(), "no_op", {}, 0.0,
                                why="link healthy; nothing to diagnose")
            self.skipped_stable += 1
            self._last_panel = self._panel_of(features)
            self._last_available = tuple(sorted(ctx.available))
            d = self._last_decision
            # Repeat the belief, but never repeat a costly call -- re-spending budget
            # on an unchanged situation is exactly the pathology the gate exists to stop.
            return Decision(d.belief, "no_op", {}, d.confidence,
                            unrecoverable=d.unrecoverable,
                            why="situation unchanged; holding")

        self._last_panel = self._panel_of(features)
        self._last_available = tuple(sorted(ctx.available))
        prompt = SYSTEM.split("Reply with ONLY")[0].rstrip() + "\n\n" \
            + render_panel(features, ctx, self.reg) + "\n\n" + REPLY_SPEC

        rec = DecisionRecord(episode=self.episode, step=self.step, t_sim=ctx.t,
                             agent=self.name, scenario_family=self.family, prompt=prompt)

        before_hits = self.p.cache_hits
        t0 = time.time()
        try:
            raw = self.p.complete(prompt)
        except LlmError as e:
            self.provider_errors += 1
            rec.error, rec.parse_mode, rec.latency_s = str(e)[:200], "failed", time.time() - t0
            rec.call, rec.belief, rec.why = "no_op", uniform_belief(), "provider error; abstaining"
            self._emit(rec)
            return self._remember(Decision(uniform_belief(), "no_op", {}, 0.0,
                                           why="LLM provider error; abstaining"))
        rec.latency_s = round(time.time() - t0, 3)
        rec.cache_hit = self.p.cache_hits > before_hits
        rec.raw_response = raw[:4000]

        out = parse_decision(raw)
        rec.parse_mode = out.mode
        if out.mode == "repaired":
            self.repairs += 1
        if not out.ok:
            self.parse_failures += 1
            rec.call, rec.belief = "no_op", uniform_belief()
            rec.why, rec.validation_note = "unparseable reply; abstaining", out.detail[:200]
            self._emit(rec)
            return self._remember(Decision(uniform_belief(), "no_op", {}, 0.0,
                                           why="LLM reply unparseable; abstaining"))

        d = out.data
        raw_belief = d.get("belief", {}) or {}
        belief = normalise({c: _f(raw_belief.get(c, 0.0)) for c in CAUSES})
        call, args, note = self.reg.validate(str(d.get("call", "no_op")),
                                             d.get("args") or {}, ctx.available)

        rec.features = [round(float(x), 6) for x in features]
        rec.available = list(ctx.available)
        rec.belief, rec.call, rec.args = belief, call, args
        rec.confidence = _f(d.get("confidence", 0.0))
        rec.why = str(d.get("why", ""))[:400]
        rec.validation_note = note
        rec.cost_charged = self.reg.cost_of(call)
        self._emit(rec)

        if self.verbose:
            print(f"  [{self.step}] {call:<18} p({max(belief,key=belief.get)})="
                  f"{max(belief.values()):.2f}  {rec.why[:70]}")

        return self._remember(Decision(belief, call, args,
                                       confidence=rec.confidence,
                                       unrecoverable=_f(d.get("unrecoverable", 0.0)),
                                       why=rec.why))

    def _remember(self, d: Decision) -> Decision:
        self._last_decision = d
        return d

    def _emit(self, rec: DecisionRecord) -> None:
        if self.trace is not None:
            self.trace.append(rec)

    def stats(self) -> dict:
        s = dict(self.p.stats())
        s.update(parse_failures=self.parse_failures, repairs=self.repairs,
                 provider_errors=self.provider_errors,
                 skipped_quiet=self.skipped_quiet, skipped_stable=self.skipped_stable)
        return s


_REPLY_CACHE: list = []


def reply_spec() -> str:
    """REPLY_SPEC with the measured prototype table substituted in panel units.

    Generated, never hand-written: a hand-written table quoted unit-mapped features on
    the [-1,+1] scale while the panel shows them as percentages, so the model was asked
    to match `offered_load 35%` against `-0.30`. Same number, two notations, no way to
    reconcile them. Generating the table with the panel's own formatter makes that class
    of mistake impossible.
    """
    if not _REPLY_CACHE:
        try:
            from harness.prototypes import render as render_protos
            table = render_protos(_decode)
        except Exception as e:                                   # noqa: BLE001
            table = f"(prototype table unavailable: {e})"
        _REPLY_CACHE.append(REPLY_SPEC.replace("{PROTOTYPES}", table))
    return _REPLY_CACHE[0]


def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if x == x and abs(x) != float("inf") else default
    except (TypeError, ValueError):
        return default
