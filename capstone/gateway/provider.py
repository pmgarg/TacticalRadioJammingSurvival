"""
LLM gateway (the glc_v5 / ClaudeCLIProvider role from the HLD §11).

One place where every LLM call goes, so the teacher pipeline is provider-agnostic,
cached, audited and cost-accounted. Nothing else in the project talks to a model.

Caching is not an optimisation here, it is a correctness requirement: the design says
traces must be reproducible and re-scorable without re-running the model (A17), so an
identical prompt must always yield the identical decision.

Subprocess invocation ported from glc_v5's ClaudeCLIProvider
(D:\\Projects\\Assignment17\\glc_v5\\glc\\providers.py), which found and fixed a real bug
worth carrying over rather than rediscovering:

  The prompt must go over STDIN, not argv. `claude -p "<the whole prompt>"` puts the
  entire prompt on the command line, which hits Windows' ~32K argv length limit on
  anything but a short prompt (WinError 206) -- and this harness's prompts (telemetry
  panel + decision rules + tool catalogue) are routinely several KB. `-p` with NO value
  is documented `claude` CLI behaviour for "read the prompt from stdin instead" -- no
  length ceiling, and no shell-escaping of the prompt needed either.

Also carried over: `--output-format json` (structured reply with a `result` field and
real `usage` token counts, instead of parsing raw stdout text), `--tools ""` (this is a
plain text-completion adapter, not an agentic coding session -- tool use stays off on
the CLI side), and an explicit full `--model` id rather than a bare alias (glc_v5 found
that alias names can resolve to an older dated snapshot instead of the latest one).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CACHE_DIR = os.environ.get("LLM_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "llm_cache"))


class LlmError(RuntimeError):
    pass


# Failures worth replaying: the process died, was overloaded, or timed out. A refusal or a
# malformed *reply* is NOT transient -- that is the parser's business, and retrying it
# would be the hidden second opinion the design forbids.
_TRANSIENT = ("exited 1", "exited -", "timeout after", "overloaded", "rate limit",
              "rate_limit", "429", "500", "502", "503", "529", "connection",
              "non-JSON output")


def _transient(e: Exception) -> bool:
    m = str(e).lower()
    return any(t.lower() in m for t in _TRANSIENT)


class ClaudeCliProvider:
    """Routes prompts through the locally authenticated Claude Code CLI (`claude -p`)."""

    name = "claude-cli"

    def __init__(self, cache: bool = True, timeout_s: int = 180,
                 model: str | None = None, system: str | None = None,
                 effort: str | None = None):
        self.cache = cache
        self.timeout_s = timeout_s
        self.model = model or CLAUDE_MODEL
        # Optional: split off a system prompt via --append-system-prompt instead of
        # folding it into the user prompt string. Every existing caller in this repo
        # builds one combined prompt and leaves this unset -- that keeps working
        # unchanged; new callers may pass it for a cleaner separation.
        self.system = system
        # TOKEN BUDGET (capstone): the CLI's --effort was never passed, so every call ran
        # at whatever the CLI's own default is. Measured on a real batch: the reply itself
        # is ~130-150 tokens (checked directly against raw_response), but usage.output_tokens
        # averaged 2,634/call -- ~2,500 tokens/call of invisible extended thinking that never
        # appears in the trace. This is a fixed-schema, bounded decision task (one panel in,
        # one of 12 actions out), not open-ended reasoning; --effort lets that cost be capped
        # without touching the reply itself. Unset by default (LLM_EFFORT env var opts in)
        # so existing cached runs and behaviour are untouched until this is validated.
        self.effort = effort or os.environ.get("LLM_EFFORT") or None
        self.calls = 0
        self.cache_hits = 0
        self.errors = 0
        self.retries = 0
        self.max_attempts = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
        self.backoff_s = float(os.environ.get("LLM_BACKOFF_S", "2.0"))
        self.total_wall_s = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _key(self, prompt: str) -> str:
        h = hashlib.sha256(
            self.model.encode() + b"\x00" + (self.system or "").encode() + b"\x00"
            + (self.effort or "").encode() + b"\x00" + prompt.encode()).hexdigest()[:32]
        return os.path.join(CACHE_DIR, h + ".json")

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Cache -> call -> (bounded transport retry) -> cache.

        THE RETRY IS TRANSPORT-LEVEL AND IT IS NOT THE PARSE REPAIR.
        DESIGN section 5 commits to "no hidden retries": one parse repair, recorded as
        parse_mode "repaired", then abstain. That governs what to do with a reply the model
        actually produced, and it is unchanged -- harness/parser.py still gets exactly one
        repair. This is the other failure: no reply was produced at all, because the CLI
        died. Replaying an unanswered prompt is not a second opinion, it is the first one.

        Why it is needed: at 32 concurrent CLI processes, 64.6% of teacher decisions came
        back "claude exited 1" in a full-corpus run -- 766 of 1186. The identical prompt
        replayed on its own returned a clean, correct answer, so nothing was wrong with the
        prompt or the model; the failure is purely contention. Measured after: 16/16 at
        4, 8 and 16 workers. Concurrency is now capped AND transient failures are retried,
        because 540 episodes x ~20 calls turns even a 1% failure rate into 100+ silently
        lost decisions, each one an abstention the verifier has to score.

        Retries are counted (`self.retries`) so they appear in the run summary rather than
        hiding a degraded provider behind a healthy-looking result.
        """
        if system is not None:
            self.system = system      # enters the cache key via _key()
        path = self._key(prompt)
        if self.cache and os.path.exists(path):
            self.cache_hits += 1
            return json.load(open(path))["response"]

        last = None
        for attempt in range(self.max_attempts):
            if attempt:
                self.retries += 1
                time.sleep(self.backoff_s * (2 ** (attempt - 1)))
            try:
                return self._complete_once(prompt, path)
            except LlmError as e:
                last = e
                if not _transient(e):
                    raise
        raise LlmError(f"{self.max_attempts} attempts failed; last: {last}")

    def _complete_once(self, prompt: str, path: str) -> str:

        cmd = [CLAUDE_BIN, "-p", "--output-format", "json", "--tools", "",
               "--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        if self.system:
            # --system-prompt REPLACES the Claude Code CLI's own system prompt and
            # tool catalogue. Appending would keep ~25K tokens of agent scaffold we
            # have no use for: this is a text-completion call, not a coding session.
            cmd += ["--system-prompt", self.system]

        t0 = time.time()
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            self.errors += 1
            raise LlmError(f"timeout after {self.timeout_s}s") from e
        except FileNotFoundError as e:
            self.errors += 1
            raise LlmError(
                f"'{CLAUDE_BIN}' not found on PATH -- install the Claude Code CLI, "
                f"or set CLAUDE_BIN to its full path") from e
        self.calls += 1
        self.total_wall_s += time.time() - t0

        if r.returncode != 0:
            self.errors += 1
            # stderr alone is not enough to diagnose a failure: the CLI can exit
            # non-zero with an error body on stdout (e.g. --output-format json
            # still emitting {"is_error": true, ...}) and empty stderr. A prior
            # concurrent-workers run hit this blind spot -- 14 straight failures
            # logged only "claude exited 1: " with nothing else to go on.
            detail = r.stderr.strip()[:300] or r.stdout.strip()[:300] or "(no stdout or stderr)"
            raise LlmError(f"claude exited {r.returncode}: {detail}")

        try:
            payload = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            self.errors += 1
            raise LlmError(f"claude returned non-JSON output: {r.stdout[:300]}") from e

        if payload.get("is_error"):
            self.errors += 1
            raise LlmError(f"claude error: {str(payload.get('result', ''))[:300]}")

        out = (payload.get("result") or "").strip()
        usage = payload.get("usage") or {}
        self.total_input_tokens += usage.get("input_tokens", 0) or 0
        self.total_output_tokens += usage.get("output_tokens", 0) or 0
        self.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0

        if self.cache:
            json.dump({"prompt": prompt, "response": out, "model": self.model,
                       "usage": usage}, open(path, "w"))
        return out

    def stats(self) -> dict:
        return {"provider": self.name, "model": self.model, "calls": self.calls,
                "cache_hits": self.cache_hits, "errors": self.errors,
                "mean_wall_s": round(self.total_wall_s / max(1, self.calls), 1),
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cache_read_input_tokens": self.total_cache_read_tokens}


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates fences/prose).

    Kept even though `--output-format json` means `complete()` already unwraps the
    CLI's own envelope: the model's *reply text* can still be fenced or wrapped in
    prose if it didn't follow the "reply with ONLY JSON" instruction -- that's exactly
    what harness/parser.py's repair-once-then-abstain path exists to catch.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        raise LlmError(f"no JSON object in reply: {text[:200]}")
    return json.loads(t[i:j + 1])


class StubProvider:
    """A deterministic, offline stand-in for the model.

    The LLM path -- render the panel, ask, parse, validate against the contract, record the
    trace -- is a lot of machinery, and none of it can be exercised in CI or on a machine
    with no model access. Without a stub, "does the harness work?" and "is the model any
    good?" are the same question, and a plumbing bug hides behind a bad answer.

    So this answers from the FEATURE VECTOR using the same physics the design documents:
    a raised noise floor means an emitter, the per-channel spread separates barrage from
    spot, TX-shadow loss means reactive. It is not a model and must never be reported as
    one -- it exists so the harness can be tested with the model absent.
    """

    name = "stub"

    def __init__(self, **_kw):
        self.calls = 0
        self.cache_hits = 0

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        import json as _json

        def val(name: str, default: float = 0.0) -> float:
            """Read the panel's VALUE COLUMN.

            render_panel() lays each row out as f"{name:<24}{value:>7} {bar}{level}  {desc}",
            so the number is the first token after the name -- not a trailing float at the
            end of the line, which is where the description lives. Getting this wrong made
            every feature read as its default and the stub answered "fading" to everything;
            a good reminder that a stub with a parsing bug is worse than no stub, because it
            fails quietly and looks like a bad model.
            """
            for line in prompt.splitlines():
                if line.startswith(name) and len(line) > len(name):
                    tok = line[len(name):].strip().split()
                    if tok:
                        t = tok[0].rstrip("%")
                        try:
                            v = float(t)
                        except ValueError:
                            continue
                        # decoded percentages come back as 0..100; renormalise to [-1,+1]
                        return (v / 50.0 - 1.0) if tok[0].endswith("%") else v
            return default

        noise = val("noise_delta_base", -1.0)
        bad = val("scan_bad_frac", -1.0)
        spread = val("scan_noise_spread", 0.0)
        period = val("scan_periodicity", -1.0)
        shadow = val("tx_shadow_loss_delta", -1.0)
        hb = val("heartbeat_gap", -1.0)
        load = val("offered_load", -1.0)
        spr = val("pdr_spread", -1.0)

        if shadow >= 0.3:
            cause = "reactive"
        elif noise >= 0.3 or bad >= 0.0:
            if period >= 0.3:
                cause = "sweep"
            elif bad >= 0.9 and spread <= 0.3:
                cause = "barrage"
            else:
                cause = "spot"
        elif hb >= 0.4 or spr >= 0.4:
            cause = "node_loss"
        elif load >= 0.3:
            cause = "congestion"
        else:
            cause = "fading"

        call = {"barrage": "fallback_to_lora", "spot": "hop_channel", "sweep": "hop_channel",
                "reactive": "change_tdma_slot", "fading": "set_tx_power",
                "node_loss": "reroute", "congestion": "change_tdma_slot",
                "hidden_term": "change_tdma_slot"}[cause]
        if "spectrum_scan" in prompt and bad <= -0.9 and noise <= 0.0:
            call, cause = "spectrum_scan", cause      # buy evidence before committing

        belief = {c: 0.02 for c in ["barrage", "spot", "reactive", "sweep", "fading",
                                    "node_loss", "congestion", "hidden_term"]}
        belief[cause] = 0.86
        return _json.dumps({"belief": belief, "call": call, "args": {}, "confidence": 0.86,
                            "unrecoverable": 0.0,
                            "why": f"stub: physics rules -> {cause}"})

    def stats(self) -> dict:
        return {"provider": self.name, "calls": self.calls, "cache_hits": 0, "mean_wall_s": 0.0}


def make_provider(kind: str = "auto", **kw):
    """`claude` for the real teacher, `stub` for offline testing, `auto` to fall back."""
    if kind == "stub":
        return StubProvider(**kw)
    p = ClaudeCliProvider(**kw)
    if kind == "claude":
        return p
    try:                                   # auto: probe once, fall back if unavailable
        p.complete('Reply with ONLY: {"ok":true}')
        return p
    except Exception as e:                                   # noqa: BLE001
        print(f"[provider] claude unavailable ({str(e)[:60]}); using StubProvider", flush=True)
        return StubProvider(**kw)
