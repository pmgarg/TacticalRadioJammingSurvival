"""
LLM gateway (the glc_v5 / ClaudeCLIProvider role from the HLD §11).

One place where every LLM call goes, so the teacher pipeline is provider-agnostic,
cached, audited and cost-accounted. Nothing else in the project talks to a model.

Caching is not an optimisation here, it is a correctness requirement: the design says
traces must be reproducible and re-scorable without re-running the model (A17), so an
identical prompt must always yield the identical decision.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "/opt/node22/bin/claude")
CACHE_DIR = os.environ.get("LLM_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "llm_cache"))


class LlmError(RuntimeError):
    pass


class ClaudeCliProvider:
    """Routes prompts through the Claude Code CLI (`claude -p`)."""

    name = "claude-cli"

    def __init__(self, cache: bool = True, timeout_s: int = 180,
                 model: str | None = None):
        self.cache = cache
        self.timeout_s = timeout_s
        self.model = model
        self.calls = 0
        self.cache_hits = 0
        self.total_wall_s = 0.0
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _key(self, prompt: str) -> str:
        h = hashlib.sha256((self.model or "default").encode() + b"\x00"
                           + prompt.encode()).hexdigest()[:32]
        return os.path.join(CACHE_DIR, h + ".json")

    def complete(self, prompt: str) -> str:
        path = self._key(prompt)
        if self.cache and os.path.exists(path):
            self.cache_hits += 1
            return json.load(open(path))["response"]
        cmd = [CLAUDE_BIN, "-p", prompt]
        if self.model:
            cmd += ["--model", self.model]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise LlmError(f"timeout after {self.timeout_s}s") from e
        self.calls += 1
        self.total_wall_s += time.time() - t0
        if r.returncode != 0:
            raise LlmError(f"claude exited {r.returncode}: {r.stderr[:300]}")
        out = r.stdout.strip()
        if self.cache:
            json.dump({"prompt": prompt, "response": out, "model": self.model},
                      open(path, "w"))
        return out

    def stats(self) -> dict:
        return {"provider": self.name, "calls": self.calls,
                "cache_hits": self.cache_hits,
                "mean_wall_s": round(self.total_wall_s / max(1, self.calls), 1)}


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates fences/prose)."""
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

    def complete(self, prompt: str) -> str:
        self.calls += 1
        import json as _json
        import re as _re

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
