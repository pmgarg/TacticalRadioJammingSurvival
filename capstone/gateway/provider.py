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


class ClaudeCliProvider:
    """Routes prompts through the locally authenticated Claude Code CLI (`claude -p`)."""

    name = "claude-cli"

    def __init__(self, cache: bool = True, timeout_s: int = 180,
                 model: str | None = None, system: str | None = None):
        self.cache = cache
        self.timeout_s = timeout_s
        self.model = model or CLAUDE_MODEL
        # Optional: split off a system prompt via --append-system-prompt instead of
        # folding it into the user prompt string. Every existing caller in this repo
        # builds one combined prompt and leaves this unset -- that keeps working
        # unchanged; new callers may pass it for a cleaner separation.
        self.system = system
        self.calls = 0
        self.cache_hits = 0
        self.errors = 0
        self.total_wall_s = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _key(self, prompt: str) -> str:
        h = hashlib.sha256(
            self.model.encode() + b"\x00" + (self.system or "").encode() + b"\x00"
            + prompt.encode()).hexdigest()[:32]
        return os.path.join(CACHE_DIR, h + ".json")

    def complete(self, prompt: str) -> str:
        path = self._key(prompt)
        if self.cache and os.path.exists(path):
            self.cache_hits += 1
            return json.load(open(path))["response"]

        cmd = [CLAUDE_BIN, "-p", "--output-format", "json", "--tools", "",
               "--model", self.model]
        if self.system:
            cmd += ["--append-system-prompt", self.system]

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
            raise LlmError(f"claude exited {r.returncode}: {r.stderr.strip()[:300]}")

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
