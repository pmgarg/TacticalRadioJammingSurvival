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
