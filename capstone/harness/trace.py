"""Records-first scoring: an append-only JSONL trace of every decision.

Every record holds the full prompt, the raw response, the parsed decision, latency and
cache status -- so any reviewer can replay a run offline, without a model or an API key.
That is what makes an LLM-driven result auditable instead of anecdotal.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict


@dataclass
class DecisionRecord:
    episode: str
    step: int
    t_sim: float
    agent: str
    scenario_family: str = ""
    prompt: str = ""
    raw_response: str = ""
    parse_mode: str = ""            # clean | repaired | failed
    belief: dict = field(default_factory=dict)
    call: str = ""
    args: dict = field(default_factory=dict)
    confidence: float = 0.0
    why: str = ""
    validation_note: str = ""
    cost_charged: int = 0
    latency_s: float = 0.0
    cache_hit: bool = False
    error: str = ""
    # The features the teacher actually saw. Without these a trace records WHAT the teacher
    # said but not WHAT IT SAW, so it can be replayed and audited but never distilled --
    # which is the one thing the traces exist for (DESIGN v2.0 section 9.5 step 2).
    features: list = field(default_factory=list)
    available: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class TraceStore:
    def __init__(self, path: str, keep_prompts: bool = True):
        self.path = path
        self.keep_prompts = keep_prompts
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._fh = open(path, "a", buffering=1)
        self.n = 0
        self.started = time.time()

    def append(self, rec: DecisionRecord) -> None:
        if not self.keep_prompts:
            rec.prompt = ""
        self._fh.write(rec.to_json() + "\n")
        self.n += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---- replay ----------------------------------------------------------
    @staticmethod
    def load(path: str) -> list[dict]:
        out = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    @staticmethod
    def summary(path: str) -> dict:
        recs = TraceStore.load(path)
        if not recs:
            return {"records": 0}
        modes, calls = {}, {}
        for r in recs:
            modes[r.get("parse_mode", "")] = modes.get(r.get("parse_mode", ""), 0) + 1
            calls[r.get("call", "")] = calls.get(r.get("call", ""), 0) + 1
        eps = {r["episode"] for r in recs}
        lat = [r.get("latency_s", 0.0) for r in recs]
        return {"records": len(recs), "episodes": len(eps), "parse_modes": modes,
                "calls": dict(sorted(calls.items(), key=lambda kv: -kv[1])),
                "mean_latency_s": round(sum(lat) / len(lat), 2),
                "cache_hit_rate": round(
                    sum(1 for r in recs if r.get("cache_hit")) / len(recs), 3)}
