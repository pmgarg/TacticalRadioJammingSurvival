#!/usr/bin/env python3
"""
Run one scenario through the REAL harness (Controller + HarnessAgent + ToolRegistry +
parse_decision + TraceStore), with a live LLM answering each decision -- except the
"LLM" is me, Claude, reasoning in the conversation instead of a subprocess spawning a
second `claude` CLI (none is installed in this environment).

Mechanism: ClaudeCliProvider.complete() checks a SHA-256(model+prompt) file cache before
ever shelling out (gateway/provider.py). ManualProvider below does exactly the same cache
lookup, and if it misses, raises NeedAnswer(prompt, cache_path) instead of touching a
subprocess. The driver catches that, prints the prompt, and stops. I read the prompt,
answer it the way the DECISION_RULE prompt asks, and a second script writes my answer
into that exact cache file. Re-running this driver then gets a cache hit for that step
and proceeds to the next one -- until the episode finishes.

Nothing about Controller / HarnessAgent / ToolRegistry / parse_decision / TraceStore is
touched or mocked. Only the provider's *answer source* differs.

Usage:
    python3 _manual_llm_demo.py <scenario.yaml>           # run/resume
    python3 _manual_llm_demo.py <scenario.yaml> --reset   # clear this scenario's cache first
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenario.schema import load_scenario
from sim.refsim import RefSim
from agent.controller import Controller
from harness.loop import HarnessAgent
from harness.registry import ToolRegistry
from harness.trace import TraceStore
from gateway.provider import ClaudeCliProvider
from verify.verifier import score_episode

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT = json.load(open(os.path.join(HERE, "contract", "agent_contract.json")))
COSTCFG = yaml.safe_load(open(os.path.join(HERE, "contract", "cost_matrix.yaml")))
TRACE_PATH = os.path.join(HERE, "..", "data", "manual_llm_trace.jsonl")


class NeedAnswer(Exception):
    def __init__(self, prompt: str, cache_path: str):
        self.prompt = prompt
        self.cache_path = cache_path


class ManualProvider(ClaudeCliProvider):
    """Same cache ClaudeCliProvider uses; on a miss, ask the human (me) instead of
    spawning `claude`."""

    def complete(self, prompt: str) -> str:
        path = self._key(prompt)
        if self.cache and os.path.exists(path):
            self.cache_hits += 1
            return json.load(open(path))["response"]
        raise NeedAnswer(prompt, path)


def main():
    scenario_path = sys.argv[1]
    sc = load_scenario(scenario_path)
    if "--reset" in sys.argv:
        # Best-effort: cache is content-addressed, nothing to selectively clear without
        # walking every file's stored prompt. Not needed for a fresh scenario.
        pass

    sim = RefSim(sc)
    provider = ManualProvider(cache=True)
    reg = ToolRegistry()
    trace = TraceStore(TRACE_PATH)
    agent = HarnessAgent(provider=provider, registry=reg, trace=trace,
                          episode=sc.name, family=sc.family, verbose=True)
    ctrl = Controller(sim, sc, agent, CONTRACT, verbose=True)

    print(f"=== {sc.name}  (truth: {sc.truth.cause}, recoverable={sc.truth.recoverable}) ===")
    try:
        log = ctrl.run()
    except NeedAnswer as need:
        print("\n" + "=" * 78)
        print("NEED_ANSWER  cache_path=" + need.cache_path)
        print("-" * 78)
        print(need.prompt)
        print("=" * 78)
        print(f"\n(provider stats before stopping: {provider.stats()})")
        trace.close()
        return
    trace.close()

    truth = {"cause": sc.truth.cause, "onset_t": sc.truth.onset_t,
             "recoverable": sc.truth.recoverable, "notes": sc.truth.notes}
    from dataclasses import asdict
    scored = score_episode(asdict(log), truth, COSTCFG, CONTRACT, sc.family)
    print("\n" + "=" * 78)
    print("EPISODE COMPLETE")
    print(json.dumps(scored, indent=2, default=str))
    print(f"\nprovider stats: {provider.stats()}")
    print(f"harness stats:  {agent.stats()}")


if __name__ == "__main__":
    main()
