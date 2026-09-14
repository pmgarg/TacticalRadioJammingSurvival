"""Typed tool catalogue.

Generated FROM contract/agent_contract.json so the tools the LLM may call and the tools
the simulator implements cannot drift apart. One source of truth, mechanically enforced.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

CONTRACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "contract", "agent_contract.json")


@dataclass
class ToolSpec:
    name: str
    kind: str                       # "diagnose" | "act"
    cost: int                       # information cost units (diagnose) or 0
    budget: int                     # budget slots consumed
    duration_ms: int
    args: dict = field(default_factory=dict)
    separates: list = field(default_factory=list)
    cls: str = ""                   # free|mild|costly|expensive|terminal (act only)

    def signature(self) -> str:
        a = ", ".join(f"{k}: {v}" for k, v in self.args.items()) or ""
        return f"{self.name}({a})"


class ToolRegistry:
    """The set of calls an agent may make, with prices, arg types and validation."""

    def __init__(self, contract_path: str = CONTRACT):
        self.contract = json.load(open(contract_path))
        self.tools: dict[str, ToolSpec] = {}
        for name, d in self.contract.get("diagnose", {}).items():
            self.tools[name] = ToolSpec(name, "diagnose", int(d.get("cost", 0)),
                                        int(d.get("budget", 0)),
                                        int(d.get("duration_ms", 0)),
                                        d.get("args", {}) or {},
                                        d.get("separates", []) or [])
        for name, d in self.contract.get("act", {}).items():
            self.tools[name] = ToolSpec(name, "act", 0, int(d.get("budget", 0)), 0,
                                        d.get("args", {}) or {}, [],
                                        d.get("class", ""))

    # ---- introspection ---------------------------------------------------
    def diagnostics(self) -> list[ToolSpec]:
        return [t for t in self.tools.values() if t.kind == "diagnose"]

    def actions(self) -> list[ToolSpec]:
        return [t for t in self.tools.values() if t.kind == "act"]

    def hypotheses(self) -> list[str]:
        return [h["id"] for h in self.contract["hypotheses"] if h["id"] != "unknown"]

    # ---- prompt surface --------------------------------------------------
    def render_catalogue(self, available: list[str] | None = None) -> str:
        """The tool block we put in front of the model. Costs included on purpose:
        an agent that does not know the price cannot reason economically."""
        out = ["DIAGNOSTIC TESTS  (cost, then what they separate)"]
        for t in sorted(self.diagnostics(), key=lambda x: x.cost):
            if available is not None and t.name not in available:
                continue
            sep = "; ".join(t.separates)
            out.append(f"  {t.signature():<44} cost={t.cost} {t.duration_ms}ms  {sep}")
        out.append("RECOVERY ACTIONS  (budget slots consumed)")
        for t in self.actions():
            if available is not None and t.name not in available:
                continue
            out.append(f"  {t.signature():<44} budget={t.budget} class={t.cls}")
        return "\n".join(out)

    # ---- validation ------------------------------------------------------
    def validate(self, call: str, args: dict, available: list[str]) -> tuple[str, dict, str]:
        """Return (call, args, note). Never raises: an invalid call degrades to no_op
        with a recorded reason, because a crashed teacher poisons a whole episode."""
        if call not in self.tools:
            return "no_op", {}, f"unknown tool {call!r}"
        if call not in available:
            return "no_op", {}, f"{call} is masked at this step"
        spec = self.tools[call]
        clean, dropped = {}, []
        for k, v in (args or {}).items():
            if k in spec.args:
                clean[k] = v
            else:
                dropped.append(k)
        note = f"dropped args {dropped}" if dropped else ""
        return call, clean, note

    def cost_of(self, call: str) -> int:
        t = self.tools.get(call)
        return t.cost if t else 0
