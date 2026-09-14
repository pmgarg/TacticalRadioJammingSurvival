"""A tiny rule DSL, and an evaluator for it.

The point of this file is that the agent's policy can be WRITTEN DOWN, argued with, and
diffed. A 12,756-parameter network is a good controller and a terrible explanation; a
rule set is the opposite. We keep both: the network decides, and the rules are the
auditable statement of what it is supposed to be doing, checked against it.

Grammar (deliberately too small to hide anything in):

    rule  := { "id", "if": [clause, ...], "then": {...}, "note" }
    clause:= [feature_name, op, value]        op in  > >= < <= == !=
    then  := {"hypothesis": <cause>}  or  {"call": <tool>}  or both

All clauses must hold (conjunction only). Disjunction is expressed as two rules. There is
no arithmetic, no nesting and no negation beyond the operators, so a rule cannot quietly
become a second model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

from percept.features import FEATURE_NAMES

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
OPS = {
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: abs(a - b) < 1e-6,
    "!=": lambda a, b: abs(a - b) >= 1e-6,
}


@dataclass
class Rule:
    id: str
    clauses: list                      # [[feature, op, value], ...]
    then: dict                         # {"hypothesis": ...} and/or {"call": ...}
    note: str = ""
    support: int = 0                   # episodes where the antecedent fired
    precision: float = 0.0             # of those, fraction where `then` was right

    def matches(self, f: list[float]) -> bool:
        for c in self.clauses:
            if len(c) != 3:
                return False
            name, op, val = c
            i = IDX.get(name)
            if i is None or i >= len(f) or op not in OPS:
                return False
            try:
                if not OPS[op](f[i], float(val)):
                    return False
            except (TypeError, ValueError):
                return False
        return True


@dataclass
class RuleSet:
    rules: list = field(default_factory=list)
    version: str = "v0"
    provenance: str = ""

    # ---- validation ------------------------------------------------------
    @staticmethod
    def validate_rule(d: dict) -> tuple[bool, str]:
        if not isinstance(d, dict):
            return False, "not an object"
        cl = d.get("if") or d.get("clauses")
        if not isinstance(cl, list) or not cl:
            return False, "no clauses"
        for c in cl:
            if not (isinstance(c, list) and len(c) == 3):
                return False, f"bad clause {c!r}"
            if c[0] not in IDX:
                return False, f"unknown feature {c[0]!r}"
            if c[1] not in OPS:
                return False, f"unknown op {c[1]!r}"
            try:
                float(c[2])
            except (TypeError, ValueError):
                return False, f"non-numeric threshold {c[2]!r}"
        th = d.get("then")
        if not isinstance(th, dict) or not ({"hypothesis", "call"} & set(th)):
            return False, "then must set hypothesis and/or call"
        return True, ""

    @classmethod
    def from_dicts(cls, ds: list[dict], version="v1", provenance="") -> tuple["RuleSet", list]:
        rules, rejected = [], []
        for i, d in enumerate(ds):
            ok, why = cls.validate_rule(d)
            if not ok:
                rejected.append({"index": i, "reason": why, "rule": d})
                continue
            rules.append(Rule(id=str(d.get("id", f"r{i}")),
                              clauses=d.get("if") or d.get("clauses"),
                              then=d["then"], note=str(d.get("note", ""))[:200]))
        return cls(rules, version, provenance), rejected

    # ---- application -----------------------------------------------------
    def apply(self, f: list[float]) -> list[Rule]:
        return [r for r in self.rules if r.matches(f)]

    def suggest(self, f: list[float]) -> dict:
        """First matching rule wins; order is the priority. Explicit, not learned."""
        for r in self.rules:
            if r.matches(f):
                return {"rule": r.id, **r.then, "note": r.note}
        return {}

    # ---- io --------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        json.dump({"version": self.version, "provenance": self.provenance,
                   "rules": [asdict(r) for r in self.rules]},
                  open(path, "w"), indent=1)

    @classmethod
    def load(cls, path: str) -> "RuleSet":
        d = json.load(open(path))
        return cls([Rule(**r) for r in d["rules"]], d.get("version", "v0"),
                   d.get("provenance", ""))

    def render(self) -> str:
        out = [f"# rule set {self.version}  ({len(self.rules)} rules)  {self.provenance}"]
        for r in self.rules:
            cond = " AND ".join(f"{c[0]} {c[1]} {c[2]}" for c in r.clauses)
            act = ", ".join(f"{k}={v}" for k, v in r.then.items())
            out.append(f"[{r.id}] IF {cond}\n      THEN {act}"
                       f"   (support={r.support} precision={r.precision:.2f})"
                       + (f"\n      # {r.note}" if r.note else ""))
        return "\n".join(out)
