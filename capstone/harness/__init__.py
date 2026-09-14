"""Our own agent harness. No LangChain, no LangGraph, no framework.

Four pieces, each small enough to read in one sitting:
  registry.py  typed tool catalogue, generated from contract/agent_contract.json
  parser.py    strict JSON extraction with exactly one repair attempt, then abstain
  trace.py     append-only JSONL record of every decision, replayable offline
  loop.py      the agent loop itself: observe -> render -> ask -> validate -> act -> record
"""
from .registry import ToolRegistry, ToolSpec          # noqa: F401
from .parser import parse_decision, ParseOutcome      # noqa: F401
from .trace import TraceStore, DecisionRecord         # noqa: F401
