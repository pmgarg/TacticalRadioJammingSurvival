# AGENTIC PLAN — re-reading the instructor's brief, and the corrected build

**Status date:** 2026-09-11
**Companion to:** `docs/ARCHITECTURE.md` (what exists), `docs/DESIGN.md` (the merged HLD)

---

## 0. The correction, stated plainly

The brief asks for **an agentic harness with an LLM in the teacher, and a complete agentic
loop for the student running against SDR.** What we built is an excellent *privileged-expert
distillation* system: a rule-based oracle that is handed the ground-truth cause, plays a
scripted playbook, and teaches a small net to imitate it.

That is a legitimate ML result. It is **not** the assignment. The oracle cannot *reason*
about a situation the instructor invents on the spot, because it does not reason at all — it
looks up `true_cause` in a dict. If the instructor says "now the jammer also spoofs beacons
from a dead node", the oracle has no entry and the student has never seen it.

So the teacher must become an LLM that reads telemetry, forms hypotheses, calls diagnostic
tools, and *writes down the rules* — and the student must be distilled from that, inside a
harness we built ourselves.

**Constraint, explicit:** no LangChain, no LangGraph, no agent framework of any kind. The
harness is ours. It is ~600 lines of plain Python and that is a feature, not a compromise —
every control-flow decision is legible, every tool call is recorded, and there is no hidden
retry, hidden prompt-mutation, or hidden state to explain to a grader.

---

## 1. Requirement-by-requirement audit against the brief

| # | Instructor requirement | Status | What closes it |
|---|---|---|---|
| R1 | **Build your own agent harness** (no framework) | 🟡 **half** — `gateway/provider.py` is ours and works, but it is a completion client, not a harness: no tool registry, no loop, no trace store | §3 — `capstone/harness/` |
| R2 | **LLM in the teacher** driving decisions | 🟡 **proven, not wired** — `llm_teacher.py` decides correctly on 3/3 families but teaches nothing | §4 Week 1 |
| R3 | **Wrap an open-source tool** | ✅ **ns-3.45** is wrapped: scenario JSON in, telemetry + live socket out, and we upstream-patched a real bug in it | already done, needs writing up as such |
| R4 | **Task set with deterministic predicates** | ✅ 540 scenarios, `verify/verifier.py` predicates, hash-split | already done |
| R5 | **Mutation corpus + detection-rate sweep** | ❌ **missing** | §4 Week 2 |
| R6 | **Records-first scoring** (every decision traceable) | 🟡 `EpisodeLog` exists; LLM prompts/responses are not in it | §3 `TraceStore` |
| R7 | **Refusal task** | ✅ 100 % on every variant, structurally enforced | already done |
| R8 | **Student agentic loop against SDR** | ❌ Phase 2 | §4 Week 4 |

Three of eight are genuinely missing. Two are half-built. That is a two-week hole, not a
rewrite — the physics, the corpus, the simulator and the safety envelope all survive
unchanged, because they were built behind a frozen contract.

---

## 2. What "modern agentic" means here, concretely

Not buzzwords — five specific design choices, each with a reason and each cheap to defend.

**2.1 Tools are typed, and the schema is generated from the contract.**
`contract/agent_contract.json` already defines nine diagnostic tools with costs, durations
and argument types. The harness emits the LLM tool schema *from that file*, so the tools the
LLM can call and the tools the simulator implements can never drift apart. One source of
truth, mechanically enforced.

**2.2 The LLM sees a panel, not a float vector.**
Handing a model `[-0.31, 0.88, ...]` is throwing away the only thing an LLM is good at.
`render_state()` presents 23 curated features as a table with plain-English meanings and
units. This is why the LLM correctly cited `tx_shadow_loss_delta` and `silent_loss_rate`
when diagnosing `reactive` — the exact family the numeric student handles worst.

**2.3 Cost-awareness is in the prompt *and* in the envelope.**
The LLM is told every tool's price and its remaining budget, so it reasons economically. But
it is also physically masked by `controller.py`, so a hallucinated `hop_channel` under a
fading diagnosis is *refused*, not merely discouraged. **Belt and braces: the safety property
is structural, and the LLM's job is to be efficient inside it, not to be trusted.**

**2.4 The LLM writes the policy, not just the answer.**
Beyond per-step decisions, the teacher runs a **policy-induction pass**: given a batch of
episodes with outcomes, it proposes candidate rules in a tiny DSL
(`if <feature> <op> <value> and ... then <hypothesis>/<tool>`). Each proposed rule is then
*evaluated against the held-out corpus* and kept only if it improves expected cost. The LLM
proposes; the data disposes. This is the piece that satisfies "help us define the tools,
rules and policies for the student" — and it produces an artefact a human can read and
argue with.

**2.5 Distillation is trace-level, with caching that makes it affordable.**
An LLM call is ~7–17 s. 540 scenarios × ~30 decisions is prohibitive at full price. Three
mitigations, all already partly built: SHA-256 prompt caching (repeat states are free);
**teacher-on-disagreement** — run the cheap oracle everywhere, invoke the LLM only where the
student and oracle disagree or the student's confidence is low (this is DAgger's
active-learning argument, and it cuts LLM calls by ~10×); and a **golden set** of ~200 fully
LLM-labelled episodes for the headline numbers.

---

## 3. The harness we are building (R1, R6)

```
capstone/harness/
  registry.py    ToolSpec + ToolRegistry; schema generated from agent_contract.json
  loop.py        AgentLoop: observe → render → call LLM → parse → validate → execute → record
  parser.py      strict JSON extraction, repair-once, then abstain (never silently guess)
  trace.py       TraceStore: append-only JSONL, one record per decision, includes the
                 full prompt, the raw response, latency, cache hit/miss, tool result
  policy.py      RuleSet DSL + evaluator; load/save; LLM-proposed rules
  induce.py      policy-induction pass: batch episodes → LLM → candidate rules → corpus eval
  mutate.py      mutation corpus generator + detection-rate sweep
```

Design commitments:

* **No hidden retries.** One repair attempt on malformed JSON, logged as such; then abstain.
  An abstention is a recorded outcome, not an error.
* **Every LLM call is replayable.** The TraceStore holds prompt + response + provider stats,
  so any reviewer can re-derive a decision offline without an API key.
* **The loop is synchronous and single-threaded.** Concurrency is added only at the
  *episode* level. Debuggability beats throughput for a capstone.
* **Failure is a first-class outcome.** Provider timeout, unparseable response and
  budget-exhausted are distinct recorded states, and the verifier scores them.

---

## 4. Revised 4-week calendar (2 people, ~8 person-weeks)

Weeks are ordered so that **the thing the brief actually grades lands first**.

### Week 1 — Close the LLM loop (owner: both, it is the critical path)
* `harness/registry.py` + `loop.py` + `trace.py` + `parser.py`
* Wire `LlmTeacherAgent` into `gen_traces.py`, `dagger.py`, `evaluate.py`
* **Golden set:** 200 episodes fully LLM-labelled, traces on disk
* **Agreement study:** LLM teacher vs privileged oracle — per-family confusion matrix, and
  crucially the cases where the LLM is *right and the oracle's playbook is wrong*
* **Exit gate:** a student trained on LLM traces alone scores within 5 pts of the
  oracle-trained student, and FP(fading→jam) stays 0.000

### Week 2 — Policy induction + mutation corpus (R5, G2)
* `policy.py` + `induce.py`: LLM proposes rules, corpus evaluates, keep-if-improves
* Ship `data/policy_v1.yaml` — human-readable rules the student's head is regularised toward
* `mutate.py`: perturb scenarios along 6 axes (jammer power ±6 dB, onset ±5 s, topology
  jitter, duty cycle, node count, channel plan) → `data/mutations.jsonl`
* **Detection-rate sweep:** accuracy vs mutation magnitude, per family — this is the
  robustness curve the brief wants
* Register the harness on the EAG V3 platform (R3/G3)
* **Exit gate:** ≥50 mutations per family, sweep curve plotted, no cliff below 3 dB

### Week 3 — Fix `reactive`, harden, re-measure (G5)
* Diagnose the closed-loop feedback pathology: the agent's own `silent_listen` changes the
  statistic it is measuring. Candidate fix: **randomised probing schedule** so the reactive
  jammer cannot phase-lock, plus a paired-sample estimator over interleaved TX/silent windows
  rather than sequential ones
* Full re-run: 540 scenarios, refsim + ns-3, all four agents (oracle, LLM, student, baseline)
* **Exit gate:** `reactive` ≥70 % offline and ≥3/5 live; every headline number in
  `RESULTS.md` regenerated from one reproducible command

### Week 4 — Demo, docs, hardware stretch
* **Instructor-proof demo:** a CLI where the instructor describes *any* scenario in English →
  the LLM teacher writes the scenario JSON → ns-3 runs it → the student is scored live.
  This is the single most persuasive five minutes of the presentation, and it is only
  possible because the teacher is an LLM
* Regenerate `ARCHITECTURE.md` figures; final `RESULTS.md`; presentation deck
* **Stretch (only if Weeks 1–3 hit their gates):** ESP32-S3 feature-extractor C port +
  nRF24 channel scan on the bench. Do **not** start this before Week 4 — an unfinished
  hardware port is worth less than a finished simulation story

---

## 5. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| LLM latency makes the corpus unaffordable | high | caching + teacher-on-disagreement + 200-episode golden set (§2.5) |
| LLM is *less* accurate than the privileged oracle | **near-certain, and fine** | the oracle cheats; the honest comparison is LLM-vs-student. Report both and say so |
| LLM non-determinism breaks reproducibility | medium | cache is keyed on prompt SHA-256, so a re-run replays byte-identical decisions; the cache is a committed artefact |
| Policy induction proposes overfit rules | medium | rules only accepted on held-out improvement; `sweep` stays held out |
| Week 4 hardware overruns | high | it is explicitly a stretch behind three gates |
| ns-3 patch not applied on the demo machine | **medium, and fatal** | pre-flight check script that refuses to run unpatched |

---

## 6. What we will say in the presentation

1. **The problem is a decision under asymmetric cost**, not a classification problem.
   Fading→jamming costs 10; the reverse costs 1. Everything follows from that.
2. **We built our own harness.** Here are the ~600 lines. No framework. Every decision
   replayable from disk.
3. **The teacher is an LLM that reasons**, and here it is diagnosing a reactive jammer by
   citing the TX-shadow statistic — a feature our numeric student still gets wrong.
4. **The student is 49.8 KB of numpy** and runs the same loop.
5. **Safety is structural.** FP(fading→jamming) = 0.000 and refusal = 100 % because the
   envelope makes the mistake unreachable, not because the model learned not to.
6. **Here is what does not work:** `sweep` held out is 0 %, `reactive` is weak in the closed
   loop, int8 failed its gate, and RTL-SDR cannot see 2.4 GHz. We found a real crash in
   ns-3.45's OLSR and it invalidated one of our own earlier numbers, which we retracted.
7. **Instructor, name a scenario.** [live demo]

Point 6 is not a weakness in the presentation. It is the strongest evidence that points 1–5
are measured rather than asserted.
