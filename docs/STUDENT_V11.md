# Student v11 — rebuilt from the full 540-episode teacher corpus

## First, the direct answer

**The teacher has already been run over all 540 episodes.** That corpus exists and is
intact: `data/traces/llm_full` — 540 episode files, 9,503 ticks, **4,977 genuine uncached
clean LLM decisions**. Re-running it was never the missing step.

The missing step was **training the student from it**, which costs zero model calls. That
is what this is. (A second 540 run with the new 35-row panel is progressing separately,
but the A/B says it will be equivalent — see `TEACHER_AB_RESULT.md`.)

---

## Result on the live ns-3 bridge

27 test episodes, scans must be bought, same verifier as every other number in the report.

| metric | baseline | **student v11** |
|---|---|---|
| classification accuracy | 29.6% | **77.8%** |
| accuracy, seen families only | 33.3% | **87.5%** |
| expected cost (lower better) | 2.000 | **0.296** |
| survival rate | 70.4% | **92.6%** |
| detection latency, median | 1.70 s | **0.90 s** |
| censoring rate | 51.9% | **22.2%** |
| channel hops (mean) | 0.37 | **0.11** |
| FP fading→jamming (acted) | 0.000 | **0.000** |
| refusal gate | PASS | **PASS** |

Per family:

| family | baseline | student v11 |
|---|---|---|
| barrage | 67% | **100%** |
| congestion | 0% | **100%** |
| fading | 0% | **100%** |
| node_loss | 67% | **100%** |
| refusal | 100% | **100%** |
| hidden_term | 33% | **67%** |
| reactive | 0% | **67%** |
| spot | 0% | **67%** |
| **sweep (held out)** | 0% | **0%** |

**v10 scored 77% on this bridge; v11 is 87.5% on seen families.**

### Two caveats, stated plainly

1. **n = 3 episodes per family.** Twenty-seven episodes total. Treat single-family numbers
   as directional, not precise.
2. **`sweep` is deliberately held out** and the net scores 0% on it. That is the
   generalisation test and the net fails it — exactly as documented. Catching an unseen
   family is the induced rules' job (`agent/hybrid.py`), not the network's.

### The student beats the teacher (40.7%). That is not a paradox.

The teacher only ever sees the evidence panel and never the label. The student is distilled
from the teacher's decisions **and** trained against 13,700 labelled ns-3 windows. It learns
from ground truth the teacher never had. The teacher's job is to produce reasoning traces
and the rule layer; it is not the accuracy ceiling.

---

## Two dead design claims are now actually live

Both were flagged earlier in this project as claims the code did not honour:

**1. `abstain_threshold` was 0.0 — the student never abstained.**
Calibrating on ns-3 *corpus windows* produced 0.000 again, with the tool's own warning:
*"the calibration set is easier than deployment."* It is. In a corpus window spectrum data
is free; on the live bridge it has to be bought. Recalibrated on **live bridge rows**
(`data/traces_bridge/train.jsonl`):

```
abstain_threshold = 0.51        (was 0.000)
```

That is the same root cause as the failure-analysis mismatch: separability measured on
free-scan corpus windows does not transfer to an agent that must pay for evidence.

**2. `student_weights.h` was a 162-byte stub — the int8 gate had never run.**
It has now, and it passed:

```
cause head   all int8                   agreement 0.9988  L1 0.00522  PASS
call head    int8 except output layer   agreement 0.9968  L1 0.01056  PASS
weights      16,684 bytes (16.3 KB)     ESP32 budget 512 KB -> 31x headroom
student_weights.h  55.5 KB real C header for ESP-IDF
```

---

## The rule layer: `policy_v4.json`

Nine rules proposed by the LLM, measured on held-out data, **2 accepted, 7 dropped**:

| accepted | support | precision |
|---|---|---|
| `barrage_wideband_evidence` | 22 | 1.00 |
| `spot_narrowband_evidence` | 19 | 0.95 |

Dropped: sweep, fading, node_loss, congestion (×2), reactive, hidden_term — all below their
precision bar. The induction is working as designed (the LLM proposes, the corpus disposes),
and the two that survive are exactly the two families the whole system handles best. Do not
present this as a nine-rule policy.

---

## Files

```
data/student_v11/student_bundle.json       the student (abstain 0.51)
data/student_v11/student_weights.h         55.5 KB C header for ESP-IDF
data/student_v11/student_int8.json         int8 weights
data/student_v11/int8_equivalence.json     the gate result
data/student_v11/abstain_calibration.json  calibration record
data/policy_v4.json                        the 2 surviving rules
data/headline_v11.json                     the table above
```

Point the demo at it with
`STUDENT_BUNDLE=../data/student_v11/student_bundle.json python3 demo/live_demo.py ...`
