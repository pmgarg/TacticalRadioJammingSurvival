# ns-3.45 OLSR crashes under strong interference (found and patched)

## Symptom
`ns3.45-jamming-sim` died with **SIGILL (exit 132)** part-way through an episode,
silently, with no error output. **23 of 30** test scenarios crashed.

## Why it went unnoticed
Our corpus generator checked only that `<out>.percept.csv` **existed**. A crashed run
still leaves a partial CSV, so it reported `episodes ok=375 failed=0` while feeding the
model truncated episodes. **The generator now checks the process exit code and that the
log reaches ≥80% of the configured duration.** This is the single most important process
fix of the session: a validation that only proves a file exists proves nothing.

## Root cause
Reproducible and interference-dependent: no jammer → clean exit; jammer at −30 dBm →
clean exit; jammer at full power → SIGILL.

`src/olsr/model/olsr-header.cc`, `MessageHeader::Deserialize`:

```cpp
m_messageType = (MessageType)i.ReadU8();
NS_ASSERT(m_messageType >= HELLO_MESSAGE && m_messageType <= HNA_MESSAGE); // no-op when optimized
...
switch (m_messageType) {
  case MID_MESSAGE: ... case HNA_MESSAGE: ...
  default:
    NS_ASSERT(false);   // <-- compiles to NOTHING in an optimized build
}
```

With `NS_ASSERT` disabled the `default:` body is empty, so GCC may assume the switch
value is always a valid enumerator and emit `ud2` → SIGILL. A frame corrupted by strong
jamming carries an out-of-range message type and lands there.

A second, independent undefined-behaviour path is in
`olsr-routing-protocol.cc :: RecvOlsr`:

```cpp
sizeLeft -= messageHeader.GetSerializedSize();   // can underflow to ~4e9 -> near-infinite loop
```

## Fix (`sim_ns3/ns3-olsr-robustness.patch`)
1. `Deserialize` validates the message type and `m_messageSize`; on malformed input it
   consumes **exactly the header** and returns that size — it must return the bytes
   consumed, and an earlier attempt returning 0 corrupted the packet buffer and turned
   SIGILL into SIGABRT/SIGSEGV across the board.
2. The `default:` branch is given a defined return so the compiler cannot assume
   unreachability.
3. `RecvOlsr` stops parsing a packet whose message size is zero or exceeds the bytes
   remaining, and skips messages whose type did not survive the channel.

No routing logic is altered — malformed input is refused instead of invoking UB.

**Result: 0 crashes across the corpus (was 23/30).** Worth reporting upstream.

## What it invalidated
Every ns-3 number measured before the patch came from truncated episodes, including the
previously reported "82.7% on ns-3 features". Corrected figure on the clean corpus:
**80.8%**. It also explains the erratic live results (61% / 39% / 28% / 22% for the same
model): episodes were silently ending at different points.
