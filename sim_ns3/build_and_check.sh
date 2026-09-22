#!/usr/bin/env bash
# Build the patched jamming-sim and prove the AUDIT F3 fixes actually took effect.
#
#   cd <repo> && bash sim_ns3/build_and_check.sh
#
# Runs on YOUR Mac (ns-3 is a macOS build). Takes ~2-4 min incremental.
# Then:  python3 capstone/tests/test_world_ns3.py   <- asserts each fix, from the CSVs.
set -uo pipefail

NS3_DIR="${NS3_DIR:-$HOME/Documents/NS3/ns-3-dev}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/data/ns3_check"
FAMILIES="${FAMILIES:-barrage_all_channels spot_single_channel sweeping_jammer reactive_on_tx fading_no_attacker dead_peer congestion_burst hidden_terminal refusal_broadband_no_los}"

say() { printf "\n\033[1m== %s\033[0m\n" "$*"; }
die() { printf "\033[31mFAIL: %s\033[0m\n" "$*" >&2; exit 1; }

[ -d "$NS3_DIR" ] || die "ns-3 not at $NS3_DIR (set NS3_DIR=...)"

say "0. OLSR robustness patch"
# The patch stops ns-3.45's OLSR dying with SIGILL under strong jamming (NS3_BUG.md).
# 23 of 30 scenarios crashed without it, and a crashed run still leaves a partial CSV.
if grep -qi "AUDIT\|robustness\|messageSize > sizeLeft" "$NS3_DIR/src/olsr/model/olsr-header.cc" 2>/dev/null; then
  echo "   looks applied"
else
  echo "   NOT detected. Applying sim_ns3/ns3-olsr-robustness.patch ..."
  # -p0, not -p1: the patch's +++ lines are already relative to $NS3_DIR
  # (e.g. "src/olsr/model/olsr-header.cc", no a/ or b/ prefix) -- -p1 strips
  # "src/" and looks for a path that does not exist. Found the hard way: the
  # detection grep above was also case-sensitive and missed the applied
  # patch's own "ROBUSTNESS PATCH" marker (capitalised), so this branch
  # looked untested for a while when the fix was in fact already live.
  ( cd "$NS3_DIR" && patch -p0 --forward < "$REPO/sim_ns3/ns3-olsr-robustness.patch" ) \
    || echo "   (patch reported an issue -- check manually; a partial CSV proves nothing)"
fi

say "1. install sources into scratch/jamming/"
# MUST be a subdirectory, not the scratch root. ns-3's scratch/CMakeLists.txt builds ONE
# executable per directory, from every .cc in it, and fails hard if a directory contains a
# source with no main(). Dropping mesh-node-app.cc into scratch/ directly makes CMake try
# to build it as its own program -> "do not contain a main function" and the build dies.
mkdir -p "$NS3_DIR/scratch/jamming"
cp "$REPO/sim_ns3/jamming-sim.cc" "$REPO/sim_ns3/mesh-node-app.cc" "$REPO/sim_ns3/mesh-node-app.h" \
   "$NS3_DIR/scratch/jamming/" || die "copy failed"
# a stale flat copy from an older README would break the build for the same reason
rm -f "$NS3_DIR/scratch/jamming-sim.cc" "$NS3_DIR/scratch/mesh-node-app.cc" \
      "$NS3_DIR/scratch/mesh-node-app.h" 2>/dev/null

say "2. build"
( cd "$NS3_DIR" && ./ns3 build jamming-sim ) || die "BUILD FAILED -- paste the error back"
BIN="$(find "$NS3_DIR/build" -name '*jamming-sim*' -type f -perm -u+x | head -1)"
[ -n "$BIN" ] || die "built, but no binary found under $NS3_DIR/build"
echo "   binary: $BIN"
case "$BIN" in
  *-debug) echo "   NOTE: this is a DEBUG build (~10x slower). Fine for a demo; do NOT" ;
           echo "         regenerate the 540-scenario corpus on it." ;;
esac

say "3. run every family"
mkdir -p "$OUT"
ok=0; bad=0
for f in $FAMILIES; do
  Y="$REPO/scenarios/$f.yaml"
  [ -f "$Y" ] || { echo "   skip $f (no yaml)"; continue; }
  ( cd "$REPO/capstone" && python3 -m sim.export_ns3 "../scenarios/$f.yaml" -o "$OUT/$f.cfg" ) \
    || { echo "   export failed: $f"; bad=$((bad+1)); continue; }
  # Invoke the binary directly rather than via `./ns3 run`: for a scratch SUBDIRECTORY the
  # run-target spelling varies by ns-3 version, and `./ns3 run` also swallows the exit code
  # behind its own. We need the simulator's real exit status -- that is the whole point.
  ( cd "$NS3_DIR" && "$BIN" --config="$OUT/$f.cfg" --out="$OUT/$f" --tdmaSlots=4 ) \
      >"$OUT/$f.log" 2>&1
  rc=$?
  # A file that exists proves nothing -- check the exit code AND that the log reached
  # >=80% of the configured duration. That is the process fix from NS3_BUG.md.
  rows=$(wc -l < "$OUT/$f.percept.csv" 2>/dev/null || echo 0)
  if [ $rc -eq 0 ] && [ "$rows" -gt 400 ]; then
    echo "   ok   $f   (exit 0, $rows percept rows)"; ok=$((ok+1))
  else
    echo "   FAIL $f   (exit $rc, $rows percept rows)  -> $OUT/$f.log"; bad=$((bad+1))
  fi
done

say "done: $ok ok, $bad failed"
echo "CSVs in $OUT"
echo
echo "Now run the gate that checks the fixes actually took effect:"
echo "   python3 capstone/tests/test_world_ns3.py"
[ $bad -eq 0 ] || exit 1
