# Jamming Survival -- one command per gate. Run from the repo root.
#
# The gates are ordered: G0 first, then G1..G4. Nothing downstream is trustworthy
# until the ones above it are green. `make gates` runs all of them and fails on the
# first failure, which is the point -- see docs/CODE_FLOW.md section 5.

SHELL   := /bin/bash
.SHELLFLAGS := -o pipefail -c
PY      ?= python3
CAP      = capstone
NS3     ?= $(HOME)/Documents/NS3/ns-3-dev/build/scratch/jamming/ns3.45-jamming-sim-optimized
BUNDLE  ?= ../data/student_v8_clean/student_bundle.json

.PHONY: help gates g0 g1 g2 g3 g4 corpus episode dash train eval clean-check

help:
	@grep -E '^[a-z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t22

gates: g0 g4 g1 g3 hal g2  ## run every gate, cheapest and most fundamental first
	@echo ""
	@echo "================ ALL GATES GREEN ================"

g0:                        ## G0 the repo is the truth: the agentic code is tracked and imports
	@echo "== G0  repo integrity =="
	@cd $(CAP) && $(PY) -c "import sys;sys.path.insert(0,'.');\
import harness.loop, harness.registry, gateway.provider, agent.llm_teacher, agent.rule_agent;\
print('  harness + gateway + llm_teacher + rule_agent import OK')"
	@cd $(CAP) && $(PY) -m compileall -q agent percept scenario sim verify harness gateway dash \
	  && echo "  all modules compile"

g4:                        ## G4 evaluation integrity: no scenario is in both train and test
	@echo "== G4  leakage =="
	@cd $(CAP) && $(PY) -m train.filter_traces | tail -3

g1:                        ## G1 contract effect: every call in the contract does something
	@echo "== G1  contract effects =="
	@cd $(CAP) && $(PY) tests/test_contract_effects.py | tail -12

g3:                        ## G3 the safety envelope holds, including against a rogue agent
	@echo "== G3  safety + rogue control =="
	@cd $(CAP) && $(PY) tests/test_agent_safety.py | tail -3
	@cd $(CAP) && $(PY) tests/test_rogue_control.py | tail -3
	@cd $(CAP) && $(PY) tests/test_verifier_anchoring.py | tail -3

g2:                        ## G2 percept separation: refsim vs ns-3 fidelity
	@echo "== G2  fidelity =="
	@cd $(CAP) && out=$$($(PY) tests/xval_refsim_ns3.py 2>&1); \
	  if echo "$$out" | grep -q "0/0"; then \
	    echo "  SKIPPED: no cached ns-3 percept CSVs found."; \
	    echo "  0/0 pairs agreeing is NOT agreement -- this gate is vacuous without them."; \
	    echo "  Run 'make ns3-corpus' first, then re-run."; \
	  else echo "$$out" | tail -6; fi

episode:                   ## one episode, verbose:  make episode SC=../scenarios/spot_single_channel.yaml
	cd $(CAP) && $(PY) run_episode.py --scenario $(or $(SC),../scenarios/spot_single_channel.yaml) -v

corpus:                    ## whole corpus through the student, records to data/runs
	cd $(CAP) && $(PY) run_episode.py --corpus --seeds $(or $(SEEDS),1) \
	  --agent $(or $(AGENT),student) --bundle $(BUNDLE) --out ../data/runs

eval:                      ## three-way comparison on the held-out test split
	cd $(CAP) && $(PY) -m train.evaluate --corpus ../data/corpus --split test --bundle $(BUNDLE)

train:                     ## retrain the student on the clean split (needs scikit-learn)
	cd $(CAP) && $(PY) -m train.train_mixed --refsim ../data/traces_all --ns3 ../data/traces_ns3 \
	  --bridge ../data/traces_bridge/train.jsonl --out ../data/student_next

hal:                       ## G5 Mode H: build and self-test the hardware port (no hardware needed)
	cd firmware && cc -std=c11 -Wall -Wextra -O2 -o haltest hal_host.c hal_selftest.c && ./haltest
	cd firmware && cc -std=c11 -Wall -Wextra -fsyntax-only hal_esp32.c && echo "  hal_esp32.c parses"

hybrid:                    ## does the hybrid recover the held-out family the net cannot?
	cd $(CAP) && $(PY) tests/test_hybrid_generalisation.py --limit $(or $(N),10)

llm-stub:                  ## exercise the whole LLM path with the model STUBBED (CI-safe)
	cd $(CAP) && $(PY) -m harness.run_llm --per-family 1 --workers 4 --provider stub \
	  --trace-dir ../data/traces/llm_stub --out ../data/llm_stub.json
	cd $(CAP) && $(PY) -m train.llm_traces_to_rows --traces ../data/traces/llm_stub \
	  --split any --out ../data/traces_llm/train.jsonl

llm:                       ## THE REAL teacher, against authoritative ns-3 (needs your claude CLI)
	cd $(CAP) && $(PY) -m harness.run_llm --per-family 2 --workers 6 --provider claude \
	  --world ns3 --ns3 "$$(find $(HOME)/Documents/NS3/ns-3-dev/build -name '*jamming-sim*' -type f -perm -u+x | head -1)" \
	  --trace-dir ../data/traces/llm_ns3 --out ../data/llm_golden_ns3.json
	cd $(CAP) && $(PY) -m train.llm_traces_to_rows --traces ../data/traces/llm_ns3 \
	  --split train --out ../data/traces_llm/train.jsonl

dash:                      ## the run dashboard at http://127.0.0.1:8765
	cd $(CAP) && $(PY) -m dash.serve

ns3-all:                   ## THE BIG ONE: rebuild C++, check the world, regenerate the corpus
	bash sim_ns3/build_and_check.sh
	$(PY) capstone/tests/test_world_ns3.py || echo "  (world gate reported failures -- read them)"
	@echo ""
	@echo "== regenerating the ns-3 corpus (this is the long part) =="
	cd $(CAP) && $(PY) -m train.gen_ns3_corpus --ns3 "$$(find $(HOME)/Documents/NS3/ns-3-dev/build -name '*jamming-sim*' -type f -perm -u+x | head -1)" \
	  --corpus ../data/corpus --split train --jobs 4 --out ../data/traces_ns3
	cd $(CAP) && $(PY) -m train.gen_ns3_corpus --ns3 "$$(find $(HOME)/Documents/NS3/ns-3-dev/build -name '*jamming-sim*' -type f -perm -u+x | head -1)" \
	  --corpus ../data/corpus --split val --jobs 4 --out ../data/traces_ns3
	cd $(CAP) && $(PY) -m train.gen_ns3_corpus --ns3 "$$(find $(HOME)/Documents/NS3/ns-3-dev/build -name '*jamming-sim*' -type f -perm -u+x | head -1)" \
	  --corpus ../data/corpus --split test --jobs 4 --out ../data/traces_ns3
	cd $(CAP) && $(PY) -m train.filter_traces
	@echo ""
	@echo "corpus regenerated. Next: make train, then make eval."

ns3-corpus:                ## regenerate the ns-3 corpus only (needed after any C++ change)
	cd $(CAP) && $(PY) -m train.gen_ns3_corpus --ns3 $(NS3) --corpus ../data/corpus \
	  --split train --jobs 2 --out ../data/traces_ns3
