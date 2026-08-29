# sgm-bench top-level Makefile
#
#   make                 -> bin/sgm_ref bin/gen_synthetic bin/baseline_*
#   make golden          -> data/synthetic pair + data/golden/synthetic.pgm (+ sha256)
#   make check           -> runs every built sgm_* binary against golden
#   make MCPU=cortex-a55 -> tuning flags for a target core (default: native)
#
# Add an implementation by listing its .c files under IMPL_<name>_SRCS and
# appending <name> to IMPLS. The harness links against exactly one SGM_IMPL.

CC      ?= gcc
MCPU    ?= native
BOARD   ?= unknown
OPT     ?= -O3
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)

# Reference gets -O2 and NO -mcpu: it must be portable and boring.
REF_CFLAGS  := -O2 -Wall -Wextra -std=gnu11 -Icommon
IMPL_CFLAGS := $(OPT) -mcpu=$(MCPU) -fopenmp -Wall -Wextra -std=gnu11 -Icommon
DEFS        := -DGIT_SHA='"$(GIT_SHA)"'

COMMON  := common/util.c
HARNESS := common/harness.c

IMPLS := a55
IMPL_a55_SRCS := a55/sgm_a55.c common/census_neon.c
# --- implementations (uncomment / add as they appear) ---
# IMPLS += a55
# IMPL_a55_SRCS := a55/sgm_a55.c common/census_neon.c common/hamming_neon.c
# IMPLS += a720
# IMPL_a720_SRCS := a720/sgm_a720.c common/census_neon.c common/hamming_neon.c

BINS := bin/sgm_ref bin/gen_synthetic bin/baseline_aggregate bin/baseline_census $(addprefix bin/sgm_,$(IMPLS))

all: $(BINS)

bin:
	mkdir -p bin

bin/sgm_ref: common/sgm_ref.c $(HARNESS) $(COMMON) common/sgm.h common/sgm_params.h | bin
	$(CC) $(REF_CFLAGS) $(DEFS) -DCFLAGS_STR='"$(REF_CFLAGS)"' -o $@ common/sgm_ref.c $(HARNESS) $(COMMON)

bin/gen_synthetic: common/gen_synthetic.c common/sgm_params.h | bin
	$(CC) -O2 -Wall -Icommon -o $@ common/gen_synthetic.c

# Phase 0 baselines: the original standalone benchmarks, built unchanged.
bin/baseline_aggregate: a55/baseline/sgm_aggregate_neon.c | bin
	$(CC) $(OPT) -mcpu=$(MCPU) -fopenmp -o $@ $< -lm
bin/baseline_census: a55/baseline/sgm_census_cost.c | bin
	$(CC) $(OPT) -mcpu=$(MCPU) -fopenmp -o $@ $< -lm

define IMPL_RULE
bin/sgm_$(1): $$(IMPL_$(1)_SRCS) $$(HARNESS) $$(COMMON) common/sgm.h common/sgm_params.h | bin
	$$(CC) $$(IMPL_CFLAGS) $$(DEFS) -DCFLAGS_STR='"$$(IMPL_CFLAGS)"' -o $$@ $$(IMPL_$(1)_SRCS) $$(HARNESS) $$(COMMON)
endef
$(foreach i,$(IMPLS),$(eval $(call IMPL_RULE,$(i))))

# ---- golden ----
SYN_W ?= 1920
SYN_H ?= 1080
data/synthetic/left.pgm: bin/gen_synthetic
	mkdir -p data/synthetic
	./bin/gen_synthetic $(SYN_W) $(SYN_H) data/synthetic 1

golden: bin/sgm_ref data/synthetic/left.pgm
	mkdir -p data/golden
	./bin/sgm_ref data/synthetic/left.pgm data/synthetic/right.pgm -w 0 -n 1 -o data/golden/synthetic.pgm
	sha256sum data/golden/synthetic.pgm > data/golden/synthetic.sha256
	@echo "params: D=$$(grep -E '^#define SGM_D ' common/sgm_params.h | awk '{print $$3}') paths=$$(grep -E '^#define SGM_PATHS' common/sgm_params.h | awk '{print $$3}')" > data/golden/synthetic.params
	@cat data/golden/synthetic.sha256 data/golden/synthetic.params
	@if [ -f data/real/left.pgm ]; then \
	  ./bin/sgm_ref data/real/left.pgm data/real/right.pgm -w 0 -n 1 -o data/golden/real.pgm && \
	  sha256sum data/golden/real.pgm > data/golden/real.sha256; fi

# ---- calibrate the cost gate on a build you trust ----
# The roofline gate is armed by a calibration recorded per implementation and
# per board. Run this ONLY on a build you have reason to believe is healthy:
# it records what "good" costs, and everything afterwards is checked against it.
roofline-cal: all
	@mkdir -p data/golden
	@for b in bin/sgm_*; do \
	  [ "$$b" = bin/sgm_ref ] && continue; \
	  $$b data/synthetic/left.pgm data/synthetic/right.pgm -w 1 -n 5 \
	     --board $(BOARD) --roofline-calibrate || exit $$?; \
	done

# ---- check every implementation against golden ----
check: all
	@for b in bin/sgm_*; do \
	  [ "$$b" = bin/sgm_ref ] && continue; \
	  $$b data/synthetic/left.pgm data/synthetic/right.pgm -g data/golden/synthetic.pgm -w 1 -n 3 --board $(BOARD) || exit $$?; \
	  if [ -f data/real/left.pgm ]; then $$b data/real/left.pgm data/real/right.pgm -g data/golden/real.pgm -w 1 -n 3 || exit $$?; fi; \
	done

clean:
	rm -rf bin

.PHONY: all golden check clean roofline-cal
