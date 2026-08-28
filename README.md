# sgm-bench

Software SGM stereo across Cortex-A55 / A720 / Mali / CUDA, bit-exact against
a scalar reference.
Read `CLAUDE.md` — it is the plan, the rules, and the phase list.

Quick start on a board:

    make MCPU=cortex-a55 golden       # oracle + synthetic pair + golden map
    ./scripts/pin.sh                  # show cpu topology by micro-architecture
    make check                        # every sgm_* binary vs golden (exit 2 on mismatch)
