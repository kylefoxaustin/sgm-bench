#!/usr/bin/env bash
# pin.sh — select cores by micro-architecture, not by cpu number.
#
#   source scripts/pin.sh            # defines functions
#   pin_list a720                    # -> "4,5,6,7" (cpus whose MIDR part == A720)
#   pin_run a55 6 ./bin/sgm_a55 ...  # run under taskset on the first 6 A55s,
#                                    #   OMP_NUM_THREADS=6, governor=performance,
#                                    #   logs per-core freq before and after
#
# MIDR_EL1 part numbers (bits [15:4]):
#   0xd05 Cortex-A55   0xd80 Cortex-A520   0xd81 Cortex-A720
#   0xd03 Cortex-A53   0xd0a Cortex-A75    0xd41 Cortex-A78
#   0xd4d Cortex-A715  0xd4e Cortex-X3     0xd82 Cortex-X4

declare -A SGM_PART=( [a55]=0xd05 [a520]=0xd80 [a720]=0xd81 [a53]=0xd03 [a75]=0xd0a [a78]=0xd41 [a78c]=0xd4b [a715]=0xd4d [x3]=0xd4e [x4]=0xd82 )

_midr_part() {  # cpu number -> part id as 0x???
  local f=/sys/devices/system/cpu/cpu$1/regs/identification/midr_el1
  [ -r "$f" ] || { echo "?"; return; }
  local m; m=$(cat "$f")
  printf "0x%03x" $(( (m >> 4) & 0xfff ))
}

pin_list() {  # uarch name -> comma list of cpus
  local want=${SGM_PART[$1]:-$1} out=()
  for c in /sys/devices/system/cpu/cpu[0-9]*; do
    local n=${c##*cpu}
    [ "$(_midr_part "$n")" = "$want" ] && out+=("$n")
  done
  (IFS=,; echo "${out[*]}")
}

pin_first_n() {  # uarch, n -> comma list of first n cpus of that uarch
  pin_list "$1" | tr ',' '\n' | head -n "$2" | paste -sd, -
}

log_freq() {  # comma list -> prints "cpuN: kHz" lines
  for n in ${1//,/ }; do
    local f=/sys/devices/system/cpu/cpu$n/cpufreq/scaling_cur_freq
    [ -r "$f" ] && echo "  cpu$n: $(cat "$f") kHz" || echo "  cpu$n: freq n/a"
  done
}

set_perf_governor() {
  for n in ${1//,/ }; do
    local g=/sys/devices/system/cpu/cpu$n/cpufreq/scaling_governor
    [ -w "$g" ] && echo performance > "$g" 2>/dev/null
  done
}

pin_run() {  # uarch, nthreads, cmd...
  local uarch=$1 nth=$2; shift 2
  local cpus; cpus=$(pin_first_n "$uarch" "$nth")
  [ -z "$cpus" ] && { echo "no cpus of type $uarch found" >&2; return 1; }
  set_perf_governor "$cpus"
  echo "== pin: $uarch x$nth -> cpus $cpus"; log_freq "$cpus"
  OMP_NUM_THREADS=$nth OMP_PLACES=cores OMP_PROC_BIND=close taskset -c "$cpus" "$@"
  local rc=$?
  echo "== after:"; log_freq "$cpus"
  return $rc
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "cpu topology by MIDR:"
  for c in /sys/devices/system/cpu/cpu[0-9]*; do n=${c##*cpu}; echo "  cpu$n part $(_midr_part "$n")"; done
  for u in a55 a520 a720 a53 a78 a78c a715 x4; do l=$(pin_list $u); [ -n "$l" ] && echo "  $u: $l"; done
fi
