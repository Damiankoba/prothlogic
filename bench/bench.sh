#!/bin/bash
# ProthLogic speed benchmark: tests a list of numbers and prints a table (also saved as a CSV file).
#
# Usage (from the bench/ folder, after ./build.sh):
#   ./bench.sh                      on the CPU, with 1 thread and with all cores: short list bench_quick.txt
#   ./bench.sh bench_list.txt       on the CPU, the full list (several hours)
#   GPU=1 ./bench.sh                on the NVIDIA card (--gpu-only): full list bench_list.txt (about 15-30 minutes)
#   GPU=1 ./bench.sh bench_quick.txt   on the card, short list
#   THREADS="1 8" ./bench.sh        threads per candidate to measure (default: 1 and the number of cores)
#   CPUS=0-7 ./bench.sh             run on these cores only (taskset -c)
#   BIG=3000000 ./bench.sh          from this n on, measure only the speed of the first 1% instead of the full test
#   BIN=../prothlogic ./bench.sh    path to the program (default: prothlogic next to the sources)
#
# The list holds "n k" on each line; lines starting with # are skipped.
# Below BIG the test runs to the end and the table shows the full time. From BIG on, the test is stopped after its
# first progress line and the time of 1% is computed from the speed (ms per squaring).
# For a fair result, run nothing else on the same cores or on the same card.

CORES="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
[ "${CORES:-0}" -ge 1 ] || CORES="$(nproc)"
GPU="${GPU:-0}"
if [ "$GPU" = "1" ]; then
    THREADS="1"
    BIG="${BIG:-8000000}"
    GPUFLAG="--gpu-only"
else
    THREADS="${THREADS:-1 $CORES}"
    BIG="${BIG:-1876758}"
    GPUFLAG=""
fi
MAXWAIT="${MAXWAIT:-5400}"     # time limit of one measurement [s]

# The program is next to the sources: the script's folder when the .cpp files are there, otherwise one folder up.
TOOLS="$(cd "$(dirname "$0")" && pwd)"
if ls "$TOOLS"/*.cpp >/dev/null 2>&1; then SRC="$TOOLS"; else SRC="$(dirname "$TOOLS")"; fi
if [ -n "$BIN" ]; then BIN_ABS="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
else BIN_ABS="$SRC/prothlogic"; fi
DEFAULT_LIST="$TOOLS/bench_quick.txt"
[ "$GPU" = "1" ] && DEFAULT_LIST="$TOOLS/bench_list.txt"
LIST="${1:-$DEFAULT_LIST}"
if [ ! -x "$BIN_ABS" ]; then echo "Program not found: $BIN_ABS (run ./build.sh first)"; exit 2; fi
if [ ! -f "$LIST" ]; then echo "List not found: $LIST"; exit 2; fi
LIST_ABS="$(cd "$(dirname "$LIST")" && pwd)/$(basename "$LIST")"
cd "$TOOLS"

TS=""
[ -n "$CPUS" ] && command -v taskset >/dev/null 2>&1 && TS="taskset -c $CPUS"

STAMP="$(date +%Y%m%d_%H%M)"
CSV="$(pwd)/bench_${STAMP}.csv"
[ "$GPU" = "1" ] && CSV="$(pwd)/bench_gpu_${STAMP}.csv"
rm -rf bench_tmp; mkdir -p bench_tmp; cd bench_tmp

if pgrep -x "$(basename "$BIN_ABS")" >/dev/null 2>&1; then
    echo "NOTE: another $(basename "$BIN_ABS") process is running - if it shares the cores or the card, the times will be too high."
    echo
fi

SB=""
command -v stdbuf >/dev/null 2>&1 && SB="stdbuf -oL"

echo "# bin=$BIN_ABS md5=$(md5sum "$BIN_ABS" | cut -d' ' -f1) date=$(date '+%F %T') BIG=$BIG GPU=$GPU CPUS=$CPUS" > "$CSV"
echo "n,k,threads,fft,engine,ms_iter,max_err,t_full_s,t_1pct_s,result" >> "$CSV"

printf "%-9s %-7s %-4s %-6s %-24s %9s %9s %10s %10s  %s\n" n k thr FFT engine ms/iter MaxErr t_full_s t_1pct_s result

run_one() {   # run_one <n> <k> <threads>
    local n="$1" k="$2" t="$3"
    local log="log_${n}_${k}_${t}.txt"
    rm -f check_k${k}_n${n}.bin "$log"

    $SB $TS "$BIN_ABS" --n "$n" --k "$k" --workers 1 --threads "$t" --report 1 $GPUFLAG > "$log" 2>&1 &
    local pid=$! waited=0 stopped=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$n" -ge "$BIG" ] && grep -q "\[STEP\]" "$log" 2>/dev/null; then
            kill "$pid" 2>/dev/null; stopped=1; break
        fi
        if [ "$waited" -ge "$MAXWAIT" ]; then kill "$pid" 2>/dev/null; stopped=2; break; fi
        sleep 1; waited=$((waited+1))
    done
    wait "$pid" 2>/dev/null
    rm -f check_k${k}_n${n}.bin check_k${k}_n${n}.bin.time

    local fft eng ms err tfull res t1
    if [ "$GPU" = "1" ]; then
        # "[GPU] k*2^n+1: NTT 2^18 x 3 primes, 32-bit digits, ..." or
        # "[GPU] k*2^n+1: HYBRID NTT 2^16 complex (2^17 digits) x 3 primes, 24-bit digits, ..."
        fft="$(grep -o 'NTT 2^[0-9]*' "$log" | head -1 | sed 's/NTT //')"
        eng="$(grep -m1 '\[GPU\].*NTT 2^' "$log" | sed -n 's/.*NTT 2^[0-9]*.* x \([0-9]*\) primes, \([0-9]*\)-bit digits.*/NTT \1p \2-bit/p')"
        if grep -q '\[GPU\].*HYBRID NTT' "$log"; then eng="GPU HYB $eng"; fft="${fft}c"; elif [ -n "$eng" ]; then eng="GPU $eng"; fi
    else
        fft="$(grep -o 'FFT_Len = 2^[0-9]*' "$log" | head -1 | sed 's/.*= //')"
        eng="$(grep 'HYBRID ENGINE' "$log" | head -1 | sed 's/.*Selected: //; s/ *\[FFT_Len.*//')"
    fi
    ms="$(grep -m1 '\[STEP\]' "$log" | sed -n 's/.*| \([0-9.]*\) ms\/iter.*/\1/p')"
    err="$(grep -m1 '\[STEP\]' "$log" | sed -n 's/.*Max Err: \([0-9.eE+-]*\) .*/\1/p')"
    if [ -z "$ms" ]; then ms="$(grep -m1 'Speed:' "$log" | grep -o '[0-9.]* ms' | cut -d' ' -f1)"; fi
    tfull="$(grep -o 'Time : [0-9.]* sec' "$log" | tail -1 | sed 's/Time : //; s/ sec//')"
    # on the card: the average over the whole test (the first progress line is only 1% and includes the warm-up)
    if [ "$GPU" = "1" ] && [ -n "$tfull" ]; then
        ms="$(awk -v t="$tfull" -v n="$n" 'BEGIN{printf "%.4f", t*1000.0/(n-1)}')"
    fi
    if grep -q 'FOUND PRIME' "$log"; then res="PRIME"
    elif grep -q 'Proth RES64' "$log"; then res="$(grep -o 'RES64: 0x[0-9A-F]*' "$log" | head -1 | sed 's/RES64: 0x//')"
    elif [ -z "$ms" ]; then res="NO-DATA (see bench_tmp/$log)"; else res="(stopped)"; fi
    [ "$stopped" = "2" ] && res="TIMEOUT"
    t1=""
    [ -n "$ms" ] && t1="$(awk -v ms="$ms" -v n="$n" 'BEGIN{printf "%.1f", ms*(n-1)*0.01/1000.0}')"
    [ "$n" -ge "$BIG" ] && tfull=""

    local tl="$t"
    [ "$GPU" = "1" ] && tl="gpu"
    printf "%-9s %-7s %-4s %-6s %-24s %9s %9s %10s %10s  %s\n" "$n" "$k" "$tl" "${fft:--}" "${eng:--}" "${ms:--}" "${err:--}" "${tfull:--}" "${t1:--}" "$res"
    echo "$n,$k,$tl,${fft},\"${eng}\",${ms},${err},${tfull},${t1},${res}" >> "$CSV"
}

while read -r n k rest || [ -n "$n" ]; do   # (|| ...: also the last line without a line end)
    case "$n" in ''|\#*) continue;; esac
    for t in $THREADS; do
        run_one "$n" "$k" "$t"
    done
done < "$LIST_ABS"

echo
echo "Results saved: $CSV"
