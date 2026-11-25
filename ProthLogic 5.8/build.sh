#!/bin/bash
# Build ProthLogic for the hardware of THIS machine (detection of the CPU and the NVIDIA cards).
#
# Usage:
#   ./build.sh            normal build (CPU AVX-512 engine + GPU)
#   ./build.sh gpu        GPU-only build for a CPU without AVX-512: the card sieves and tests (always --gpu-only),
#                         no CPU engine (fft_engine.cpp / fft_pack_vbmi.cpp left out, fft_stub.cpp instead), no --tune
#   ./build.sh profile    with [PROFILE] timings (-DLOTR_PROFILE)
#   ./build.sh warn       with warnings (-Wall -Wextra), saved to build_warnings.txt next to this script
#   (options can be combined: ./build.sh profile warn, ./build.sh gpu warn)
#
# CPU:  AVX-512 F/DQ/VBMI/IFMA + AVX2/FMA are required (the FFT/NTT kernels); -march=native tunes for this CPU.
#       ./build.sh gpu: any x86-64 CPU (it only feeds the card); needs an NVIDIA card and nvcc.
# GPU:  nvcc + nvidia-smi -> the GPU part (the sieve) is ALWAYS compiled, for the compute capability of every card
#       found (RTX 5060 Ti = sm_120); a card newer than nvcc knows gets PTX of the newest architecture nvcc has
#       (compiled by the driver). Whether the card also tests candidates is chosen at run time, not here.
#       No card or no nvcc -> CPU-only binary (-DLOTR_NO_CUDA, gpu_stub.cpp): the GPU sieve reports it is missing.
#
# Source folder: the script's own folder when the .cpp files are there, otherwise the folder one up.
# The binary is always next to the sources.
# NOTE: g++ *.cpp compiles ALL .cpp files in the source folder. Keep experimental .cpp files elsewhere.

set -e
set -o pipefail

PROFILE_FLAG=""
WARN_FLAGS=""
GPU_ONLY=0
for arg in "$@"; do
    case "$arg" in
        profile) PROFILE_FLAG="-DLOTR_PROFILE" ;;
        warn)    WARN_FLAGS="-Wall -Wextra" ;;
        gpu)     GPU_ONLY=1 ;;
        *) echo "Unknown option: $arg (allowed: gpu, profile, warn)"; exit 2 ;;
    esac
done

TOOLS="$(cd "$(dirname "$0")" && pwd)"
if ls "$TOOLS"/*.cpp >/dev/null 2>&1; then cd "$TOOLS"; else cd "$TOOLS/.."; fi

# ---- CPU
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
CPU_FLAGS=" $(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2) "
MISSING=""
for f in avx512f avx512dq avx512vbmi avx512ifma avx2 fma; do
    case "$CPU_FLAGS" in *" $f "*) ;; *) MISSING="$MISSING $f" ;; esac
done
if [ "$GPU_ONLY" = 1 ]; then
    if [ -n "$MISSING" ]; then
        echo "[build] CPU: $CPU_MODEL ($(nproc) threads, missing:$MISSING) - not needed: GPU-only build"
    else
        echo "[build] CPU: $CPU_MODEL ($(nproc) threads, AVX-512 present but not used: GPU-only build)"
    fi
elif [ -n "$MISSING" ]; then
    echo "[build] CPU: $CPU_MODEL - missing:$MISSING"
    echo "[build] ProthLogic needs AVX-512 (F, DQ, VBMI, IFMA): no CPU engine for this CPU"
    echo "[build] sieve and tests on the NVIDIA card only: ./build.sh gpu"
    exit 1
else
    echo "[build] CPU: $CPU_MODEL ($(nproc) threads, AVX-512 F/DQ/VBMI/IFMA)"
fi

# ---- GPU
GPU=0
GENCODE=""
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[build] GPU: no nvcc (CUDA toolkit) - CPU-only binary"
elif ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[build] GPU: no NVIDIA card (nvidia-smi) - CPU-only binary"
else
    KNOWN=$(nvcc --list-gpu-arch 2>/dev/null | sed -n 's/^compute_\([0-9]*\)$/\1/p' | sort -n)
    NEWEST=$(echo "$KNOWN" | tail -1)
    while IFS=, read -r name cap; do
        name=$(echo "$name" | sed 's/^ *//;s/ *$//')
        c=$(echo "$cap" | tr -d ' .')
        [ -z "$c" ] && continue
        if [ -z "$KNOWN" ] || echo "$KNOWN" | grep -qx "$c"; then   # (an old nvcc without --list-gpu-arch: as reported)
            g="-gencode arch=compute_$c,code=sm_$c"
            echo "[build] GPU: $name (sm_$c)"
        else
            g="-gencode arch=compute_$NEWEST,code=compute_$NEWEST"
            echo "[build] GPU: $name (sm_$c) - unknown to this nvcc: PTX compute_$NEWEST (JIT by the driver; update CUDA)"
        fi
        case " $GENCODE " in *" $g "*) ;; *) GENCODE="$GENCODE $g" ;; esac
    done < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
    [ -n "$GENCODE" ] && GPU=1
fi
if [ "$GPU_ONLY" = 1 ] && [ "$GPU" != 1 ]; then
    echo "[build] ./build.sh gpu needs an NVIDIA card and nvcc: no build"
    exit 1
fi

if [ "$GPU_ONLY" = 1 ]; then
    # no -mavx512* / -mavx2 / -mfma: -march=native alone gives only what this CPU has
    CXXFLAGS="-O3 -march=native -g -fopenmp -funroll-loops -fomit-frame-pointer -ffast-math -std=c++17 -pthread -DLOTR_GPU_ONLY"
else
    CXXFLAGS="-O3 -march=native -g -fopenmp -mavx512f -mavx512vbmi -mavx512dq -mavx2 -mfma -funroll-loops -fomit-frame-pointer -mprefer-vector-width=512 -ffast-math -std=c++17 -pthread"
fi
if [ "$GPU" = 1 ]; then
    CUDA_LIB="$(dirname "$(dirname "$(command -v nvcc)")")/lib64"
    # GPU parts (gpu_engine.cu = the sieve, gpu_proth_engine.cu = the Proth test of --gpu): each rebuilt when its .cu
    # (or for the engine its .h / gpu_hybrid.cuh) is newer than the .o or the card set (architectures) or the nvcc
    # version changed (the code of nvcc 13.3 is ~5% faster than 12.8 - no objects of an older nvcc)
    STAMP="$GENCODE $(nvcc --version | tail -1)"
    ARCH_CHANGED=0
    [ "$(cat .gpu_arch 2>/dev/null)" != "$STAMP" ] && ARCH_CHANGED=1
    if [ ! -f gpu_engine.o ] || [ gpu_engine.cu -nt gpu_engine.o ] || [ "$ARCH_CHANGED" = 1 ]; then
        echo "[build] nvcc gpu_engine.cu ($GENCODE )"
        nvcc -O3 $GENCODE --use_fast_math -c gpu_engine.cu -o gpu_engine.o
    else
        echo "[build] gpu_engine.o up to date (skipping nvcc)"
    fi
    # (no --use_fast_math: k_q needs exact ldexp / llround)
    if [ ! -f gpu_proth_engine.o ] || [ gpu_proth_engine.cu -nt gpu_proth_engine.o ] || [ gpu_proth_engine.h -nt gpu_proth_engine.o ] || [ gpu_hybrid.cuh -nt gpu_proth_engine.o ] || [ "$ARCH_CHANGED" = 1 ]; then
        echo "[build] nvcc gpu_proth_engine.cu ($GENCODE ) $(nvcc --version | tail -1)"
        nvcc -O3 $GENCODE -c gpu_proth_engine.cu -o gpu_proth_engine.o
    else
        echo "[build] gpu_proth_engine.o up to date (skipping nvcc)"
    fi
    echo "$STAMP" > .gpu_arch
    OBJS="gpu_engine.o gpu_proth_engine.o"
    LIBS="-lgmpxx -lgmp -L$CUDA_LIB -lcudart"
    KIND="CPU + GPU"
    [ "$GPU_ONLY" = 1 ] && KIND="GPU only, no AVX-512"
else
    OBJS=""
    LIBS="-lgmpxx -lgmp"
    CXXFLAGS="$CXXFLAGS -DLOTR_NO_CUDA"
    KIND="CPU only"
fi

# every .cpp to its own object in .obj/, in parallel (all threads), and only when stale: the .cpp or any .h is
# newer than its .o, or the flags changed
# ./build.sh gpu: its own .obj_gpu (switching between the two builds recompiles nothing that is up to date) and without
# the AVX-512 CPU engine
OBJDIR=.obj
SKIP=""
if [ "$GPU_ONLY" = 1 ]; then OBJDIR=.obj_gpu; SKIP=" fft_engine.cpp fft_pack_vbmi.cpp "; fi
mkdir -p "$OBJDIR"
FLAGS="$CXXFLAGS $PROFILE_FLAG $WARN_FLAGS"
if [ "$(cat "$OBJDIR/.flags" 2>/dev/null)" != "$FLAGS" ]; then rm -f "$OBJDIR"/*.o; fi
NEWEST_H="$(ls -t *.h 2>/dev/null | head -1)"
STALE=""
CPPOBJS=""
for f in *.cpp; do
    case "$SKIP" in *" $f "*) continue ;; esac
    o="$OBJDIR/${f%.cpp}.o"
    CPPOBJS="$CPPOBJS $o"
    if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || { [ -n "$NEWEST_H" ] && [ "$NEWEST_H" -nt "$o" ]; }; then STALE="$STALE $f"; fi
done
echo "$FLAGS" > "$OBJDIR/.flags"
if [ -n "$STALE" ]; then
    echo "[build] g++ -c$STALE ${PROFILE_FLAG} ${WARN_FLAGS} ($KIND, $(nproc) at once)"
    if [ -n "$WARN_FLAGS" ]; then
        printf '%s\n' $STALE | xargs -P "$(nproc)" -I{} sh -c 'g++ -c "$1" $2 -o "$3/$(basename "$1" .cpp).o"' _ {} "$FLAGS" "$OBJDIR" 2>&1 | tee "$TOOLS/build_warnings.txt"
        echo "[build] warnings saved to $TOOLS/build_warnings.txt ($(grep -c 'warning:' "$TOOLS/build_warnings.txt" || true) warnings)"
    else
        printf '%s\n' $STALE | xargs -P "$(nproc)" -I{} sh -c 'g++ -c "$1" $2 -o "$3/$(basename "$1" .cpp).o"' _ {} "$FLAGS" "$OBJDIR"
    fi
else
    echo "[build] all .cpp objects up to date (skipping g++ -c)"
fi
echo "[build] link ($KIND)"
g++ $CPPOBJS $OBJS $CXXFLAGS $PROFILE_FLAG $LIBS -o prothlogic

echo "[build] done: $(pwd)/prothlogic ($KIND)"
