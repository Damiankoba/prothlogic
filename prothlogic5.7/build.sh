#!/bin/bash
# Build ProthLogic (the usual build commands in one file).
#
# Usage:
#   ./build.sh            normal build
#   ./build.sh profile    with [PROFILE] timings (-DLOTR_PROFILE)
#   ./build.sh warn       with warnings (-Wall -Wextra), saved to build_warnings.txt
#   (options can be combined: ./build.sh profile warn)
#
# NOTE: g++ *.cpp compiles ALL .cpp files in this folder. Keep experimental .cpp files elsewhere.

set -e
set -o pipefail

PROFILE_FLAG=""
WARN_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        profile) PROFILE_FLAG="-DLOTR_PROFILE" ;;
        warn)    WARN_FLAGS="-Wall -Wextra" ;;
        *) echo "Unknown option: $arg (allowed: profile, warn)"; exit 2 ;;
    esac
done

cd "$(dirname "$0")"

# GPU part: rebuilt only when gpu_engine.cu is newer than gpu_engine.o
if [ ! -f gpu_engine.o ] || [ gpu_engine.cu -nt gpu_engine.o ]; then
    echo "[build] nvcc gpu_engine.cu"
    nvcc -O3 -arch=sm_120 --use_fast_math -c gpu_engine.cu -o gpu_engine.o
else
    echo "[build] gpu_engine.o up to date (skipping nvcc)"
fi

echo "[build] g++ *.cpp ${PROFILE_FLAG} ${WARN_FLAGS}"
if [ -n "$WARN_FLAGS" ]; then
    g++ *.cpp gpu_engine.o -O3 -march=native -g -fopenmp -mavx512f -mavx512vbmi -mavx512dq -mavx2 -mfma -funroll-loops -fomit-frame-pointer -mprefer-vector-width=512 -ffast-math -std=c++17 -pthread $PROFILE_FLAG $WARN_FLAGS -lgmpxx -lgmp -L/usr/local/cuda/lib64 -lcudart -lcufft -o prothlogic 2>&1 | tee build_warnings.txt
    echo "[build] warnings saved to build_warnings.txt ($(grep -c 'warning:' build_warnings.txt || true) warnings)"
else
    g++ *.cpp gpu_engine.o -O3 -march=native -g -fopenmp -mavx512f -mavx512vbmi -mavx512dq -mavx2 -mfma -funroll-loops -fomit-frame-pointer -mprefer-vector-width=512 -ffast-math -std=c++17 -pthread $PROFILE_FLAG -lgmpxx -lgmp -L/usr/local/cuda/lib64 -lcudart -lcufft -o prothlogic
fi

echo "[build] done: ./prothlogic"
