// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// GPU-only build (./build.sh gpu: -DLOTR_GPU_ONLY, for CPUs without AVX-512): fft_engine.cpp and fft_pack_vbmi.cpp
// (the AVX-512 FFT of the CPU tests) are not compiled. cli.cpp then always runs --gpu-only, whose path (the sieve and
// execute_proth_test_gpu) never calls the CPU engine; these functions only satisfy the linker and stop the program
// if something reaches them anyway.
// In a normal build this file is empty (fft_engine.cpp provides them).

#ifdef LOTR_GPU_ONLY

#include "fft_engine.h"
#include <cstdio>
#include <cstdlib>
#include <string>

[[noreturn]] static void no_cpu_engine() {
    std::fprintf(stderr, "[ERROR] GPU-only build (./build.sh gpu): the CPU engine (AVX-512 FFT) is not in this binary\n");
    std::exit(1);
}

bool fft_init(FFTContext&, FFTBackend, size_t) { no_cpu_engine(); }
void fft_destroy(FFTContext&) {}
std::string get_fft_engine_name(const FFTContext&) { return "none (GPU-only build)"; }
void fft_square_karatsuba(FFTContext&, uint64_t*, size_t, unsigned, uint64_t) { no_cpu_engine(); }
void fft_mul_karatsuba(FFTContext&, uint64_t*, const uint64_t*, const uint64_t*, size_t, unsigned, uint64_t) { no_cpu_engine(); }
bool fft_uses_scaled_form() { return true; }
void fft_square_repeat(FFTContext&, uint64_t*, size_t, unsigned, uint64_t, unsigned) { no_cpu_engine(); }
bool fft_len_supported(size_t) { return false; }
bool fft_ibdwt_supported(size_t) { return false; }
void fft_kernel_bench() { no_cpu_engine(); }

#endif
