// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// CPU-only build (build.sh on a machine without an NVIDIA card or nvcc: -DLOTR_NO_CUDA): the GPU sieve functions of gpu_engine.h
// report that this binary has no GPU part, the GPU Proth engine (gpu_proth_engine.h) reports no device (--gpu -> CPU only).
// In a normal build this file is empty (gpu_engine.cu and gpu_proth_engine.cu provide them).

#ifdef LOTR_NO_CUDA

#include "gpu_engine.h"
#include "gpu_proth_engine.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

bool gpu_probe(int, std::string& info) {
    info = "this binary was built without CUDA (no NVIDIA card / nvcc when build.sh ran)";
    return false;
}
int gpu_streams(unsigned, uint64_t) { return 1; }
int gpu_streams_auto(unsigned, uint64_t, size_t) { return 1; }
GpuEngine* gpu_create(unsigned, uint64_t, int, std::string& info) { gpu_probe(0, info); return nullptr; }
void gpu_destroy(GpuEngine*) {}
void gpu_set(GpuEngine*, int, const mpz_t) { throw std::runtime_error("no GPU part"); }
bool gpu_get(GpuEngine*, int, mpz_t) { throw std::runtime_error("no GPU part"); }
void gpu_copy(GpuEngine*, int, int) { throw std::runtime_error("no GPU part"); }
void gpu_square(GpuEngine*, int, unsigned) { throw std::runtime_error("no GPU part"); }
void gpu_mul(GpuEngine*, int, int, int) { throw std::runtime_error("no GPU part"); }
void gpu_sync(GpuEngine*) { throw std::runtime_error("no GPU part"); }
void gpu_poke(GpuEngine*, int) { throw std::runtime_error("no GPU part"); }

[[noreturn]] static void no_gpu() {
    std::fprintf(stderr, "[GPU] this binary was built without CUDA (no NVIDIA card / nvcc when build.sh ran): "
                         "the GPU sieve is not available\n");
    std::exit(1);
}

extern "C" {
void init_gpu_sieve(size_t, uint64_t) { no_gpu(); }
void finalize_gpu_sieve(uint8_t*) { no_gpu(); }
void init_sparse_gpu_sieve(const uint64_t*, uint32_t, uint64_t) { no_gpu(); }
void finalize_sparse_gpu_sieve(uint8_t*) { no_gpu(); }
void load_small_primes_to_gpu(const uint64_t*, uint64_t) { no_gpu(); }
void generate_sieve_window_on_gpu_vram(uint64_t, uint64_t, uint64_t) { no_gpu(); }
int extract_primes_on_gpu(uint64_t, uint64_t, uint64_t) { no_gpu(); }
int extract_sparse_primes_on_gpu(uint64_t, uint64_t, uint64_t) { no_gpu(); }
void process_prime_batch_gpu_from_vram(uint32_t, uint64_t, uint64_t) { no_gpu(); }
void flush_prime_batch_gpu_from_vram(uint32_t, uint64_t, uint64_t) { no_gpu(); }
void process_sparse_prime_batch_gpu_from_vram(uint32_t) { no_gpu(); }
void flush_sparse_prime_batch_gpu_from_vram(uint32_t) { no_gpu(); }
void process_prime_batch_gpu_for_n_range(uint64_t, uint32_t, uint32_t) { no_gpu(); }
void flush_prime_batch_gpu_for_n_range(uint64_t, uint32_t, uint32_t) { no_gpu(); }
void set_gpu_sieve_mask(const uint8_t*, size_t) { no_gpu(); }
}

#endif
