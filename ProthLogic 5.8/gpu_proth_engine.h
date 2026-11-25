// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// GPU Proth engine (CUDA, gpu_proth_engine.cu; RES64 checked against the CPU path and GMP). Values mod N = k*2^n + 1
// live on the card in the scaled form of the CPU classic path (Y = X*2^n mod N, see proth_logic.cpp): square and mul
// return -k*S mod N, so -k*Y^2 = X^2*2^n and -k*Ya*Yb = (Xa*Xb)*2^n keep the form.
// Integer NTT (31-bit primes, exact - no round-off error). Each engine has its own CUDA stream.
// A CUDA error throws std::runtime_error (the caller hands the candidate back to the CPU; the program keeps running).
// CPU-only build (-DLOTR_NO_CUDA): gpu_stub.cpp - gpu_probe/gpu_create report that the binary has no GPU part.

#include <cstdint>
#include <string>
#include <gmp.h>

struct GpuEngine;
enum { GPU_X = 0, GPU_V0 = 1, GPU_D = 2, GPU_W = 3, GPU_SLOTS = 4 };   // value slots (Gerbicz-Li: state, v_0, d, check)

bool gpu_probe(int device, std::string& info);   // false: no such CUDA device (the reason in info); true: its name
// how many candidates of this size the card should test at once (each its own engine and stream). Classic engine:
// 2 while primes x length <= 2^20, else 1 (larger working sets thrash the L2). Hybrid engine: 3 while primes x complex
// length <= 2^18, 2 up to 3 * 2^17, else 1. The rules were measured on an RTX 5060 Ti (36 SMs, 32 MB L2); a larger
// card scales the counts above 1 by min(SMs / 36, L2 / 32 MB). LOTR_GPU_STREAMS=1..16 forces the count.
int gpu_streams(unsigned n, uint64_t k);
// the same, measured: with cands > 1 a quick test of 1, 2, ... engines at once (~0.4 s each, a line
// "[GPU] quick test ...") picks the fastest count that fits the card memory; kept per n, k. LOTR_GPU_STREAMS forces
// the count, LOTR_GPU_AUTOTUNE=0 = gpu_streams alone
int gpu_streams_auto(unsigned n, uint64_t k, size_t cands);
// nullptr when there is no plan for this n, k (the reason in info); info = the plan otherwise
GpuEngine* gpu_create(unsigned n, uint64_t k, int device, std::string& info);
void gpu_destroy(GpuEngine* g);
void gpu_set(GpuEngine* g, int slot, const mpz_t v);   // 0 <= v < N
bool gpu_get(GpuEngine* g, int slot, mpz_t v);         // v = value mod N in [0, N); false = digits out of their bounds
void gpu_copy(GpuEngine* g, int dst, int src);
void gpu_square(GpuEngine* g, int slot, unsigned count);   // count squarings in a row (queued, no wait)
void gpu_mul(GpuEngine* g, int dst, int a, int b);         // dst = -k*a*b mod N (dst may be a or b)
void gpu_sync(GpuEngine* g);                               // waits for the queued work (the thread sleeps, no spinning)
void gpu_poke(GpuEngine* g, int slot);                     // test: corrupts one digit (Gerbicz-Li injection)
