// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef FFT_ENGINE_H
#define FFT_ENGINE_H

#include "fft_types.h"
#include <cstddef>
#include <cstdint>

bool fft_init(FFTContext& ctx, FFTBackend backend, size_t fft_len);
void fft_destroy(FFTContext& ctx);

void fft_square_karatsuba(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val);

void fft_mul_karatsuba(FFTContext& ctx, uint64_t* out, const uint64_t* X, const uint64_t* Y, size_t n_limbs, unsigned n_bits, uint64_t k_val);

// true: squarings/products return -k*S mod N, so the caller keeps values as Y = X*2^n mod N (see fft_pack_vbmi.h);
// false (LOTR_DIV_OLD=1): plain S mod N.
bool fft_uses_scaled_form();

// count squarings of v in a row (scaled form); the IBDWT keeps its digits between them (limbs only at the end).
// ctx.last_max_diff = the max rounding error over the squarings.
void fft_square_repeat(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val, unsigned count);

// FFT lengths the fast squaring path handles: powers of two >= 256 and 3*2^p / 7*2^p (radix-3 / radix-7).
bool fft_len_supported(size_t len);

// FFT lengths the IBDWT square can use (fused two-pass Bailey).
bool fft_ibdwt_supported(size_t len);

// LOTR_KBENCH=1: microbenchmark of the batch-8 column FFT kernels (one thread), printed to stdout.
void fft_kernel_bench();

#endif
