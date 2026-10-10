// Copyright (C) Damian Koba.
#ifndef FFT_ENGINE_AVX2_H
#define FFT_ENGINE_AVX2_H

#include "fft_types.h"
#include <cstddef>
#include <cstdint>

bool fft_init_avx2(FFTContext& ctx, FFTBackend backend, size_t fft_len);
double* fft_stockham_radix8_avx2(FFTContext& ctx, bool inverse);
void unpack_karatsuba_overlap_add_avx2(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val);
void fft_square_karatsuba_avx2(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val);
void fft_mul_karatsuba_avx2(FFTContext& ctx, uint64_t* out, const uint64_t* X, const uint64_t* Y, size_t n_limbs, unsigned n_bits, uint64_t k_val);

#endif