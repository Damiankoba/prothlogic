// Copyright (C) Damian Koba.

#ifndef FFT_ENGINE_H
#define FFT_ENGINE_H

#include "fft_types.h"
#include <cstddef>
#include <cstdint>

bool fft_init(FFTContext& ctx, FFTBackend backend, size_t fft_len);
void fft_destroy(FFTContext& ctx);

double* fft_stockham_radix4(FFTContext& ctx, bool inverse);
double* fft_stockham_radix4(double* in_re, double* in_im, double* out_re, double* out_im, size_t n, size_t tw_stride, FFTContext& ctx, bool inverse);

void transpose_blocked_complex(const double* src_re, const double* src_im, double* dst_re, double* dst_im, size_t n1, size_t n2);
void bailey_twiddle_multiply(double* re, double* im, size_t n1, size_t n2, bool inverse);
void fft_bailey_2d(FFTContext& ctx, bool inverse);

void pack_karatsuba_limbs(const uint64_t* in, size_t n_limbs, FFTContext& ctx);
void unpack_karatsuba_overlap_add(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val);
void fft_square_karatsuba(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val);


#endif 
