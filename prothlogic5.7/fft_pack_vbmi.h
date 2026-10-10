#pragma once
#include <cstdint>
#include <cstddef>
#include "fft_types.h"

// Packs the operand (16-bit chunks) into re/im before the squaring FFT: lower half -> re, upper half -> im.
void direct_pack_square(const uint64_t* limbs, size_t n_limbs, FFTContext& ctx);

// Packing for the real-FFT square: interleaved digits re[j] = d[2j], im[j] = d[2j+1], rest of the buffer = 0.
void direct_pack_square_real(const uint64_t* limbs, size_t n_limbs, FFTContext& ctx);

// Packs the operand into one real buffer (used by fft_mul_karatsuba).
void pack_to_fft_buffers(const uint64_t* limbs, size_t n_limbs, double* out, size_t n_fft);

// Assembles the three convolutions (A^2, AB, B^2) into a number and reduces it mod k*2^n+1 (division by k).
// ptr_AB == nullptr: real-FFT mode, coefficient m = (m even) ptr_A2[m/2] : ptr_B2[m/2].
void unpack_karatsuba_overlap_add_vbmi(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val,
                                       double* ptr_A2, double* ptr_AB, double* ptr_B2);
