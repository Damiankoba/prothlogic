// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include "fft_types.h"

// Pack fused into pass A of the fused square: instead of writing the digits into re/im (direct_pack_square_real),
// pass A reads them itself, 8 complex positions at a time.
// Default ("local" balanced digits, no pass over the number): with the B-bit fields f_i of x and h_i = top bit of f_i,
//   d_i = f_i - h_i*2^B + h_(i-1)
// so sum d_i 2^(B i) = x exactly (the terms telescope) and |d_i| <= 2^(B-1): balanced digits without a carry chain
// (they can differ from the y = x + K digits, the value and the FFT result do not). Needs the top field of x to have
// its top bit clear (always for x < N: the buffer has >= 64 spare bits).
// LOTR_NO_BALANCED=1: the plain digits of y = x + K with K = 0, as direct_pack_square_real.
struct DigitSrc {
    const uint8_t* yb = nullptr;   // y (or x with local digits) as bytes
    bool local = false;            // true: local balanced digits from x (yb = x, no extra limbs after the end)
    unsigned bits = 16;            // digit width
    size_t j_used = 0;             // complex positions holding digits (re[j] = d[2j], im[j] = d[2j+1]); multiple of 8
    int bias16 = 0;                // 16-bit digits of y: 32768 (balanced) or 0
    __m512i idx_e, idx_o, sh_e, sh_o, mask, bias;   // wide digits of y: byte permutation and shifts (as in pack_real_wide)
    __m512i lidx_e, lidx_o, lidx_p, lsh_e, lsh_o, lsh_p;   // local wide digits: fields 2t, 2t+1, 2t-1 of a window 8 bytes earlier
};

// Prepares the digit source: local digits (nothing to compute), or y = x + K with LOTR_NO_BALANCED=1.
void pack_prepare_digits(const uint64_t* limbs, size_t n_limbs, const FFTContext& ctx, DigitSrc& ds);

// Complex positions j0 .. j0+7 (j0 a multiple of 8): re = digits 2j, im = digits 2j+1 (zero for j >= j_used).
inline void digits8(const DigitSrc& d, size_t j0, __m512d& re, __m512d& im) {
    if (j0 >= d.j_used) {
        re = _mm512_setzero_pd();
        im = _mm512_setzero_pd();
        return;
    }
    if (d.local) {
        if (d.bits == 16) {
            // 4 limbs from limb j0/2: 32-bit lane t = fields 2t (low 16 bits) and 2t+1 (high 16 bits)
            const uint64_t* const x = reinterpret_cast<const uint64_t*>(d.yb);
            const __m256i v = _mm256_loadu_si256((const __m256i*)(x + j0 / 2));
            const __m256i fe = _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
            const __m256i fo = _mm256_srli_epi32(v, 16);
            const __m256i he = _mm256_srli_epi32(fe, 15);
            const __m256i ho = _mm256_srli_epi32(fo, 15);
            // h of field 2t-1: odd field of lane t-1; lane 0: top bit of the limb before (0 at the start)
            const int prev = (j0 > 0) ? (int)(x[j0 / 2 - 1] >> 63) : 0;
            __m256i hp = _mm256_permutevar8x32_epi32(ho, _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6));
            hp = _mm256_blend_epi32(hp, _mm256_set1_epi32(prev), 0x01);
            const __m256i de = _mm256_add_epi32(_mm256_sub_epi32(fe, _mm256_slli_epi32(he, 16)), hp);
            const __m256i dd = _mm256_add_epi32(_mm256_sub_epi32(fo, _mm256_slli_epi32(ho, 16)), he);
            re = _mm512_cvtepi32_pd(de);
            im = _mm512_cvtepi32_pd(dd);
            return;
        }
        // window from 8 bytes before the 16 fields (field -1 = the last field of the previous step); masked load:
        // bytes before the buffer (j0 = 0) and after the 2B+8 needed bytes are neither read nor faulting
        const unsigned B = d.bits;
        const size_t sb = (size_t)B * j0 / 4;
        const __mmask64 full = (__mmask64)((1ULL << (2 * B + 8)) - 1);
        const __mmask64 m = (j0 == 0) ? (full & ~(__mmask64)0xFF) : full;
        const __m512i win = _mm512_maskz_loadu_epi8(m, (const void*)(d.yb + sb - 8));
        const __mmask64 low4 = 0x0F0F0F0F0F0F0F0FULL;
        const __m512i fmask = _mm512_set1_epi64((long long)((1ULL << B) - 1));
        const __m512i fe = _mm512_and_si512(_mm512_srlv_epi64(_mm512_maskz_permutexvar_epi8(low4, d.lidx_e, win), d.lsh_e), fmask);
        const __m512i fo = _mm512_and_si512(_mm512_srlv_epi64(_mm512_maskz_permutexvar_epi8(low4, d.lidx_o, win), d.lsh_o), fmask);
        const __m512i fp = _mm512_and_si512(_mm512_srlv_epi64(_mm512_maskz_permutexvar_epi8(low4, d.lidx_p, win), d.lsh_p), fmask);
        const __m128i c_top = _mm_cvtsi32_si128((int)B - 1);   // shift counts in a register (B is not a constant)
        const __m128i c_b = _mm_cvtsi32_si128((int)B);
        const __m512i he = _mm512_srl_epi64(fe, c_top);
        const __m512i ho = _mm512_srl_epi64(fo, c_top);
        const __m512i hp = _mm512_srl_epi64(fp, c_top);
        const __m512i de = _mm512_add_epi64(_mm512_sub_epi64(fe, _mm512_sll_epi64(he, c_b)), hp);
        const __m512i dd = _mm512_add_epi64(_mm512_sub_epi64(fo, _mm512_sll_epi64(ho, c_b)), he);
        re = _mm512_cvtepi64_pd(de);
        im = _mm512_cvtepi64_pd(dd);
        return;
    }
    if (d.bits == 16) {
        // 8 complex positions = 16 digits of 16 bits = 4 limbs, from limb j0/2 (byte 4*j0)
        const __m256i v = _mm256_loadu_si256((const __m256i*)(d.yb + 4 * j0));
        const __m256i vb = _mm256_set1_epi32(d.bias16);
        const __m256i ev = _mm256_sub_epi32(_mm256_and_si256(v, _mm256_set1_epi32(0xFFFF)), vb);
        const __m256i od = _mm256_sub_epi32(_mm256_srli_epi32(v, 16), vb);
        re = _mm512_cvtepi32_pd(ev);
        im = _mm512_cvtepi32_pd(od);
        return;
    }
    // 16 digits of B bits = 2B bytes from byte B*j0/4 (a whole byte for j0 a multiple of 8)
    const __m512i win = _mm512_loadu_si512((const void*)(d.yb + (size_t)d.bits * j0 / 4));
    const __mmask64 low4 = 0x0F0F0F0F0F0F0F0FULL;
    __m512i de = _mm512_maskz_permutexvar_epi8(low4, d.idx_e, win);
    __m512i dd = _mm512_maskz_permutexvar_epi8(low4, d.idx_o, win);
    de = _mm512_sub_epi64(_mm512_and_si512(_mm512_srlv_epi64(de, d.sh_e), d.mask), d.bias);
    dd = _mm512_sub_epi64(_mm512_and_si512(_mm512_srlv_epi64(dd, d.sh_o), d.mask), d.bias);
    re = _mm512_cvtepi64_pd(de);
    im = _mm512_cvtepi64_pd(dd);
}

// Packs the operand (16-bit chunks) into re/im before the squaring FFT: lower half -> re, upper half -> im.
void direct_pack_square(const uint64_t* limbs, size_t n_limbs, FFTContext& ctx);

// Packing for the real-FFT square: interleaved digits re[j] = d[2j], im[j] = d[2j+1], rest of the buffer = 0.
// Digit width = ctx.digit_bits (16, or 17..20 for the wide-digit path).
void direct_pack_square_real(const uint64_t* limbs, size_t n_limbs, FFTContext& ctx);

// Wide digits (bits != 16): limbs are processed in blocks of 'bits' limbs = 64 digits (64*bits bits).
inline size_t wide_blocks(size_t n_limbs, unsigned bits) { return (n_limbs + bits - 1) / bits; }

// Number of complex positions j (re/im pairs) holding the digits of an n_limbs number in the real-FFT pack.
inline size_t real_pack_used(size_t n_limbs, unsigned bits) {
    return (bits == 16) ? 2 * n_limbs : 32 * wide_blocks(n_limbs, bits);
}

// Packs the operand into one real buffer (used by fft_mul_karatsuba).
void pack_to_fft_buffers(const uint64_t* limbs, size_t n_limbs, double* out, size_t n_fft);

// Assembles the three convolutions (A^2, AB, B^2) into a number S and reduces it mod N = k*2^n+1.
// Scaled form (default): out = -k*S mod N (no division by k); values are kept as Y = X*2^n mod N, for which
// -k*Y^2 = (X^2)*2^n and -k*Ya*Yb = (Xa*Xb)*2^n, so squarings and products stay in the same form.
// LOTR_DIV_OLD=1: out = S mod N (division of the upper part by k), values are plain X.
// ptr_AB == nullptr: real-FFT mode, coefficient m = (m even) ptr_A2[m/2] : ptr_B2[m/2].
void unpack_karatsuba_overlap_add_vbmi(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val,
                                       double* ptr_A2, double* ptr_AB, double* ptr_B2);

// true = unpack_karatsuba_overlap_add_vbmi returns the scaled form (see above); false with LOTR_DIV_OLD=1.
bool unpack_scaled_enabled();
