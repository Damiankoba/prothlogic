// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "fft_pack_vbmi.h"
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <utility>
#include <atomic>
#include <gmp.h>
#include <immintrin.h>
#include <emmintrin.h>


// =====================================================================
// BALANCED DIGITS (16-bit): x = sum d_i * 2^(16 i), d_i in [-32768, 32767].
// Instead of a carry chain per chunk: y = x + K, where K has 0x8000 in every chunk; then
// d_i = chunk_i(y) - 32768 (no carries when subtracting). The addition y = x + K is a plain multi-precision
// addition (_addcarry_u64), computed in parallel segments with carries propagated between segments.
// This lowers the amplitude and removes the DC component of the spectrum, so the FFT rounding error drops by
// two orders of magnitude (allows larger N for the same FFT length). LOTR_NO_BALANCED=1 disables it (K=0, bias=0) for A/B.
// =====================================================================
static bool balanced_digits_enabled() {
    static const bool v = []() { const char* e = std::getenv("LOTR_NO_BALANCED"); return !(e && e[0] == '1'); }();
    return v;
}

// y = x + K (K = 0x8000800080008000 in every limb); returns the carry out of the top limb.
static unsigned char balanced_make_y(const uint64_t* x, size_t n, uint64_t* y) {
    const unsigned long long K = balanced_digits_enabled() ? 0x8000800080008000ULL : 0ULL;
    auto seg = [&](size_t lo, size_t hi) -> unsigned char {
        unsigned char c = 0;
        for (size_t i = lo; i < hi; ++i) {
            unsigned long long r;
            c = _addcarry_u64(c, x[i], K, &r);
            y[i] = r;
        }
        return c;
    };
    int T = omp_get_max_threads();
    if (T > 64) T = 64;
    if (T <= 1 || n < 8192) return seg(0, n);

    unsigned char co[64];
    int used = 1;
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        if (t == 0) used = nt;
        const size_t b = n * (size_t)t / (size_t)nt;
        const size_t e = n * (size_t)(t + 1) / (size_t)nt;
        co[t] = seg(b, e);
    }
    // Carries between segments: add 1 at the start of the next segment and propagate if needed.
    unsigned char in = 0;
    for (int t = 0; t < used; ++t) {
        const size_t b = n * (size_t)t / (size_t)used;
        const size_t e = n * (size_t)(t + 1) / (size_t)used;
        unsigned char out = co[t];
        if (in) {
            size_t idx = b;
            unsigned char c = 1;
            while (c && idx < e) {
                y[idx] = y[idx] + 1;
                c = (y[idx] == 0) ? 1 : 0;
                ++idx;
            }
            out = (unsigned char)(out | c);
        }
        in = out;
    }
    return in;
}


// y = x + K (K[i] or the constant Kc when K == nullptr) over limbs [0, n) inside a parallel region: this thread's
// segment with carry-in 0, barrier, then (single) the carries between segments. All threads must call it.
// Returns nothing; *top_carry gets the carry out of the top limb (valid after the call).
static void make_y_in_region(const uint64_t* x, size_t nx, const uint64_t* K, uint64_t Kc, size_t n, uint64_t* y,
                             unsigned char* co, int* used, unsigned char* top_carry) {
    const int t = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    if (t == 0) *used = nt;
    const size_t b = n * (size_t)t / (size_t)nt;
    const size_t e = n * (size_t)(t + 1) / (size_t)nt;
    unsigned char c = 0;
    for (size_t i = b; i < e; ++i) {
        unsigned long long r;
        c = _addcarry_u64(c, (i < nx) ? x[i] : 0ULL, K ? K[i] : Kc, &r);
        y[i] = r;
    }
    co[t] = c;
#pragma omp barrier
#pragma omp single
    {
        const int u_n = *used;
        unsigned char in = 0;
        for (int u = 0; u < u_n; ++u) {
            size_t idx = n * (size_t)u / (size_t)u_n;
            const size_t ue = n * (size_t)(u + 1) / (size_t)u_n;
            unsigned char o = co[u];
            if (in) {
                unsigned char cc = 1;
                while (cc && idx < ue) {
                    y[idx] = y[idx] + 1;
                    cc = (y[idx] == 0) ? 1 : 0;
                    ++idx;
                }
                o = (unsigned char)(o | cc);
            }
            in = o;
        }
        *top_carry = in;
    }
}

static void balanced_warn_top_carry() {
    static bool warned = false;
    // The calibration (dcore.cpp) squares a number filled with ones up to the top limb, where this carry is normal.
    // In a real test x < N and the FFT has headroom, so we only warn with LOTR_VERIFY_UNPACK=1.
    if (!warned && std::getenv("LOTR_VERIFY_UNPACK") != nullptr) {
        warned = true;
        std::fprintf(stderr, "[!] BALANCED PACK: carry out of the top limb (number fills the whole FFT buffer) - result may be wrong\n");
    }
}

void direct_pack_square(const uint64_t* __restrict limbs, size_t n_limbs, FFTContext& ctx) {
    size_t split_chunks = ctx.fft_len / 2;
    size_t limbs_per_split = split_chunks / 4;

    // y = x + K in the calling thread's buffer (the parallel region gets the raw pointer)
    static thread_local std::vector<uint64_t> tl_y;
    if (tl_y.size() < n_limbs + 16) tl_y.resize(n_limbs + 16);
    uint64_t* const y = tl_y.data();
    if (balanced_make_y(limbs, n_limbs, y)) balanced_warn_top_carry();
    const int bias = balanced_digits_enabled() ? 32768 : 0;
    const __m512i v_bias = _mm512_set1_epi32(bias);

#pragma omp parallel
    {
        // =========================================================
        // PHASE 1: real part (RE) = lower half of the digits
        // =========================================================
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < limbs_per_split; i += 4) {
            // prefetch the input 16 limbs ahead into L1
            _mm_prefetch((const char*)&y[i + 16], _MM_HINT_T0);

            if (i + 3 < n_limbs) {
                __m256i v_in = _mm256_loadu_si256((const __m256i*) & y[i]);
                __m512i v_int32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(v_in), v_bias);

                // regular store (not streaming): keeps the data in cache for the forward FFT
                _mm512_store_pd(&ctx.re[i * 4], _mm512_cvtepi32_pd(_mm512_castsi512_si256(v_int32)));
                _mm512_store_pd(&ctx.re[i * 4 + 8], _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v_int32, 1)));
            }
            else {
                for (size_t k = 0; k < 4; k++) {
                    if (i + k < n_limbs) {
                        uint64_t L = y[i + k];
                        ctx.re[(i + k) * 4 + 0] = static_cast<double>((int)(L & 0xFFFF) - bias);
                        ctx.re[(i + k) * 4 + 1] = static_cast<double>((int)((L >> 16) & 0xFFFF) - bias);
                        ctx.re[(i + k) * 4 + 2] = static_cast<double>((int)((L >> 32) & 0xFFFF) - bias);
                        ctx.re[(i + k) * 4 + 3] = static_cast<double>((int)((L >> 48) & 0xFFFF) - bias);
                    }
                    else if ((i + k) * 4 < split_chunks) {
                        ctx.re[(i + k) * 4 + 0] = 0.0; ctx.re[(i + k) * 4 + 1] = 0.0;
                        ctx.re[(i + k) * 4 + 2] = 0.0; ctx.re[(i + k) * 4 + 3] = 0.0;
                    }
                }
            }
        }

        // =========================================================
        // PHASE 2: imaginary part (IM) = upper half of the digits
        // =========================================================
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < limbs_per_split; i += 4) {
            size_t actual_i = limbs_per_split + i;

            // PREFETCH
            _mm_prefetch((const char*)&y[actual_i + 16], _MM_HINT_T0);

            if (actual_i + 3 < n_limbs) {
                __m256i v_in = _mm256_loadu_si256((const __m256i*) & y[actual_i]);
                __m512i v_int32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(v_in), v_bias);

                _mm512_store_pd(&ctx.im[i * 4], _mm512_cvtepi32_pd(_mm512_castsi512_si256(v_int32)));
                _mm512_store_pd(&ctx.im[i * 4 + 8], _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v_int32, 1)));
            }
            else {
                for (size_t k = 0; k < 4; k++) {
                    if (actual_i + k < n_limbs) {
                        uint64_t L = y[actual_i + k];
                        ctx.im[(i + k) * 4 + 0] = static_cast<double>((int)(L & 0xFFFF) - bias);
                        ctx.im[(i + k) * 4 + 1] = static_cast<double>((int)((L >> 16) & 0xFFFF) - bias);
                        ctx.im[(i + k) * 4 + 2] = static_cast<double>((int)((L >> 32) & 0xFFFF) - bias);
                        ctx.im[(i + k) * 4 + 3] = static_cast<double>((int)((L >> 48) & 0xFFFF) - bias);
                    }
                    else if ((i + k) * 4 < split_chunks) {
                        ctx.im[(i + k) * 4 + 0] = 0.0; ctx.im[(i + k) * 4 + 1] = 0.0;
                        ctx.im[(i + k) * 4 + 2] = 0.0; ctx.im[(i + k) * 4 + 3] = 0.0;
                    }
                }
            }
        }

        // =========================================================
        // PHASE 3: zero the rest of the buffer
        // =========================================================
#pragma omp for schedule(static) nowait
        for (size_t i = split_chunks; i < ctx.fft_len; i += 8) {
            _mm512_store_pd(&ctx.re[i], _mm512_setzero_pd());
            _mm512_store_pd(&ctx.im[i], _mm512_setzero_pd());
        }
    }
}

// =====================================================================
// WIDE DIGITS (ctx.digit_bits = B in 17..20, real-FFT square only).
// x = sum d_i * 2^(B i), balanced d_i in [-2^(B-1), 2^(B-1)): y = x + K, K = 2^(B-1) in every B-bit chunk,
// d_i = chunk_i(y) - 2^(B-1) (same idea as the 16-bit balanced digits). 64 digits = 64*B bits = exactly B limbs,
// so the number is handled in blocks of B limbs (G = ceil(n_limbs / B) blocks, 64*G digits).
// =====================================================================

// y = x + K over n limbs (x[i] = 0 for i >= nx); parallel segments with carries propagated between them
// (same structure as balanced_make_y). Returns the carry out of the top limb.
static unsigned char wide_make_y(const uint64_t* x, size_t nx, const uint64_t* K, size_t n, uint64_t* y) {
    auto seg = [&](size_t lo, size_t hi) -> unsigned char {
        unsigned char c = 0;
        for (size_t i = lo; i < hi; ++i) {
            unsigned long long r;
            c = _addcarry_u64(c, (i < nx) ? x[i] : 0ULL, K[i], &r);
            y[i] = r;
        }
        return c;
    };
    int T = omp_get_max_threads();
    if (T > 64) T = 64;
    if (T <= 1 || n < 8192) return seg(0, n);

    unsigned char co[64];
    int used = 1;
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        if (t == 0) used = nt;
        const size_t b = n * (size_t)t / (size_t)nt;
        const size_t e = n * (size_t)(t + 1) / (size_t)nt;
        co[t] = seg(b, e);
    }
    unsigned char in = 0;
    for (int t = 0; t < used; ++t) {
        const size_t b = n * (size_t)t / (size_t)used;
        const size_t e = n * (size_t)(t + 1) / (size_t)used;
        unsigned char out = co[t];
        if (in) {
            size_t idx = b;
            unsigned char c = 1;
            while (c && idx < e) {
                y[idx] = y[idx] + 1;
                c = (y[idx] == 0) ? 1 : 0;
                ++idx;
            }
            out = (unsigned char)(out | c);
        }
        in = out;
    }
    return in;
}

// K = 2^(B-1) in every B-bit digit of LIM limbs (64*LIM/B digits), cached per calling thread.
static const uint64_t* wide_K(unsigned B, size_t LIM) {
    static thread_local std::vector<uint64_t> tl_K;
    static thread_local unsigned K_bits = 0;
    static thread_local size_t K_lim = 0;
    if (K_bits != B || K_lim != LIM) {
        tl_K.assign(LIM, 0);
        const size_t n_dig = 64 * LIM / B;
        for (size_t i = 0; i < n_dig; ++i) {
            const size_t bit = (size_t)B * i + (B - 1);
            tl_K[bit >> 6] |= 1ULL << (bit & 63);
        }
        K_bits = B;
        K_lim = LIM;
    }
    return tl_K.data();
}

// Byte-permutation indices and shifts for the even (lane t = digit 2t) and odd (digit 2t+1) digits of a 16-digit step.
static void wide_extract_tables(unsigned B, __m512i& v_idx_e, __m512i& v_idx_o, __m512i& v_sh_e, __m512i& v_sh_o) {
    alignas(64) uint8_t idx_e[64], idx_o[64];
    alignas(64) int64_t sh_e[8], sh_o[8];
    for (int t = 0; t < 8; ++t) {
        const unsigned be = B * (2 * t), bo = B * (2 * t + 1);
        for (int b = 0; b < 8; ++b) {
            idx_e[8 * t + b] = (uint8_t)((b < 4) ? (be >> 3) + b : 0);
            idx_o[8 * t + b] = (uint8_t)((b < 4) ? (bo >> 3) + b : 0);
        }
        sh_e[t] = be & 7;
        sh_o[t] = bo & 7;
    }
    v_idx_e = _mm512_load_si512((const void*)idx_e);
    v_idx_o = _mm512_load_si512((const void*)idx_o);
    v_sh_e = _mm512_load_si512((const void*)sh_e);
    v_sh_o = _mm512_load_si512((const void*)sh_o);
}

// Pack with B-bit digits: 16 digits (2*B bytes, a whole number of bytes) per step. One 64-byte load covers them;
// a byte permutation (VBMI) moves the 4 bytes holding each digit into its 64-bit lane, then a per-lane shift and
// mask extract the digit (shift <= 7, 7 + B <= 32). Even digits -> re, odd digits -> im (re[j] = d[2j], im[j] = d[2j+1]).
static void pack_real_wide(const uint64_t* __restrict limbs, size_t n_limbs, FFTContext& ctx) {
    const unsigned B = ctx.digit_bits;
    const size_t n_fft = ctx.fft_len;
    const size_t G = wide_blocks(n_limbs, B);
    const size_t LIM = G * B;          // limbs of whole 64-digit blocks
    const size_t n_dig = 64 * G;       // digits of x (<= n_fft)
    const size_t j_used = n_dig / 2;   // re/im [0, j_used) = digits

    // y gets 16 extra limbs: the last 64-byte load reads past the last block (those bytes are masked off)
    static thread_local std::vector<uint64_t> tl_y;
    if (tl_y.size() < LIM + 16) tl_y.assign(LIM + 16, 0);
    uint64_t* const y = tl_y.data();
    const uint64_t* const K = wide_K(B, LIM);
    int T = omp_get_max_threads();
    if (T > 64) T = 64;
    // Small sizes / 1 thread: y = x + K first (own region), then the digit extraction below.
    // Otherwise both run in one region: y over this thread's limbs (carry-in 0), barrier, the carries between
    // segments (single), then the extraction of the digits of (roughly) the same limbs.
    const bool fused = T > 1 && LIM >= 8192;
    if (!fused && wide_make_y(limbs, n_limbs, K, LIM, y)) balanced_warn_top_carry();

    __m512i v_idx_e, v_idx_o, v_sh_e, v_sh_o;
    wide_extract_tables(B, v_idx_e, v_idx_o, v_sh_e, v_sh_o);
    const __m512i v_mask = _mm512_set1_epi64((long long)((1ULL << B) - 1));
    const __m512i v_bias = _mm512_set1_epi64((long long)(1ULL << (B - 1)));
    const __mmask64 low4 = 0x0F0F0F0F0F0F0F0FULL;   // low 4 bytes of every 64-bit lane

    double* const re = ctx.re;
    double* const im = ctx.im;
    const size_t pad = ctx.lay_pad;
    const unsigned lg = ctx.lay_lg;
    auto P = [pad, lg](size_t j) -> size_t { return j + (j >> lg) * pad; };
    const uint8_t* const yb = reinterpret_cast<const uint8_t*>(y);
    const size_t n_steps = n_dig / 16;
    const size_t j_zero_end = (ctx.pack_zero_end != 0 && ctx.pack_zero_end < n_fft) ? ctx.pack_zero_end : n_fft;

    unsigned char co[64];
    int used = 1;
    unsigned char top_c = 0;
#pragma omp parallel num_threads(fused ? T : omp_get_max_threads())
    {
        if (fused) make_y_in_region(limbs, n_limbs, K, 0, LIM, y, co, &used, &top_c);
#pragma omp for schedule(static) nowait
        for (size_t q = 0; q < n_steps; ++q) {
            const __m512i win = _mm512_loadu_si512((const void*)(yb + 2 * (size_t)B * q));
            __m512i de = _mm512_maskz_permutexvar_epi8(low4, v_idx_e, win);
            __m512i dd = _mm512_maskz_permutexvar_epi8(low4, v_idx_o, win);
            de = _mm512_sub_epi64(_mm512_and_si512(_mm512_srlv_epi64(de, v_sh_e), v_mask), v_bias);
            dd = _mm512_sub_epi64(_mm512_and_si512(_mm512_srlv_epi64(dd, v_sh_o), v_mask), v_bias);
            const size_t p = P(8 * q);
            _mm512_store_pd(&re[p], _mm512_cvtepi64_pd(de));
            _mm512_store_pd(&im[p], _mm512_cvtepi64_pd(dd));
        }
#pragma omp for schedule(static) nowait
        for (size_t j = j_used; j < j_zero_end; ++j) {
            const size_t p = P(j);
            re[p] = 0.0;
            im[p] = 0.0;
        }
    }
    if (fused && top_c) balanced_warn_top_carry();
}

// Pack fused into pass A (see DigitSrc in fft_pack_vbmi.h): only y = x + K here (the same y and digits as
// direct_pack_square_real); pass A extracts the digits of the columns it needs with digits8.
void pack_prepare_digits(const uint64_t* limbs, size_t n_limbs, const FFTContext& ctx, DigitSrc& ds) {
    const unsigned B = ctx.digit_bits;
    static thread_local std::vector<uint64_t> tl_yd;
    ds.bits = B;

    // Local balanced digits straight from x (default): nothing to compute here, only the window tables.
    // Wide digits need x to fill whole 64-digit blocks (n_limbs = G*B, always in the fused path).
    static const bool use_y = !balanced_digits_enabled();
    const bool local_ok = !use_y && (B == 16 || n_limbs == wide_blocks(n_limbs, B) * B);
    if (local_ok) {
        ds.local = true;
        ds.yb = reinterpret_cast<const uint8_t*>(limbs);
        if (B == 16) {
            ds.j_used = 2 * n_limbs;
        }
        else {
            ds.j_used = 32 * wide_blocks(n_limbs, B);
            // fields 2t, 2t+1 and 2t-1 of a 16-field step, in a window starting 8 bytes (64 bits) earlier
            alignas(64) uint8_t ie[64], io[64], ip[64];
            alignas(64) int64_t se[8], so[8], sp[8];
            for (int t = 0; t < 8; ++t) {
                const unsigned be = B * (2 * t) + 64, bo = B * (2 * t + 1) + 64, bp = B * (2 * t) + 64 - B;
                for (int b = 0; b < 8; ++b) {
                    ie[8 * t + b] = (uint8_t)((b < 4) ? (be >> 3) + b : 0);
                    io[8 * t + b] = (uint8_t)((b < 4) ? (bo >> 3) + b : 0);
                    ip[8 * t + b] = (uint8_t)((b < 4) ? (bp >> 3) + b : 0);
                }
                se[t] = be & 7;
                so[t] = bo & 7;
                sp[t] = bp & 7;
            }
            ds.lidx_e = _mm512_load_si512((const void*)ie);
            ds.lidx_o = _mm512_load_si512((const void*)io);
            ds.lidx_p = _mm512_load_si512((const void*)ip);
            ds.lsh_e = _mm512_load_si512((const void*)se);
            ds.lsh_o = _mm512_load_si512((const void*)so);
            ds.lsh_p = _mm512_load_si512((const void*)sp);
        }
        return;
    }
    ds.local = false;
    if (B == 16) {
        if (tl_yd.size() < n_limbs + 16) tl_yd.assign(n_limbs + 16, 0);
        if (balanced_make_y(limbs, n_limbs, tl_yd.data())) balanced_warn_top_carry();
        ds.j_used = 2 * n_limbs;
        ds.bias16 = balanced_digits_enabled() ? 32768 : 0;
    }
    else {
        const size_t G = wide_blocks(n_limbs, B);
        const size_t LIM = G * B;
        if (tl_yd.size() < LIM + 16) tl_yd.assign(LIM + 16, 0);
        if (wide_make_y(limbs, n_limbs, wide_K(B, LIM), LIM, tl_yd.data())) balanced_warn_top_carry();
        ds.j_used = 32 * G;
        wide_extract_tables(B, ds.idx_e, ds.idx_o, ds.sh_e, ds.sh_o);
        ds.mask = _mm512_set1_epi64((long long)((1ULL << B) - 1));
        ds.bias = _mm512_set1_epi64((long long)(1ULL << (B - 1)));
    }
    ds.yb = reinterpret_cast<const uint8_t*>(tl_yd.data());
}

// Packing for the real-FFT square (fft_square_real in fft_engine.cpp): a sequence of 2*fft_len real digits
// (n_limbs*4 digits of x, the rest zero) stored as a complex sequence of length fft_len: re[j] = d[2j], im[j] = d[2j+1].
// Every 32 bits of a limb are two neighbouring digits: lower 16 bits -> re, upper 16 bits -> im.
void direct_pack_square_real(const uint64_t* __restrict limbs, size_t n_limbs, FFTContext& ctx) {
    if (ctx.digit_bits != 16) { pack_real_wide(limbs, n_limbs, ctx); return; }
    const size_t n_fft = ctx.fft_len;
    static thread_local std::vector<uint64_t> tl_y;
    if (tl_y.size() < n_limbs + 16) tl_y.resize(n_limbs + 16);
    uint64_t* const y = tl_y.data();
    // As in pack_real_wide: with several threads y = x + K runs in the same region as the extraction.
    int T = omp_get_max_threads();
    if (T > 64) T = 64;
    const bool fused = T > 1 && n_limbs >= 8192;
    if (!fused && balanced_make_y(limbs, n_limbs, y)) balanced_warn_top_carry();
    const uint64_t Kc = balanced_digits_enabled() ? 0x8000800080008000ULL : 0ULL;
    const int bias = balanced_digits_enabled() ? 32768 : 0;
    const __m256i v_bias = _mm256_set1_epi32(bias);
    const __m256i v_lo16 = _mm256_set1_epi32(0xFFFF);
    double* const re = ctx.re;
    double* const im = ctx.im;
    const size_t n_vec = n_limbs & ~(size_t)3;     // limbs handled with vectors (4 at a time = 8 digit pairs)
    const size_t j_used = 2 * n_limbs;             // re/im [0, j_used) = digits, [j_used, n_fft) = 0
    // Padded row layout (ctx.lay_pad > 0, fused square): element j at position (j >> lg)*(2^lg + pad) + (j & mask).
    // 8 consecutive j (starting at a multiple of 8) never cross a row end (rows >= 128), so vector stores are kept.
    // (j >> lg)*(2^lg + pad) + (j & mask) = j + (j >> lg)*pad; with pad = 0 this is simply j (no branches)
    const size_t pad = ctx.lay_pad;
    const unsigned lg = ctx.lay_lg;
    auto P = [pad, lg](size_t j) -> size_t { return j + (j >> lg) * pad; };

    unsigned char co[64];
    int used = 1;
    unsigned char top_c = 0;
#pragma omp parallel num_threads(fused ? T : omp_get_max_threads())
    {
        if (fused) make_y_in_region(limbs, n_limbs, nullptr, Kc, n_limbs, y, co, &used, &top_c);
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n_vec; i += 4) {
            const __m256i v = _mm256_loadu_si256((const __m256i*)&y[i]);
            const __m256i ev = _mm256_sub_epi32(_mm256_and_si256(v, v_lo16), v_bias);
            const __m256i od = _mm256_sub_epi32(_mm256_srli_epi32(v, 16), v_bias);
            const size_t p = P(2 * i);
            _mm512_store_pd(&re[p], _mm512_cvtepi32_pd(ev));
            _mm512_store_pd(&im[p], _mm512_cvtepi32_pd(od));
        }
        // Zeros above the digits; with ctx.pack_zero_end only up to that point (pass A does not read the rest)
        const size_t j_zero_end = (ctx.pack_zero_end != 0 && ctx.pack_zero_end < n_fft) ? ctx.pack_zero_end : n_fft;
#pragma omp for schedule(static) nowait
        for (size_t j = j_used; j < j_zero_end; ++j) {
            const size_t p = P(j);
            re[p] = 0.0;
            im[p] = 0.0;
        }
    }
    if (fused && top_c) balanced_warn_top_carry();
    for (size_t i = n_vec; i < n_limbs; ++i) {
        const uint64_t L = y[i];
        re[P(2 * i)]     = static_cast<double>((int)(L & 0xFFFF) - bias);
        im[P(2 * i)]     = static_cast<double>((int)((L >> 16) & 0xFFFF) - bias);
        re[P(2 * i + 1)] = static_cast<double>((int)((L >> 32) & 0xFFFF) - bias);
        im[P(2 * i + 1)] = static_cast<double>((int)((L >> 48) & 0xFFFF) - bias);
    }
}

void pack_to_fft_buffers(const uint64_t* limbs, size_t n_limbs, double* out, size_t n_fft) {
    static thread_local std::vector<uint64_t> tl_y;
    if (tl_y.size() < n_limbs + 16) tl_y.resize(n_limbs + 16);
    uint64_t* const y = tl_y.data();
    if (balanced_make_y(limbs, n_limbs, y)) balanced_warn_top_carry();
    const int bias = balanced_digits_enabled() ? 32768 : 0;
    const __m512i v_bias = _mm512_set1_epi32(bias);

    size_t i = 0;
    for (; i + 3 < n_limbs; i += 4) {
        __m256i v_in = _mm256_loadu_si256((const __m256i*) & y[i]);
        __m512i v_int32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(v_in), v_bias);
        _mm512_storeu_pd(&out[i * 4], _mm512_cvtepi32_pd(_mm512_castsi512_si256(v_int32)));
        _mm512_storeu_pd(&out[i * 4 + 8], _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v_int32, 1)));
    }
    for (; i < n_limbs; i++) {
        uint64_t L = y[i];
        out[i * 4 + 0] = static_cast<double>((int)(L & 0xFFFF) - bias);
        out[i * 4 + 1] = static_cast<double>((int)((L >> 16) & 0xFFFF) - bias);
        out[i * 4 + 2] = static_cast<double>((int)((L >> 32) & 0xFFFF) - bias);
        out[i * 4 + 3] = static_cast<double>((int)((L >> 48) & 0xFFFF) - bias);
    }
    for (size_t j = n_limbs * 4; j < n_fft; j++) {
        out[j] = 0.0;
    }
}


// Wide digits: carry ripple of one group of 64 rounded coefficients c[0..63] into B limbs out[0..B).
// digit = v & (2^B - 1), carry = v >> B (arithmetic). B is a compile-time constant and the loop is fully unrolled,
// so every bit position (B*t) and limb index is a constant: plain shifts and ORs, no branches.
template <unsigned B>
static inline __attribute__((always_inline)) void wide_ripple64(const int64_t* __restrict c, int64_t& carry,
                                                                 uint64_t* __restrict out) {
    constexpr uint64_t M = (1ULL << B) - 1;
    uint64_t w[B] = {};
#pragma GCC unroll 64
    for (unsigned t = 0; t < 64; ++t) {
        const int64_t v = c[t] + carry;
        carry = v >> B;
        const uint64_t d = (uint64_t)v & M;
        const unsigned pos = B * t, li = pos >> 6, off = pos & 63;
        w[li] |= d << off;
        if (off + B > 64) w[li + 1] |= d >> (64 - off);
    }
#pragma GCC unroll 20
    for (unsigned i = 0; i < B; ++i) out[i] = w[i];
}

// =====================================================================
// CARRY IN 8 LANES (real-FFT square, B = 16..20): a thread range of 64-coefficient groups is split into 8 equal
// segments; vector lane l runs the carry chain of segment l (the same arithmetic as wide_ripple64, 8 chains at once).
// Per step of 64 coefficients and lane: 8 loads of 8 consecutive coefficients (conversion + rounding error as before),
// 8x8 transposes (lane-major -> coefficient-major), 64 vector ripple steps with constant shifts, 8x8 transposes of the
// B finished limbs back to lane-major and stores. Every lane returns its outgoing carry; segments are joined like
// the per-thread segments (LOTR_CARRY_OLD=1 = previous scalar path, LOTR_VERIFY_UNPACK=1 compares with it).
// =====================================================================
static bool carry_lanes_enabled() {
    static const bool v = (std::getenv("LOTR_CARRY_OLD") == nullptr);
    return v;
}

// In-place 8x8 transpose of 64-bit elements: afterwards r[j] lane l = old r[l] lane j.
static inline __attribute__((always_inline)) void transpose8x8_epi64(__m512i* r) {
    const __m512i t0 = _mm512_unpacklo_epi64(r[0], r[1]);
    const __m512i t1 = _mm512_unpackhi_epi64(r[0], r[1]);
    const __m512i t2 = _mm512_unpacklo_epi64(r[2], r[3]);
    const __m512i t3 = _mm512_unpackhi_epi64(r[2], r[3]);
    const __m512i t4 = _mm512_unpacklo_epi64(r[4], r[5]);
    const __m512i t5 = _mm512_unpackhi_epi64(r[4], r[5]);
    const __m512i t6 = _mm512_unpacklo_epi64(r[6], r[7]);
    const __m512i t7 = _mm512_unpackhi_epi64(r[6], r[7]);
    const __m512i u0 = _mm512_shuffle_i64x2(t0, t2, 0x88);
    const __m512i u1 = _mm512_shuffle_i64x2(t0, t2, 0xDD);
    const __m512i u2 = _mm512_shuffle_i64x2(t4, t6, 0x88);
    const __m512i u3 = _mm512_shuffle_i64x2(t4, t6, 0xDD);
    const __m512i u4 = _mm512_shuffle_i64x2(t1, t3, 0x88);
    const __m512i u5 = _mm512_shuffle_i64x2(t1, t3, 0xDD);
    const __m512i u6 = _mm512_shuffle_i64x2(t5, t7, 0x88);
    const __m512i u7 = _mm512_shuffle_i64x2(t5, t7, 0xDD);
    r[0] = _mm512_shuffle_i64x2(u0, u2, 0x88);
    r[4] = _mm512_shuffle_i64x2(u0, u2, 0xDD);
    r[2] = _mm512_shuffle_i64x2(u1, u3, 0x88);
    r[6] = _mm512_shuffle_i64x2(u1, u3, 0xDD);
    r[1] = _mm512_shuffle_i64x2(u4, u6, 0x88);
    r[5] = _mm512_shuffle_i64x2(u4, u6, 0xDD);
    r[3] = _mm512_shuffle_i64x2(u5, u7, 0x88);
    r[7] = _mm512_shuffle_i64x2(u5, u7, 0xDD);
}

// One ripple step for coefficient T of the group (8 lanes): digit = v & (2^B-1) at bit B*T, carry = v >> B.
// The first digit touching a limb initialises it (no zeroing needed).
template <unsigned B, unsigned T>
static inline __attribute__((always_inline)) void lane_step(__m512i x, __m512i& carry, __m512i* w) {
    constexpr unsigned pos = B * T, li = pos >> 6, off = pos & 63;
    const __m512i v = _mm512_add_epi64(x, carry);
    carry = _mm512_srai_epi64(v, B);
    const __m512i d = _mm512_and_si512(v, _mm512_set1_epi64((long long)((1ULL << B) - 1)));
    if constexpr (off == 0) w[li] = d;
    else w[li] = _mm512_or_si512(w[li], _mm512_slli_epi64(d, off));
    if constexpr (off + B > 64) w[li + 1] = _mm512_srli_epi64(d, 64 - off);
}

template <unsigned B, unsigned T0, size_t... J>
static inline __attribute__((always_inline)) void lane_steps8(const __m512i* r, __m512i& carry, __m512i* w,
                                                              std::index_sequence<J...>) {
    (lane_step<B, T0 + (unsigned)J>(r[J], carry, w), ...);
}

struct LaneSrc {
    const double* A2;
    const double* B2;
    size_t n_fft;
    size_t lay_pad;
    unsigned lay_lg;
    size_t lay(size_t h) const { return h + (h >> lay_lg) * lay_pad; }
};

// Coefficients 8Q..8Q+7 of the current group of every lane: load, convert, error, transpose, 8 ripple steps.
template <unsigned B, unsigned Q>
static inline __attribute__((always_inline)) void lane_subblock(const LaneSrc& src, const size_t* hb, __m512i& carry,
                                                                __m512i* w, __m512d& v_max_err) {
    const __m512i il_idx = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    __m512i r[8];
#pragma GCC unroll 8
    for (int l = 0; l < 8; ++l) {
        const size_t h = hb[l] + 4 * Q;
        if constexpr ((Q & 1) == 0) {
            if (h + 64 < src.n_fft) {
                _mm_prefetch((const char*)&src.A2[src.lay(h + 64)], _MM_HINT_T0);
                _mm_prefetch((const char*)&src.B2[src.lay(h + 64)], _MM_HINT_T0);
            }
        }
        const size_t hp = src.lay(h);
        const __m512d r4 = _mm512_castpd256_pd512(_mm256_load_pd(&src.A2[hp]));
        const __m512d m4 = _mm512_castpd256_pd512(_mm256_load_pd(&src.B2[hp]));
        const __m512d v_sum = _mm512_permutex2var_pd(r4, il_idx, m4);
        r[l] = _mm512_cvt_roundpd_epi64(v_sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m512d v_err = _mm512_abs_pd(_mm512_sub_pd(v_sum, _mm512_cvtepi64_pd(r[l])));
        v_max_err = _mm512_max_pd(v_max_err, v_err);
    }
    transpose8x8_epi64(r);
    lane_steps8<B, 8 * Q>(r, carry, w, std::make_index_sequence<8>{});
}

template <unsigned B, size_t... Q>
static inline __attribute__((always_inline)) void lane_group(const LaneSrc& src, const size_t* hb, __m512i& carry,
                                                             __m512i* w, __m512d& v_max_err, std::index_sequence<Q...>) {
    (lane_subblock<B, (unsigned)Q>(src, hb, carry, w, v_max_err), ...);
}

// Groups [g0 + l*G8, g0 + (l+1)*G8) for lane l = 0..7 (64 coefficients -> B limbs of sq per group);
// carry_out[l] = outgoing carry of segment l (every segment starts with carry 0).
template <unsigned B>
static void carry_lanes8(const LaneSrc& src, size_t g0, size_t G8, uint64_t* sq, int64_t* carry_out, __m512d& v_max_err) {
    __m512i carry = _mm512_setzero_si512();
    for (size_t s = 0; s < G8; ++s) {
        size_t hb[8];
        uint64_t* dst[8];
        for (int l = 0; l < 8; ++l) {
            const size_t g = g0 + (size_t)l * G8 + s;
            hb[l] = 32 * g;
            dst[l] = sq + g * B;
        }
        __m512i w[B];
        lane_group<B>(src, hb, carry, w, v_max_err, std::make_index_sequence<8>{});

        // limbs back to lane-major: blocks of 8 limbs by transposes, the remaining B % 8 limbs element by element
        for (unsigned blk = 0; blk + 8 <= B; blk += 8) {
            __m512i t[8];
            for (int i = 0; i < 8; ++i) t[i] = w[blk + i];
            transpose8x8_epi64(t);
            for (int l = 0; l < 8; ++l) _mm512_storeu_si512((void*)(dst[l] + blk), t[l]);
        }
        for (unsigned i = B & ~7u; i < B; ++i) {
            alignas(64) uint64_t tmp[8];
            _mm512_store_si512((void*)tmp, w[i]);
            for (int l = 0; l < 8; ++l) dst[l][i] = tmp[l];
        }
    }
    _mm512_storeu_si512((void*)carry_out, carry);
}

// LOTR_VERIFY_UNPACK=1: every parallel unpack phase is recomputed serially and compared
// (on mismatch [VERIFY_UNPACK] MISMATCH is printed to stderr and the serial result is used).
static bool verify_unpack_enabled() {
    static const bool v = (std::getenv("LOTR_VERIFY_UNPACK") != nullptr);
    return v;
}

// Reference (original, serial) output phase of the unpack:
//   out = sq[0 .. n_bits) + rem * 2^n_bits - Q   (mod 2^(64*n_limbs)), plus the modulus k*2^n+1 if negative.
// Used as a fallback for edge cases and for verification (LOTR_VERIFY_UNPACK=1).
static void unpack_output_ref(uint64_t* out, size_t n_limbs, const uint64_t* sq, const uint64_t* Q_arr,
                              size_t split_q, unsigned split_r, uint64_t rem, uint64_t k_val)
{
    std::memset(out, 0, n_limbs * sizeof(uint64_t));
    for (size_t i = 0; i < split_q && i < n_limbs; i++) out[i] = sq[i];
    if (split_r > 0 && split_q < n_limbs) out[split_q] = sq[split_q] & ((1ULL << split_r) - 1);

    if (rem > 0 && split_q < n_limbs) {
        uint64_t rem_L = (split_r == 0) ? rem : (rem << split_r);
        uint64_t rem_H = (split_r == 0) ? 0 : (rem >> (64 - split_r));
        unsigned __int128 sum = (unsigned __int128)out[split_q] + rem_L;
        out[split_q] = (uint64_t)sum;
        uint64_t carry_add = (uint64_t)(sum >> 64);

        if (split_q + 1 < n_limbs) {
            sum = (unsigned __int128)out[split_q + 1] + rem_H + carry_add;
            out[split_q + 1] = (uint64_t)sum;
            carry_add = (uint64_t)(sum >> 64);
            size_t idx = split_q + 2;
            while (carry_add > 0 && idx < n_limbs) {
                sum = (unsigned __int128)out[idx] + carry_add;
                out[idx] = (uint64_t)sum;
                carry_add = (uint64_t)(sum >> 64);
                idx++;
            }
        }
    }

    uint8_t borrow = 0;
    for (size_t i = 0; i < n_limbs; i++) {
        unsigned long long diff;
        borrow = _subborrow_u64(borrow, out[i], Q_arr[i], &diff);
        out[i] = diff;
    }

    if (borrow > 0) {
        uint64_t local_carry = 1;
        for (size_t i = 0; i < n_limbs; i++) {
            uint64_t val_to_add = local_carry;
            local_carry = 0;
            if (i == split_q) val_to_add += (split_r == 0) ? k_val : (k_val << split_r);
            if (i == split_q + 1 && split_r > 0) val_to_add += (k_val >> (64 - split_r));
            unsigned __int128 sum = (unsigned __int128)out[i] + val_to_add;
            out[i] = (uint64_t)sum;
            local_carry += (uint64_t)(sum >> 64);
        }
    }
}

// Fast output phase: same result as unpack_output_ref, but in a single pass (no separate memset, copy and
// subtraction) and in parallel segments (borrow computed locally, then propagated sequentially, like the carries
// of the conversion). Returns false for an edge case (the caller then uses the reference version).
static bool unpack_output_fast(uint64_t* out, size_t n_limbs, const uint64_t* sq, const uint64_t* Q,
                               size_t split_q, unsigned split_r, uint64_t rem, uint64_t k_val)
{
    // Base values of limbs split_q and split_q+1 after adding rem * 2^n (as in the reference version).
    uint64_t sp0 = 0, sp1 = 0;
    if (split_q < n_limbs) {
        const uint64_t low_part = (split_r > 0) ? (sq[split_q] & ((1ULL << split_r) - 1)) : 0;
        sp0 = low_part;
        if (rem > 0) {
            const uint64_t rem_L = (split_r == 0) ? rem : (rem << split_r);
            const uint64_t rem_H = (split_r == 0) ? 0 : (rem >> (64 - split_r));
            unsigned __int128 sum = (unsigned __int128)low_part + rem_L;
            sp0 = (uint64_t)sum;
            const uint64_t carry_add = (uint64_t)(sum >> 64);
            if (split_q + 1 < n_limbs) {
                sum = (unsigned __int128)rem_H + carry_add;
                sp1 = (uint64_t)sum;
                if ((uint64_t)(sum >> 64) != 0) return false;   // carry goes further: edge case
            }
        }
    }

    const size_t endA_total = std::min(split_q, n_limbs);   // limbs [0, endA_total) = sq[i]

    // out[i] = base[i] - Q[i] - borrow for i in [lo, hi); returns the outgoing borrow (starting from 0).
    auto seg = [&](size_t lo, size_t hi) -> uint8_t {
        uint8_t br = 0;
        size_t i = lo;
        const size_t endA = std::min(hi, endA_total);
        for (; i < endA; ++i) {
            unsigned long long d;
            br = _subborrow_u64(br, sq[i], Q[i], &d);
            out[i] = d;
        }
        for (; i < hi && i <= split_q + 1; ++i) {           // special limbs split_q, split_q+1
            uint64_t b = 0;
            if (i == split_q) b = sp0;
            else if (i == split_q + 1) b = sp1;
            unsigned long long d;
            br = _subborrow_u64(br, b, Q[i], &d);
            out[i] = d;
        }
        for (; i < hi; ++i) {                                // rest: base = 0
            unsigned long long d;
            br = _subborrow_u64(br, 0, Q[i], &d);
            out[i] = d;
        }
        return br;
    };

    uint8_t borrow = 0;
    int T = omp_get_max_threads();
    if (T > 64) T = 64;
    if (T <= 1 || n_limbs < 16384) {
        borrow = seg(0, n_limbs);
    }
    else {
        uint8_t seg_br[64];
        int used = 1;
#pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num();
            const int nt = omp_get_num_threads();
            if (t == 0) used = nt;
            const size_t b = n_limbs * (size_t)t / (size_t)nt;
            const size_t e = n_limbs * (size_t)(t + 1) / (size_t)nt;
            seg_br[t] = seg(b, e);
        }
        // Borrows between segments: subtract 1 from the next segment and propagate if needed.
        uint8_t in_br = 0;
        for (int t = 0; t < used; ++t) {
            const size_t b = n_limbs * (size_t)t / (size_t)used;
            const size_t e = n_limbs * (size_t)(t + 1) / (size_t)used;
            uint8_t out_br = seg_br[t];
            if (in_br) {
                size_t idx = b;
                uint8_t c = 1;
                while (c && idx < e) {
                    const uint64_t old = out[idx];
                    out[idx] = old - 1;
                    c = (old == 0) ? 1 : 0;
                    ++idx;
                }
                out_br = (uint8_t)(out_br | c);
            }
            in_br = out_br;
        }
        borrow = in_br;
    }

    if (borrow > 0) {
        // Negative result: add the modulus k*2^n + 1 (mod 2^(64*n_limbs)). Instead of a serial loop over all limbs
        // (about every second iteration) the three terms are added with carries only as far as needed.
        auto add_at = [&](size_t i, uint64_t a) {
            while (a != 0 && i < n_limbs) {
                const uint64_t old = out[i];
                out[i] = old + a;
                a = (out[i] < old) ? 1 : 0;
                ++i;
            }
        };
        add_at(0, 1);
        add_at(split_q, (split_r == 0) ? k_val : (k_val << split_r));
        if (split_r > 0) add_at(split_q + 1, k_val >> (64 - split_r));
    }
    return true;
}

// =====================================================================
// SCALED FORM (no division by k). Values are kept as Y = X*2^n mod N. Since k*2^n == -1 (mod N), for S = Y^2:
//   -k*S = -k*L - k*H*2^n == H - k*L   (S = L + H*2^n, L = low n bits)
// and -k*Y^2 = -k*X^2*2^(2n) = (X^2)*2^n, so the result is again in the scaled form (the same for products).
// The conversions X <-> Y are done by the caller (proth_logic.cpp: start, checkpoints, end); LOTR_DIV_OLD=1 = old path.
// =====================================================================
bool unpack_scaled_enabled() {
    static const bool v = (std::getenv("LOTR_DIV_OLD") == nullptr);
    return v;
}

// Pass 1 on limbs [b, e): out[i] = H_i - (k*L)_i, starting with carry/borrow 0; returns what must be subtracted
// at limb e (multiplication carry + borrow).
static unsigned __int128 scaled_seg(uint64_t* out, const uint64_t* sq, size_t sq_len, size_t split_q, unsigned split_r,
                                    uint64_t k_val, size_t b, size_t e)
{
    const uint64_t lo_mask = (split_r > 0) ? ((1ULL << split_r) - 1) : 0;
    uint64_t mc = 0;
    unsigned char br = 0;
    for (size_t i = b; i < e; ++i) {
        const size_t s = split_q + i;
        uint64_t h = (s < sq_len) ? (sq[s] >> split_r) : 0;
        if (split_r > 0 && s + 1 < sq_len) h |= sq[s + 1] << (64 - split_r);
        const uint64_t l = (i < split_q) ? sq[i] : ((i == split_q) ? (sq[i] & lo_mask) : 0);
        const unsigned __int128 p = (unsigned __int128)l * k_val + mc;
        mc = (uint64_t)(p >> 64);
        unsigned long long d;
        br = _subborrow_u64(br, h, (uint64_t)p, &d);
        out[i] = d;
    }
    return (unsigned __int128)mc + br;
}

// Pass 1 reads H = S >> n only in n_limbs limbs: true if S has no bits above n + 64*n_limbs (always for inputs < N;
// not for unreduced inputs such as the calibration value -> the caller falls back to GMP).
static bool scaled_h_fits(const uint64_t* sq, size_t sq_len, size_t split_q, unsigned split_r, size_t n_limbs)
{
    const size_t first = split_q + n_limbs;   // limb holding bit n + 64*n_limbs (from bit split_r on)
    if (first < sq_len && (sq[first] >> split_r) != 0) return false;
    for (size_t i = first + 1; i < sq_len; ++i) {
        if (sq[i] != 0) return false;
    }
    return true;
}

// Joins the pass-1 segments [n_limbs*t/used, n_limbs*(t+1)/used): what leaves segment t is subtracted from segment t+1.
// Returns what leaves the top (2^(64*n_limbs) units).
static unsigned __int128 scaled_join(uint64_t* out, size_t n_limbs, const unsigned __int128* seg_out, int used)
{
    unsigned __int128 in = 0;
    for (int t = 0; t < used; ++t) {
        size_t i = n_limbs * (size_t)t / (size_t)used;
        const size_t e = n_limbs * (size_t)(t + 1) / (size_t)used;
        unsigned __int128 a = in;
        while (a != 0 && i < e) {
            const uint64_t old = out[i];
            const uint64_t lo = (uint64_t)a;
            out[i] = old - lo;
            a = (a >> 64) + (old < lo ? 1 : 0);
            ++i;
        }
        in = seg_out[t] + a;
    }
    return in;
}

// Pass 1: out = H - k*L over n_limbs limbs (two's complement), in parallel segments; the multiplication carry and the
// borrow leaving a segment are subtracted from the next one. Returns the final borrow (1 = negative) in *neg;
// false if k*L does not fit (cannot happen for n_limbs*64 >= n + 64).
static bool scaled_pass1(uint64_t* out, size_t n_limbs, const uint64_t* sq, size_t sq_len,
                         size_t split_q, unsigned split_r, uint64_t k_val, unsigned* neg)
{
    unsigned __int128 top = 0;   // amount subtracted at limb n_limbs (2^(64*n_limbs) units)
    int T = omp_get_max_threads();
    if (T > 64) T = 64;
    if (T <= 1 || n_limbs < 16384) {
        top = scaled_seg(out, sq, sq_len, split_q, split_r, k_val, 0, n_limbs);
    }
    else {
        unsigned __int128 seg_out[64];
        int used = 1;
#pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num();
            const int nt = omp_get_num_threads();
            if (t == 0) used = nt;
            const size_t b = n_limbs * (size_t)t / (size_t)nt;
            const size_t e = n_limbs * (size_t)(t + 1) / (size_t)nt;
            seg_out[t] = scaled_seg(out, sq, sq_len, split_q, split_r, k_val, b, e);
        }
        top = scaled_join(out, n_limbs, seg_out, used);
    }
    // H < 2^(64*n_limbs) and k*L < 2^(64*n_limbs): at most one borrow leaves the top
    if (top > 1) return false;
    *neg = (unsigned)top;
    return true;
}

// Pass 2: R = out (two's complement, sign = neg) = R_lo + R_hi*2^n with R_hi small and signed. With R_hi = q*k + r
// (0 <= r < k): R == R_lo + r*2^n - q (mod N), then at most one +-N. Returns false if R_hi does not fit (fallback).
static bool scaled_finish(uint64_t* out, size_t n_limbs, size_t split_q, unsigned split_r, uint64_t k_val, unsigned neg)
{
    const uint64_t fill = neg ? ~0ULL : 0ULL;
    auto get = [&](size_t i) -> uint64_t { return (i < n_limbs) ? out[i] : fill; };
    // bits from n upwards: hv0 = bits [n, n+64), hv1 = [n+64, n+128); everything above must be the sign fill
    const uint64_t hv0 = (split_r > 0) ? ((get(split_q) >> split_r) | (get(split_q + 1) << (64 - split_r))) : get(split_q);
    const uint64_t hv1 = (split_r > 0) ? ((get(split_q + 1) >> split_r) | (get(split_q + 2) << (64 - split_r))) : get(split_q + 1);
    if ((uint64_t)((int64_t)hv1 >> 63) != fill) return false;
    const size_t chk0 = split_q + 2 + (split_r > 0 ? 1 : 0);
    if (split_r > 0 && split_q + 2 < n_limbs && (out[split_q + 2] >> split_r) != (fill >> split_r)) return false;
    for (size_t i = chk0; i < n_limbs; ++i) {
        if (out[i] != fill) return false;
    }
    const __int128 R_hi = (__int128)(((unsigned __int128)hv1 << 64) | hv0);

    // clear bits >= n
    if (split_q < n_limbs) out[split_q] = (split_r > 0) ? (out[split_q] & ((1ULL << split_r) - 1)) : 0;
    for (size_t i = split_q + 1; i < n_limbs; ++i) out[i] = 0;

    __int128 q = R_hi / (__int128)k_val;
    __int128 r = R_hi % (__int128)k_val;
    if (r < 0) { r += (__int128)k_val; q -= 1; }
    // For inputs < N, |q| <= k + 2. A larger q (input not reduced, e.g. the all-ones calibration value) -> fallback;
    // |q| < 2^63 keeps the value below 2N, so the single +-N correction below stays exact.
    const __int128 q_lim = (__int128)1 << 63;
    if (q >= q_lim || q <= -q_lim) return false;
    const uint64_t ru = (uint64_t)r;   // r < k

    // + r*2^n
    if (split_r > 0) {
        out[split_q] |= ru << split_r;
        if (split_q + 1 < n_limbs) out[split_q + 1] = ru >> (64 - split_r);
    }
    else {
        out[split_q] = ru;
    }

    auto add_at = [&](size_t i, uint64_t a) -> uint64_t {   // returns the carry out of the top
        while (a != 0 && i < n_limbs) {
            const uint64_t old = out[i];
            out[i] = old + a;
            a = (out[i] < old) ? 1 : 0;
            ++i;
        }
        return a;
    };
    auto sub_at = [&](size_t i, uint64_t a) -> uint64_t {   // returns the borrow out of the top
        while (a != 0 && i < n_limbs) {
            const uint64_t old = out[i];
            out[i] = old - a;
            a = (old < a) ? 1 : 0;
            ++i;
        }
        return a;
    };
    auto add_N = [&]() {
        add_at(0, 1);
        add_at(split_q, (split_r == 0) ? k_val : (k_val << split_r));
        if (split_r > 0) add_at(split_q + 1, k_val >> (64 - split_r));
    };

    // - q  (|q| <= k + 2)
    if (q > 0) {
        if (sub_at(0, (uint64_t)q) != 0) add_N();   // negative -> + N
    }
    else if (q < 0) {
        add_at(0, (uint64_t)(-q));
        // value < N + k + 2, so (value >> n) <= k; value >= N  <=>  (value >> n) == k and the low part is >= 1
        const uint64_t hi = (split_r > 0) ? ((out[split_q] >> split_r) | ((split_q + 1 < n_limbs) ? (out[split_q + 1] << (64 - split_r)) : 0))
                                          : out[split_q];
        if (hi == k_val) {
            bool low_nonzero = (split_r > 0) && ((out[split_q] & ((1ULL << split_r) - 1)) != 0);
            for (size_t i = 0; !low_nonzero && i < split_q; ++i) low_nonzero = (out[i] != 0);
            if (low_nonzero) {
                // value - N = (low part) - 1
                out[split_q] = (split_r > 0) ? (out[split_q] & ((1ULL << split_r) - 1)) : 0;
                if (split_q + 1 < n_limbs) out[split_q + 1] = 0;
                sub_at(0, 1);
            }
        }
    }
    return true;
}

// Reference / fallback for the scaled form: out = (-k*S) mod N with GMP (S = sq[0 .. sq_len)).
static void scaled_ref_gmp(uint64_t* out, size_t n_limbs, const uint64_t* sq, size_t sq_len, unsigned n_bits, uint64_t k_val)
{
    mpz_t S, N;
    mpz_inits(S, N, nullptr);
    mpz_import(S, sq_len, -1, 8, 0, 0, sq);
    mpz_mul_ui(S, S, (unsigned long)k_val);
    mpz_neg(S, S);
    mpz_set_ui(N, (unsigned long)k_val);
    mpz_mul_2exp(N, N, n_bits);
    mpz_add_ui(N, N, 1);
    mpz_mod(S, S, N);
    std::memset(out, 0, n_limbs * sizeof(uint64_t));
    mpz_export(out, nullptr, -1, 8, 0, 0, S);
    mpz_clears(S, N, nullptr);
}

void unpack_karatsuba_overlap_add_vbmi(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val, double* ptr_A2, double* ptr_AB, double* ptr_B2) {

    const size_t n_fft = ctx.fft_len;
    const size_t split_chunks = n_fft / 2;
    uint64_t inv_k = (uint64_t)(((unsigned __int128)1 << 64) / k_val);
    // Wide digits (ctx.digit_bits = B != 16, real-FFT mode only): coefficient m sits at bit B*m; groups of 64
    // coefficients = 64*B bits = exactly B limbs. 16-bit digits: groups of 8 coefficients = 2 limbs.
    const unsigned dbits = ctx.digit_bits;
    const bool wide = (dbits != 16);
    const size_t wide_lim = wide ? wide_blocks(n_limbs, dbits) * dbits : 0;   // limbs of whole 64-digit blocks
    size_t max_chunks = wide ? 2 * 64 * wide_blocks(n_limbs, dbits) : n_limbs * 8;   // number of coefficients
    size_t double_limbs = wide ? 2 * wide_lim + 2 : n_limbs * 2 + 2;
    const size_t coefs_per_group = wide ? 64 : 8;
    const size_t limbs_per_group = wide ? dbits : 2;
    if (wide && ptr_AB != nullptr) {
        std::fprintf(stderr, "[ERROR] wide digits (%u bits) are supported only by the real-FFT square\n", dbits);
        ctx.last_max_diff = 1.0e30;
        return;
    }

    // Work buffers kept per calling thread (thread_local): no reallocation between iterations
    thread_local std::vector<uint64_t> sq;
    thread_local std::vector<uint64_t> Q_arr;

    if (sq.size() < double_limbs) sq.resize(double_limbs);
    if (Q_arr.size() < n_limbs + 2) Q_arr.resize(n_limbs + 2);

    size_t actual_input_chunks = n_limbs * 4;
    bool has_B_part = (actual_input_chunks > split_chunks);
    // Real-FFT mode (ptr_AB == nullptr): coefficient m = (m even) ptr_A2[m/2] : ptr_B2[m/2]
    const bool interleaved = (ptr_AB == nullptr);
    const __m512i il_idx = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    // Padded row layout (ctx.lay_pad > 0, interleaved mode only): position of element h as in direct_pack_square_real.
    // 4 consecutive h (starting at a multiple of 4) are in one row, so the 256-bit load stays aligned.
    // position = h + (h >> lg)*pad (shift + add; with pad = 0 simply h)
    const size_t lay_pad = ctx.lay_pad;
    const unsigned lay_lg = ctx.lay_lg;
    auto lay = [lay_pad, lay_lg](size_t h) -> size_t { return h + (h >> lay_lg) * lay_pad; };

    double global_max_err = 0.0;

    // Raw pointer to the calling thread's buffer: sq is thread_local, so inside parallel regions every OpenMP thread
    // must write through this pointer (its own thread_local instance would be a different, empty vector).
    uint64_t* sq_ptr = sq.data();

    // =====================================================================
    // PHASES 1+2 FUSED: conversion double -> integer and carry propagation in one pass
    // =====================================================================
#ifdef LOTR_PROFILE   // unpack phase timings: build with -DLOTR_PROFILE
    const auto tu0 = std::chrono::high_resolution_clock::now();
#endif
    // Processes groups of 8 coefficients [g_beg, g_end): double -> int conversion, rounding error, local carry (from 0).
    // Returns the outgoing carry of the segment (in units of the first limb of the next segment) and the max error.
    // Convolution coefficients can be NEGATIVE (balanced digits), so conversion and carries are signed:
    // digit = v & 0xFFFF, carry = v >> 16 (arithmetic). The full result (a square) is non-negative.
    // One group of 8 coefficients (i = 8*g): conversion, error, carries; writes limbs sq[2g], sq[2g+1].
    auto carry_group = [&](size_t i, int64_t& local_carry, __m512d& v_max_err) {
        size_t sq_idx = i >> 2;
        __m512d v_sum;
        if (interleaved) {
            const size_t h = i >> 1;   // 4 pairs (re, im) -> 8 consecutive coefficients
            // Prefetch 128 elements (16 lines) ahead: after Bailey pass B' the result sits in the L2 caches of different
            // cores (written column-wise), and fetching a line from another core's L2 is slower than from L3.
            if (h + 128 < n_fft) {
                _mm_prefetch((const char*)&ptr_A2[lay(h + 128)], _MM_HINT_T0);
                _mm_prefetch((const char*)&ptr_B2[lay(h + 128)], _MM_HINT_T0);
            }
            if (h < n_fft) {
                const size_t hp = lay(h);
                const __m512d r4 = _mm512_castpd256_pd512(_mm256_load_pd(&ptr_A2[hp]));
                const __m512d m4 = _mm512_castpd256_pd512(_mm256_load_pd(&ptr_B2[hp]));
                v_sum = _mm512_permutex2var_pd(r4, il_idx, m4);
            }
            else {
                v_sum = _mm512_setzero_pd();
            }
        }
        else {
        size_t pf_idx = i + 64;
        if (pf_idx < n_fft) _mm_prefetch((const char*)&ptr_A2[pf_idx], _MM_HINT_T0);
        if (has_B_part && pf_idx >= split_chunks && (pf_idx - split_chunks) < n_fft) _mm_prefetch((const char*)&ptr_AB[pf_idx - split_chunks], _MM_HINT_T0);
        if (has_B_part && pf_idx >= 2 * split_chunks && (pf_idx - 2 * split_chunks) < n_fft) _mm_prefetch((const char*)&ptr_B2[pf_idx - 2 * split_chunks], _MM_HINT_T0);

        __m512d v_A2 = (i < n_fft) ? _mm512_load_pd(&ptr_A2[i]) : _mm512_setzero_pd();
        __m512d v_AB = (has_B_part && (i >= split_chunks) && (i - split_chunks < n_fft)) ? _mm512_load_pd(&ptr_AB[i - split_chunks]) : _mm512_setzero_pd();
        __m512d v_B2 = (has_B_part && (i >= 2 * split_chunks) && (i - 2 * split_chunks < n_fft)) ? _mm512_load_pd(&ptr_B2[i - 2 * split_chunks]) : _mm512_setzero_pd();

        v_sum = _mm512_add_pd(_mm512_add_pd(v_A2, v_AB), v_B2);
        }

        // signed conversion (AVX-512DQ)
        __m512i v_int = _mm512_cvt_roundpd_epi64(v_sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // rounding error
        __m512d v_rounded_back = _mm512_cvtepi64_pd(v_int);
        __m512d v_err = _mm512_abs_pd(_mm512_sub_pd(v_sum, v_rounded_back));
        v_max_err = _mm512_max_pd(v_max_err, v_err);

        // 64-byte spill to L1 for the scalar carry chain
        alignas(64) int64_t temp_chunks[8];
        _mm512_store_si512((__m512i*)temp_chunks, v_int);

        // SIGNED CARRY RIPPLE (measured faster than interleaving 4 segments and than a 64-bit __int128 variant)
        int64_t v0 = temp_chunks[0] + local_carry; local_carry = v0 >> 16;
        int64_t v1 = temp_chunks[1] + local_carry; local_carry = v1 >> 16;
        int64_t v2 = temp_chunks[2] + local_carry; local_carry = v2 >> 16;
        int64_t v3 = temp_chunks[3] + local_carry; local_carry = v3 >> 16;
        sq_ptr[sq_idx++] = ((uint64_t)v0 & 0xFFFF) | (((uint64_t)v1 & 0xFFFF) << 16) | (((uint64_t)v2 & 0xFFFF) << 32) | (((uint64_t)v3 & 0xFFFF) << 48);

        int64_t v4 = temp_chunks[4] + local_carry; local_carry = v4 >> 16;
        int64_t v5 = temp_chunks[5] + local_carry; local_carry = v5 >> 16;
        int64_t v6 = temp_chunks[6] + local_carry; local_carry = v6 >> 16;
        int64_t v7 = temp_chunks[7] + local_carry; local_carry = v7 >> 16;
        sq_ptr[sq_idx++] = ((uint64_t)v4 & 0xFFFF) | (((uint64_t)v5 & 0xFFFF) << 16) | (((uint64_t)v6 & 0xFFFF) << 32) | (((uint64_t)v7 & 0xFFFF) << 48);
    };  // end of carry_group

    // Wide digits: one group = 64 coefficients (8 steps of 8): conversion and error with vectors, then the carry
    // ripple into B limbs sq[g*B .. g*B+B) (wide_ripple64, constant shifts per digit width).
    auto carry_group_wide = [&](size_t g, int64_t& local_carry, __m512d& v_max_err) {
        alignas(64) int64_t c[64];
        for (size_t s = 0; s < 8; ++s) {
            const size_t h = 32 * g + 4 * s;   // 4 pairs (re, im) -> 8 consecutive coefficients
            if (h + 128 < n_fft) {
                _mm_prefetch((const char*)&ptr_A2[lay(h + 128)], _MM_HINT_T0);
                _mm_prefetch((const char*)&ptr_B2[lay(h + 128)], _MM_HINT_T0);
            }
            __m512d v_sum = _mm512_setzero_pd();
            if (h < n_fft) {
                const size_t hp = lay(h);
                const __m512d r4 = _mm512_castpd256_pd512(_mm256_load_pd(&ptr_A2[hp]));
                const __m512d m4 = _mm512_castpd256_pd512(_mm256_load_pd(&ptr_B2[hp]));
                v_sum = _mm512_permutex2var_pd(r4, il_idx, m4);
            }
            const __m512i v_int = _mm512_cvt_roundpd_epi64(v_sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const __m512d v_err = _mm512_abs_pd(_mm512_sub_pd(v_sum, _mm512_cvtepi64_pd(v_int)));
            v_max_err = _mm512_max_pd(v_max_err, v_err);
            _mm512_store_si512((__m512i*)(c + 8 * s), v_int);
        }
        uint64_t* const dst = sq_ptr + g * dbits;
        switch (dbits) {
            case 17: wide_ripple64<17>(c, local_carry, dst); break;
            case 18: wide_ripple64<18>(c, local_carry, dst); break;
            case 19: wide_ripple64<19>(c, local_carry, dst); break;
            default: wide_ripple64<20>(c, local_carry, dst); break;
        }
    };  // end of carry_group_wide

    auto hmax = [](__m512d v) -> double {
        alignas(64) double err_arr[8];
        _mm512_store_pd(err_arr, v);
        double m = 0.0;
        for (int k = 0; k < 8; k++) {
            if (err_arr[k] > m) m = err_arr[k];
        }
        return m;
    };

    auto process_range = [&](size_t g_beg, size_t g_end, int64_t& carry_out_ref, double& max_err_out) {
        int64_t local_carry = 0;
        __m512d v_max_err = _mm512_setzero_pd();
        if (wide) { for (size_t g = g_beg; g < g_end; ++g) carry_group_wide(g, local_carry, v_max_err); }
        else      { for (size_t g = g_beg; g < g_end; ++g) carry_group(g * 8, local_carry, v_max_err); }
        max_err_out = hmax(v_max_err);
        carry_out_ref = local_carry;
    };  // end of process_range

    // Driver: for large FFTs the range is split into segments (one per thread); each computes its carries locally
    // from zero, then the incoming carries are added (or subtracted, if negative) sequentially
    // (multi-precision arithmetic: the result is identical to one serial pass).
    // Small FFTs / 1 thread: one serial pass.
    const size_t n_groups = max_chunks / coefs_per_group;
    int64_t local_carry = 0;

    // Incoming carry (signed) added to limbs [idx, lim) of a segment; returns what leaves the segment (+1, -1 or 0).
    auto add_carry_in = [&](size_t idx, size_t lim, int64_t carry) -> int64_t {
        if (carry > 0) {
            uint64_t c = (uint64_t)carry;
            while (c && idx < lim) {
                unsigned __int128 s = (unsigned __int128)sq_ptr[idx] + c;
                sq_ptr[idx] = (uint64_t)s;
                c = (uint64_t)(s >> 64);
                ++idx;
            }
            return (int64_t)c;
        }
        if (carry < 0) {
            uint64_t br = (uint64_t)(-carry);
            while (br && idx < lim) {
                const uint64_t old = sq_ptr[idx];
                sq_ptr[idx] = old - br;
                br = (old < br) ? 1 : 0;
                ++idx;
            }
            return -(int64_t)br;
        }
        return 0;
    };

    // After the carries: negative final carry check, max error, guard limbs, final carry at the top.
    auto post_carry = [&]() {
        // A square cannot be negative: a negative final carry means an FFT/memory error
        // -> force the segment to be rejected with a huge "error" (like 2^64 for values out of range).
        if (local_carry < 0) {
            global_max_err = std::max(global_max_err, 1.0e30);
            local_carry = 0;
        }
        ctx.last_max_diff = global_max_err;

        // clear the two guard limbs, then add the final carry at the top
        sq_ptr[double_limbs - 2] = 0;
        sq_ptr[double_limbs - 1] = 0;

        size_t carry_idx = n_groups * limbs_per_group;   // first limb above the coefficients (16-bit: max_chunks / 4)
        uint64_t carry_top = (uint64_t)local_carry;
        while (carry_top > 0 && carry_idx < double_limbs) {
            unsigned __int128 sum = (unsigned __int128)sq_ptr[carry_idx] + carry_top;
            sq_ptr[carry_idx] = (uint64_t)sum;
            carry_top = (uint64_t)(sum >> 64);
            carry_idx++;
        }
    };

    // Fused scaled path: carries and pass 1 of the scaled reduction (H - k*L) in ONE parallel region. Thread t carries
    // the groups holding L limbs [b_t, e_t) and H limbs [split_q + b_t, split_q + e_t) of its own pass-1 range, so
    // pass 1 reads data from its own cache (and one region start is saved).
    const size_t fs_q = n_bits / 64;
    const unsigned fs_r = n_bits % 64;
    bool fused_done = false;
    std::atomic<int> h_above{ 0 };   // fused path: some limb above H is nonzero (input not reduced)
    unsigned __int128 p1_out[64];
    int p1_used = 1;
#ifdef LOTR_PROFILE
    std::chrono::high_resolution_clock::time_point t_carry_end;
#endif

    // Lanes path (real-FFT square): units of 64 coefficients = dbits limbs; per range 8 lane segments + a scalar tail.
    const bool lanes = carry_lanes_enabled() && interleaved && dbits >= 16 && dbits <= 20
                       && (max_chunks % 64) == 0 && max_chunks <= 2 * n_fft;
    if (lanes) {
        const size_t NG = max_chunks / 64;
        const LaneSrc src{ ptr_A2, ptr_B2, n_fft, lay_pad, lay_lg };
        int T = omp_get_max_threads();
        if (T > 64) T = 64;
        if (NG * 64 < 16384) T = 1;
        const bool fuse = (T > 1) && unpack_scaled_enabled() && n_limbs >= 16384 && fs_q < n_limbs
                          && fs_q + n_limbs <= NG * dbits;
        int64_t seg_c[64][18];
        size_t seg_b[64][18], seg_e[64][18];
        int seg_n[64][2];
        double seg_err[64];
        int used = 1;
        const int n_ranges = fuse ? 2 : 1;

        // groups of thread t: one range (NG split evenly), or with fuse a low range (L limbs of its pass-1 range,
        // the last one up to the start of H) and a high range (H limbs, the last one up to NG)
        auto range_of = [&](int t, int nt, int r, size_t& gb, size_t& ge) {
            if (!fuse) {
                gb = NG * (size_t)t / (size_t)nt;
                ge = NG * (size_t)(t + 1) / (size_t)nt;
                return;
            }
            // L exists only below split_q: a thread whose pass-1 range starts above it has an empty low range
            auto lo = [&](int u) -> size_t { return std::min(n_limbs * (size_t)u / (size_t)nt, fs_q) / dbits; };
            auto hi = [&](int u) -> size_t { return (fs_q + n_limbs * (size_t)u / (size_t)nt) / dbits; };
            if (r == 0) { gb = lo(t); ge = (t + 1 < nt) ? lo(t + 1) : hi(0); }
            else        { gb = hi(t); ge = (t + 1 < nt) ? hi(t + 1) : NG; }
        };
        // carries of groups [b, e): 8 lanes + scalar tail, segments stored in slots 9*r ..
        auto run_range = [&](int t, int r, size_t b, size_t e, __m512d& vme) {
            int n = 0;
            const int base = 9 * r;
            const size_t G8 = (e > b) ? (e - b) / 8 : 0;
            if (G8 > 0) {
                alignas(64) int64_t co[8];
                switch (dbits) {
                    case 16: carry_lanes8<16>(src, b, G8, sq_ptr, co, vme); break;
                    case 17: carry_lanes8<17>(src, b, G8, sq_ptr, co, vme); break;
                    case 18: carry_lanes8<18>(src, b, G8, sq_ptr, co, vme); break;
                    case 19: carry_lanes8<19>(src, b, G8, sq_ptr, co, vme); break;
                    default: carry_lanes8<20>(src, b, G8, sq_ptr, co, vme); break;
                }
                for (int l = 0; l < 8; ++l, ++n) {
                    seg_b[t][base + n] = b + (size_t)l * G8;
                    seg_e[t][base + n] = b + (size_t)(l + 1) * G8;
                    seg_c[t][base + n] = co[l];
                }
            }
            const size_t tb = b + 8 * G8;
            if (tb < e) {
                int64_t c = 0;
                if (wide) { for (size_t g = tb; g < e; ++g) carry_group_wide(g, c, vme); }
                else      { for (size_t g8 = tb * 8; g8 < e * 8; ++g8) carry_group(g8 * 8, c, vme); }
                seg_b[t][base + n] = tb;
                seg_e[t][base + n] = e;
                seg_c[t][base + n] = c;
                ++n;
            }
            seg_n[t][r] = n;
        };
        auto work = [&](int t, int nt) {
            __m512d vme = _mm512_setzero_pd();
            for (int r = 0; r < n_ranges; ++r) {
                size_t b, e;
                range_of(t, nt, r, b, e);
                run_range(t, r, b, e, vme);
            }
            seg_err[t] = hmax(vme);
        };
        // sequential join of all segments in order of position (ranges r = 0, 1; threads in order), then the checks
        auto join = [&]() {
            int64_t carry = 0;
            for (int t = 0; t < used; ++t) {
                if (seg_err[t] > global_max_err) global_max_err = seg_err[t];
            }
            for (int r = 0; r < n_ranges; ++r) {
                for (int t = 0; t < used; ++t) {
                    for (int i = 9 * r; i < 9 * r + seg_n[t][r]; ++i) {
                        const int64_t adj = add_carry_in(dbits * seg_b[t][i], dbits * seg_e[t][i], carry);
                        carry = seg_c[t][i] + adj;
                    }
                }
            }
            local_carry = carry;

            // Verification (LOTR_VERIFY_UNPACK=1): recompute with the previous serial path and compare.
            if (verify_unpack_enabled()) {
                const size_t n_sq = limbs_per_group * n_groups;
                std::vector<uint64_t> sq_par(sq_ptr, sq_ptr + n_sq);
                int64_t carry_ser = 0;
                double err_ser = 0.0;
                process_range(0, n_groups, carry_ser, err_ser);
                bool ok = (carry_ser == local_carry);
                for (size_t i = 0; ok && i < n_sq; ++i) ok = (sq_par[i] == sq_ptr[i]);
                if (!ok) std::fprintf(stderr, "[VERIFY_UNPACK] MISMATCH: lane conversion+carry != serial\n");
                local_carry = carry_ser;
            }
        };

        if (T <= 1) {
            work(0, 1);
            join();
        }
        else if (!fuse) {
#pragma omp parallel num_threads(T)
            {
                const int t = omp_get_thread_num();
                const int nt = omp_get_num_threads();
                if (t == 0) used = nt;
                work(t, nt);
            }
            join();
        }
        else {
#pragma omp parallel num_threads(T)
            {
                const int t = omp_get_thread_num();
                const int nt = omp_get_num_threads();
                if (t == 0) { used = nt; p1_used = nt; }
                work(t, nt);
#pragma omp barrier
#pragma omp single
                {
                    join();
                    post_carry();
#ifdef LOTR_PROFILE
                    t_carry_end = std::chrono::high_resolution_clock::now();
#endif
                }
                // pass 1 of the scaled reduction on this thread's limbs (implicit barrier after single)
                const size_t b = n_limbs * (size_t)t / (size_t)nt;
                const size_t e = n_limbs * (size_t)(t + 1) / (size_t)nt;
                p1_out[t] = scaled_seg(out, sq_ptr, double_limbs, fs_q, fs_r, k_val, b, e);
                // this thread's slice of the check "no bits above H" (scaled_h_fits, limbs above fs_q + n_limbs)
                const size_t z0 = fs_q + n_limbs + 1;
                if (z0 < double_limbs) {
                    const size_t zb = z0 + (double_limbs - z0) * (size_t)t / (size_t)nt;
                    const size_t ze = z0 + (double_limbs - z0) * (size_t)(t + 1) / (size_t)nt;
                    uint64_t any = 0;
                    for (size_t i = zb; i < ze; ++i) any |= sq_ptr[i];
                    if (any != 0) h_above = 1;
                }
            }
            fused_done = true;
        }
    }
    else
    {
        int T = omp_get_max_threads();
        if (T > 64) T = 64;
        if (T <= 1 || n_groups * coefs_per_group < 16384) {
            process_range(0, n_groups, local_carry, global_max_err);
        }
        else {
            int64_t seg_carry[64];
            double seg_err[64];
            int used = 1;
#pragma omp parallel num_threads(T)
            {
                const int t = omp_get_thread_num();
                const int nt = omp_get_num_threads();
                if (t == 0) used = nt;
                const size_t b = n_groups * (size_t)t / (size_t)nt;
                const size_t e = n_groups * (size_t)(t + 1) / (size_t)nt;
                process_range(b, e, seg_carry[t], seg_err[t]);
            }

            int64_t carry = 0;
            for (int t = 0; t < used; ++t) {
                const size_t b = n_groups * (size_t)t / (size_t)used;
                const size_t e = n_groups * (size_t)(t + 1) / (size_t)used;
                if (seg_err[t] > global_max_err) global_max_err = seg_err[t];
                size_t idx = limbs_per_group * b;
                const size_t lim = limbs_per_group * e;
                int64_t carry_adj = 0;   // carry (+1) or borrow (-1) leaving the segment
                if (carry > 0) {
                    uint64_t c = (uint64_t)carry;
                    while (c && idx < lim) {
                        unsigned __int128 s = (unsigned __int128)sq_ptr[idx] + c;
                        sq_ptr[idx] = (uint64_t)s;
                        c = (uint64_t)(s >> 64);
                        ++idx;
                    }
                    carry_adj = (int64_t)c;
                }
                else if (carry < 0) {
                    uint64_t br = (uint64_t)(-carry);
                    while (br && idx < lim) {
                        const uint64_t old = sq_ptr[idx];
                        sq_ptr[idx] = old - br;
                        br = (old < br) ? 1 : 0;
                        ++idx;
                    }
                    carry_adj = -(int64_t)br;
                }
                carry = seg_carry[t] + carry_adj;
            }
            local_carry = carry;

            // Verification (LOTR_VERIFY_UNPACK=1): recompute serially and compare with the parallel result.
            if (verify_unpack_enabled()) {
                const size_t n_sq = limbs_per_group * n_groups;
                std::vector<uint64_t> sq_par(sq_ptr, sq_ptr + n_sq);
                int64_t carry_ser = 0;
                double err_ser = 0.0;
                process_range(0, n_groups, carry_ser, err_ser);
                bool ok = (carry_ser == local_carry);
                for (size_t i = 0; ok && i < n_sq; ++i) ok = (sq_par[i] == sq_ptr[i]);
                if (!ok) std::fprintf(stderr, "[VERIFY_UNPACK] MISMATCH: parallel conversion+carry != serial\n");
                local_carry = carry_ser;
            }
        }
    }

    if (!fused_done) post_carry();
#ifdef LOTR_PROFILE
    const auto tu1 = fused_done ? t_carry_end : std::chrono::high_resolution_clock::now();
#endif

    // =====================================================================
    // SCALED FORM (default): out = H - k*L (mod N) instead of the division by k (see scaled_pass1 / scaled_finish).
    // Profile slots: "div" = pass 1 (H - k*L), "out" = pass 2 (small correction).
    // =====================================================================
    if (unpack_scaled_enabled()) {
        const size_t s_q = n_bits / 64;
        const unsigned s_r = n_bits % 64;
        unsigned neg = 0;
        bool ok;
        if (fused_done) {   // pass 1 already ran in the carry region: only the join of its segments
            const unsigned __int128 top = scaled_join(out, n_limbs, p1_out, p1_used);
            ok = (top <= 1);
            neg = (unsigned)top;
        }
        else {
            ok = scaled_pass1(out, n_limbs, sq_ptr, double_limbs, s_q, s_r, k_val, &neg);
        }
        if (ok) {
            if (fused_done) {   // limbs above were checked in the region; here only the partial limb
                const size_t first = s_q + n_limbs;
                ok = (h_above == 0) && !(first < double_limbs && (sq_ptr[first] >> s_r) != 0);
            }
            else {
                ok = scaled_h_fits(sq_ptr, double_limbs, s_q, s_r, n_limbs);
            }
        }
#ifdef LOTR_PROFILE
        const auto ts2 = std::chrono::high_resolution_clock::now();
#endif
        if (ok) ok = scaled_finish(out, n_limbs, s_q, s_r, k_val, neg);
        if (!ok) {
            scaled_ref_gmp(out, n_limbs, sq_ptr, double_limbs, n_bits, k_val);   // edge case
        }
        else if (verify_unpack_enabled()) {
            // Verification (LOTR_VERIFY_UNPACK=1): compare with GMP.
            std::vector<uint64_t> ref(n_limbs);
            scaled_ref_gmp(ref.data(), n_limbs, sq_ptr, double_limbs, n_bits, k_val);
            if (std::memcmp(ref.data(), out, n_limbs * sizeof(uint64_t)) != 0) {
                std::fprintf(stderr, "[VERIFY_UNPACK] MISMATCH: scaled reduction (H - k*L) != GMP\n");
                std::memcpy(out, ref.data(), n_limbs * sizeof(uint64_t));
            }
        }
#ifdef LOTR_PROFILE
        const auto ts3 = std::chrono::high_resolution_clock::now();
        ctx.prof_unpack_clear    += std::chrono::duration<double>(tu1 - tu0).count();
        ctx.prof_unpack_scalar   += std::chrono::duration<double>(ts2 - tu1).count();
        ctx.prof_unpack_finalfix += std::chrono::duration<double>(ts3 - ts2).count();
#endif
        return;
    }

    // =====================================================================
    // PHASE 3: DIVISION H / k (H = upper part of the square above bit n), via a precomputed inverse of k
    // =====================================================================
    size_t split_q = n_bits / 64;
    unsigned split_r = n_bits % 64;
    uint64_t rem = 0;

    // Raw pointers: sq and Q_arr are thread_local (every OpenMP thread would see its OWN instance),
    // so inside parallel regions only the calling thread's pointers may be used.
    const uint64_t* const sq_rd = sq.data();
    const size_t sq_len = sq.size();
    uint64_t* const Q_ptr = Q_arr.data();

    uint64_t inv_k_64 = 0;
    if (k_val > 0xFFFFFFFFull) {
        unsigned __int128 num = ~((unsigned __int128)0);
        inv_k_64 = (uint64_t)(num / k_val);
    }

    // Division chain H/k for limbs i in [lo, hi) from the top; starts with remainder rem_start, returns the final remainder.
    auto div_chain = [&](size_t hi, size_t lo, uint64_t rem_start) -> uint64_t {
    uint64_t rem = rem_start;
    for (ptrdiff_t i = (ptrdiff_t)hi - 1; i >= (ptrdiff_t)lo; i--) {
        size_t src_idx = split_q + i;
        uint64_t low = 0;
        if (src_idx < sq_len) {
            low = sq_rd[src_idx] >> split_r;
            if (split_r > 0 && src_idx + 1 < sq_len) {
                low |= (sq_rd[src_idx + 1] << (64 - split_r));
            }
        }

        uint64_t q_val;
        if (k_val <= 0xFFFFFFFFull) {
            uint32_t low_H = (uint32_t)(low >> 32);
            uint32_t low_L = (uint32_t)(low & 0xFFFFFFFF);

            uint64_t val_H = (rem << 32) | low_H;
            uint64_t q_H = (uint64_t)(((unsigned __int128)val_H * inv_k) >> 64);
            uint64_t r_H = val_H - q_H * k_val;
            if (r_H >= k_val) { q_H++; r_H -= k_val; }
            rem = r_H;

            uint64_t val_L = (rem << 32) | low_L;
            uint64_t q_L = (uint64_t)(((unsigned __int128)val_L * inv_k) >> 64);
            uint64_t r_L = val_L - q_L * k_val;
            if (r_L >= k_val) { q_L++; r_L -= k_val; }
            rem = r_L;

            q_val = (q_H << 32) | q_L;
        }
        else {
            uint64_t high = rem;
            unsigned __int128 prod = (unsigned __int128)high * inv_k_64;
            q_val = (uint64_t)((prod + low) >> 64) + high;
            rem = low - q_val * k_val;

            uint64_t mask1 = (rem >= k_val) ? 1 : 0;
            q_val += mask1;
            rem -= mask1 * k_val;

            uint64_t mask2 = (rem >= k_val) ? 1 : 0;
            q_val += mask2;
            rem -= mask2 * k_val;
        }
        Q_ptr[i] = q_val;
    }
    return rem;
    };  // end of div_chain

    // Up to 4 independent division chains (segments [lo[c], hi[c]) with incoming remainder rr[c]) interleaved step by step.
    // Every step waits for the previous remainder (128-bit multiply + correction), so one chain leaves the core mostly
    // idle; 4 chains at once fill the gaps. Path k <= 2^32 only. On return rr[c] = final remainder.
    auto div_chains4 = [&](const size_t* hi, const size_t* lo, uint64_t* rr, int nc) {
        size_t len[4] = { 0, 0, 0, 0 };
        size_t maxlen = 0;
        for (int c = 0; c < nc; ++c) { len[c] = hi[c] - lo[c]; maxlen = std::max(maxlen, len[c]); }
        for (size_t j = 0; j < maxlen; ++j) {
#pragma GCC unroll 4
            for (int c = 0; c < 4; ++c) {
                if (j >= len[c]) continue;
                const size_t i = hi[c] - 1 - j;
                const size_t src_idx = split_q + i;
                uint64_t low = 0;
                if (src_idx < sq_len) {
                    low = sq_rd[src_idx] >> split_r;
                    if (split_r > 0 && src_idx + 1 < sq_len) low |= (sq_rd[src_idx + 1] << (64 - split_r));
                }
                uint64_t r = rr[c];
                const uint64_t val_H = (r << 32) | (low >> 32);
                uint64_t q_H = (uint64_t)(((unsigned __int128)val_H * inv_k) >> 64);
                r = val_H - q_H * k_val;
                if (r >= k_val) { q_H++; r -= k_val; }
                const uint64_t val_L = (r << 32) | (low & 0xFFFFFFFFull);
                uint64_t q_L = (uint64_t)(((unsigned __int128)val_L * inv_k) >> 64);
                r = val_L - q_L * k_val;
                if (r >= k_val) { q_L++; r -= k_val; }
                rr[c] = r;
                Q_ptr[i] = (q_H << 32) | q_L;
            }
        }
    };

    // Division driver. For large FFTs (and k < 2^24) H is split into S segments (from the top):
    //  1) in parallel: value of every segment mod k (dot product with the weight table 2^(64 j) mod k),
    //  2) sequentially (Horner over the segments): incoming remainder of every segment,
    //  3) in parallel: the actual division chain of every segment with its known incoming remainder.
    // The result (Q_arr and rem) is identical to one serial pass.
    {
        int T = omp_get_max_threads();
        if (T > 64) T = 64;
        const bool fast_ok = (k_val < (1ULL << 24));
        if (n_limbs < 4096 || !fast_ok) {
            rem = div_chain(n_limbs, 0, 0);
        }
        else {
            // 4 segments per thread (also with 1 thread: 4 interleaved chains), segments of >= 512 limbs
            const size_t S = std::min<size_t>(std::min<size_t>(4 * (size_t)T, n_limbs / 512), 256);
            const size_t L = (n_limbs + S - 1) / S;

            // Weight tables (cached per calling thread, depend only on k and L):
            //  W1[j] = 2^(64 j) mod k,  W2[j] = 2^(64 j + 32) mod k,  j = 0..L
            thread_local std::vector<uint32_t> tw1, tw2;
            thread_local uint64_t tw_k = 0;
            thread_local size_t tw_L = 0;
            if (tw_k != k_val || tw_L != L || tw1.size() != L + 1) {
                tw1.assign(L + 1, 0);
                tw2.assign(L + 1, 0);
                const uint64_t c64 = (uint64_t)((((unsigned __int128)1) << 64) % k_val);
                const uint64_t c32 = (1ULL << 32) % k_val;
                uint64_t w = 1 % k_val;
                for (size_t j = 0; j <= L; ++j) {
                    tw1[j] = (uint32_t)w;
                    tw2[j] = (uint32_t)((w * c32) % k_val);
                    w = (w * c64) % k_val;
                }
                tw_k = k_val;
                tw_L = L;
            }
            const uint32_t* const W1 = tw1.data();
            const uint32_t* const W2 = tw2.data();
            const uint64_t cL = W1[L];   // 2^(64 L) mod k

            uint64_t seg_val[256];
            uint64_t rem_in[256];
            uint64_t rem_last = 0;
            const int Si = (int)S;

#pragma omp parallel for schedule(static, 1) num_threads(T)
            for (int s = 0; s < Si - 1; ++s) {
                const size_t hi = n_limbs - (size_t)s * L;
                const size_t lo = hi - L;
                unsigned __int128 acc = 0;
                for (size_t j = 0; j < L; ++j) {
                    const size_t src_idx = split_q + lo + j;
                    uint64_t h = 0;
                    if (src_idx < sq_len) {
                        h = sq_rd[src_idx] >> split_r;
                        if (split_r > 0 && src_idx + 1 < sq_len) {
                            h |= (sq_rd[src_idx + 1] << (64 - split_r));
                        }
                    }
                    acc += (unsigned __int128)((uint64_t)(uint32_t)h * W1[j] + (h >> 32) * W2[j]);
                }
                seg_val[s] = (uint64_t)(acc % k_val);
            }

            rem_in[0] = 0;
            for (int s = 0; s < Si - 1; ++s) {
                rem_in[s + 1] = (rem_in[s] * cL + seg_val[s]) % k_val;
            }

            // groups of 4 consecutive segments, one group = 4 interleaved chains
            const int G = (Si + 3) / 4;
#pragma omp parallel for schedule(static, 1) num_threads(T)
            for (int g = 0; g < G; ++g) {
                size_t his[4], los[4];
                uint64_t rr[4];
                int nc = 0;
                for (int s = 4 * g; s < Si && s < 4 * g + 4; ++s, ++nc) {
                    his[nc] = n_limbs - (size_t)s * L;
                    los[nc] = (s == Si - 1) ? 0 : his[nc] - L;
                    rr[nc] = rem_in[s];
                }
                div_chains4(his, los, rr, nc);
                if (4 * g + nc == Si) rem_last = rr[nc - 1];
            }
            rem = rem_last;

            // Verification (LOTR_VERIFY_UNPACK=1): recompute serially and compare with the parallel result.
            if (verify_unpack_enabled()) {
                std::vector<uint64_t> Qpar(Q_ptr, Q_ptr + n_limbs);
                const uint64_t rem_ser = div_chain(n_limbs, 0, 0);
                bool ok = (rem_ser == rem);
                for (size_t i = 0; ok && i < n_limbs; ++i) ok = (Qpar[i] == Q_ptr[i]);
                if (!ok) std::fprintf(stderr, "[VERIFY_DIV] MISMATCH: parallel division != serial (k=%llu)\n", (unsigned long long)k_val);
                rem = rem_ser;   // continue with the serial (reference) result
            }
        }
    }
#ifdef LOTR_PROFILE
    const auto tu2 = std::chrono::high_resolution_clock::now();
#endif

    // =====================================================================
    // OUTPUT: out = low part + rem*2^n - Q  (mod k*2^n+1)
    // =====================================================================
    if (!unpack_output_fast(out, n_limbs, sq_rd, Q_ptr, split_q, split_r, rem, k_val)) {
        unpack_output_ref(out, n_limbs, sq_rd, Q_ptr, split_q, split_r, rem, k_val);   // edge case
    }
    else if (verify_unpack_enabled()) {
        // Verification (LOTR_VERIFY_UNPACK=1): compare with the reference version.
        std::vector<uint64_t> out_ref(n_limbs);
        unpack_output_ref(out_ref.data(), n_limbs, sq_rd, Q_ptr, split_q, split_r, rem, k_val);
        if (std::memcmp(out_ref.data(), out, n_limbs * sizeof(uint64_t)) != 0) {
            std::fprintf(stderr, "[VERIFY_UNPACK] MISMATCH: output (out) != reference\n");
            std::memcpy(out, out_ref.data(), n_limbs * sizeof(uint64_t));
        }
    }

    // Unpack phase profile (seconds): conversion+carry / division by k / output
#ifdef LOTR_PROFILE
    const auto tu3 = std::chrono::high_resolution_clock::now();
    ctx.prof_unpack_clear    += std::chrono::duration<double>(tu1 - tu0).count();
    ctx.prof_unpack_scalar   += std::chrono::duration<double>(tu2 - tu1).count();
    ctx.prof_unpack_finalfix += std::chrono::duration<double>(tu3 - tu2).count();
#endif
}


