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

// Packing for the real-FFT square (fft_square_real in fft_engine.cpp): a sequence of 2*fft_len real digits
// (n_limbs*4 digits of x, the rest zero) stored as a complex sequence of length fft_len: re[j] = d[2j], im[j] = d[2j+1].
// Every 32 bits of a limb are two neighbouring digits: lower 16 bits -> re, upper 16 bits -> im.
void direct_pack_square_real(const uint64_t* __restrict limbs, size_t n_limbs, FFTContext& ctx) {
    const size_t n_fft = ctx.fft_len;
    static thread_local std::vector<uint64_t> tl_y;
    if (tl_y.size() < n_limbs + 16) tl_y.resize(n_limbs + 16);
    uint64_t* const y = tl_y.data();
    if (balanced_make_y(limbs, n_limbs, y)) balanced_warn_top_carry();
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

#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n_vec; i += 4) {
            const __m256i v = _mm256_loadu_si256((const __m256i*)&y[i]);
            const __m256i ev = _mm256_sub_epi32(_mm256_and_si256(v, v_lo16), v_bias);
            const __m256i od = _mm256_sub_epi32(_mm256_srli_epi32(v, 16), v_bias);
            const size_t p = P(2 * i);
            _mm512_store_pd(&re[p], _mm512_cvtepi32_pd(ev));
            _mm512_store_pd(&im[p], _mm512_cvtepi32_pd(od));
        }
#pragma omp for schedule(static) nowait
        for (size_t j = j_used; j < n_fft; ++j) {
            const size_t p = P(j);
            re[p] = 0.0;
            im[p] = 0.0;
        }
    }
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

void unpack_karatsuba_overlap_add_vbmi(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val, double* ptr_A2, double* ptr_AB, double* ptr_B2) {

    const size_t n_fft = ctx.fft_len;
    const size_t split_chunks = n_fft / 2;
    uint64_t inv_k = (uint64_t)(((unsigned __int128)1 << 64) / k_val);
    size_t max_chunks = n_limbs * 8;
    size_t double_limbs = n_limbs * 2 + 2;

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
        for (size_t g = g_beg; g < g_end; ++g) carry_group(g * 8, local_carry, v_max_err);
        max_err_out = hmax(v_max_err);
        carry_out_ref = local_carry;
    };  // end of process_range

    // Driver: for large FFTs the range is split into segments (one per thread); each computes its carries locally
    // from zero, then the incoming carries are added (or subtracted, if negative) sequentially
    // (multi-precision arithmetic: the result is identical to one serial pass).
    // Small FFTs / 1 thread: one serial pass.
    const size_t n_groups = max_chunks / 8;
    int64_t local_carry = 0;
    {
        int T = omp_get_max_threads();
        if (T > 64) T = 64;
        if (T <= 1 || n_groups < 2048) {
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
                size_t idx = 2 * b;
                const size_t lim = 2 * e;
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
                std::vector<uint64_t> sq_par(sq_ptr, sq_ptr + 2 * n_groups);
                int64_t carry_ser = 0;
                double err_ser = 0.0;
                process_range(0, n_groups, carry_ser, err_ser);
                bool ok = (carry_ser == local_carry);
                for (size_t i = 0; ok && i < 2 * n_groups; ++i) ok = (sq_par[i] == sq_ptr[i]);
                if (!ok) std::fprintf(stderr, "[VERIFY_UNPACK] MISMATCH: parallel conversion+carry != serial\n");
                local_carry = carry_ser;
            }
        }
    }

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

    size_t carry_idx = max_chunks / 4;
    uint64_t carry_top = (uint64_t)local_carry;
    while (carry_top > 0 && carry_idx < double_limbs) {
        unsigned __int128 sum = (unsigned __int128)sq_ptr[carry_idx] + carry_top;
        sq_ptr[carry_idx] = (uint64_t)sum;
        carry_top = (uint64_t)(sum >> 64);
        carry_idx++;
    }
#ifdef LOTR_PROFILE
    const auto tu1 = std::chrono::high_resolution_clock::now();
#endif

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


