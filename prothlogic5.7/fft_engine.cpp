// Copyright (C) Damian Koba.

#include "fft_engine.h"
#include "llr_fft_pack.h"
#include "checkpoint_manager.h"
#include "fft_engine_avx2.h"
#include "fft_pack_vbmi.h"
#include "fft_factorizer.h"
#include "log_config.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <atomic>
#pragma GCC target("avx512f,avx512dq,fma")
#include <immintrin.h>
#include <sys/mman.h>
#include <omp.h>
#include <vector>

extern bool check_avx512_support();

#ifdef LOTR_PROFILE
// Timing of one pass inside the FFT (RAII): adds the object's lifetime to ctx.prof_pass[slot].
struct PassTimer {
    double& acc;
    std::chrono::high_resolution_clock::time_point t0;
    explicit PassTimer(double& a) : acc(a), t0(std::chrono::high_resolution_clock::now()) {}
    ~PassTimer() { acc += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count(); }
};
#define LOTR_PASS(ctx, slot) PassTimer lotr_pass_timer_((ctx).prof_pass[(slot)])
#else
#define LOTR_PASS(ctx, slot) do {} while (0)
#endif

// LOTR_ROW_MEMCPY=1 restores the old 2D FFT behaviour: copying row results from the out buffer back to in
// (rows with an odd number of Stockham stages). By default the copy is skipped (buffers are swapped).
static bool row_memcpy_legacy() {
    static const bool v = (std::getenv("LOTR_ROW_MEMCPY") != nullptr);
    return v;
}

// LOTR_TW_DIRECT=1: twiddle w^(r*col) in the Bailey passes read directly from the large w1 table (old version, for A/B)
static bool tw_direct_enabled() {
    static const bool v = (std::getenv("LOTR_TW_DIRECT") != nullptr);
    return v;
}

// w^e (e < N) from the two-level table ctx.tw2_* (built in bailey_setup) or (LOTR_TW_DIRECT=1) directly from w1.
// The large w1 table (2 x 8N bytes) does not fit in L2 at 2^18+, and the r*col indices jump around it - every read is a miss.
struct Tw2 {
    const double* lr; const double* li; const double* hr; const double* hi;
    const double* dr; const double* di;
    size_t mask; unsigned sh; bool direct;
    explicit Tw2(const FFTContext& ctx)
        : lr(ctx.tw2_lo_re.data()), li(ctx.tw2_lo_im.data()), hr(ctx.tw2_hi_re.data()), hi(ctx.tw2_hi_im.data()),
          dr(ctx.w1_re), di(ctx.w1_im), mask((size_t(1) << ctx.tw2_shift) - 1), sh(ctx.tw2_shift),
          direct(tw_direct_enabled() || ctx.tw2_N != ctx.fft_len) {}
    inline void get(size_t e, double& wr, double& wi) const {
        if (direct) { wr = dr[e]; wi = di[e]; return; }
        const size_t l = e & mask, h = e >> sh;
        wr = hr[h] * lr[l] - hi[h] * li[l];
        wi = hr[h] * li[l] + hi[h] * lr[l];
    }
};

// EXPERIMENT: LOTR_BAILEY_ODD=1 forces Bailey 2D also for odd powers of 2 (n1 = 2*n2)
// instead of Stockham 1D. Off by default (behaviour unchanged).
static bool force_bailey_odd() {
    static const bool v = (std::getenv("LOTR_BAILEY_ODD") != nullptr);
    return v;
}

// ENGINE CHOICE by FFT length (measured 2026-10, 6 threads, CCD1, ms/iter: new two-pass Bailey / old Bailey / Stockham):
//   2^15: 0.11/0.13/0.14   2^17: 0.38/0.42/0.46   2^19: 1.87/1.65/1.14   2^21: 21.6/37.4/46.1
// Even powers: always Bailey. Odd: Bailey for 2^15, 2^17 and >= 2^21, Stockham 1D for 2^19 and <= 2^13.
// LOTR_BAILEY_ODD=1 forces Bailey for all odd powers, LOTR_STOCKHAM_ODD=1 forces Stockham.
static bool use_bailey_engine(unsigned p) {
    if (p % 2 == 0) return true;
    static const bool force_stockham = (std::getenv("LOTR_STOCKHAM_ODD") != nullptr);
    if (force_stockham) return false;
    if (force_bailey_odd()) return true;
    return p == 15 || p == 17 || p >= 21;
}
bool fft_init(FFTContext& ctx, FFTBackend backend, size_t fft_len) {
    ctx.backend = backend;
    ctx.fft_len = fft_len;
    ctx.factor_plan = generate_fft_plan(fft_len);
    ctx.last_max_diff = 0.0;

    
    ctx.use_avx2_fallback = false;

    if (ctx.use_avx2_fallback) {
        return fft_init_avx2(ctx, backend, fft_len);
    }

    // fft_len/16 of headroom for the padded-row layout (ctx.lay_pad: up to 8 doubles per row, rows >= 128)
    size_t table_size = (fft_len + fft_len / 16 + 64) * sizeof(double);
    size_t total_size = table_size * 22;
    void* big_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (big_mem == MAP_FAILED) {
        big_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (big_mem == MAP_FAILED) return false;
        // No HUGETLB pool: ask for transparent huge pages (THP), fewer TLB misses on large-stride accesses.
        // LOTR_NO_THP=1 disables it (for comparisons).
        if (std::getenv("LOTR_NO_THP") == nullptr) madvise(big_mem, total_size, MADV_HUGEPAGE);
    }
    ctx.raw_allocation = big_mem;
    ctx.base_ptr = big_mem;
    ctx.total_allocation_size = total_size;

    std::memset(big_mem, 0, total_size);
    char* base = (char*)big_mem;

    ctx.re = (double*)(base + 0 * table_size);
    ctx.im = (double*)(base + 1 * table_size);
    ctx.re2 = (double*)(base + 2 * table_size);
    ctx.im2 = (double*)(base + 3 * table_size);
    ctx.w1_re = (double*)(base + 4 * table_size);
    ctx.w1_im = (double*)(base + 5 * table_size);
    ctx.w2_re = (double*)(base + 6 * table_size);
    ctx.w2_im = (double*)(base + 7 * table_size);
    ctx.w3_re = (double*)(base + 8 * table_size);
    ctx.w3_im = (double*)(base + 9 * table_size);
    ctx.proth_w_re = (double*)(base + 10 * table_size);
    ctx.proth_w_im = (double*)(base + 11 * table_size);
    ctx.k_A_re = (double*)(base + 12 * table_size);
    ctx.k_A_im = (double*)(base + 13 * table_size);
    ctx.k_B_re = (double*)(base + 14 * table_size);
    ctx.k_B_im = (double*)(base + 15 * table_size);
    ctx.k_AB_re = (double*)(base + 16 * table_size);
    ctx.k_AB_im = (double*)(base + 17 * table_size);

    ctx.g_C_re = (double*)(base + 18 * table_size);
    ctx.g_C_im = (double*)(base + 19 * table_size);
    ctx.g_D_re = (double*)(base + 20 * table_size);
    ctx.g_D_im = (double*)(base + 21 * table_size);

    size_t i = 0;
    for (; i + 7 < fft_len; i += 8) {
        alignas(64) double temp_w1_re[8], temp_w1_im[8];
        alignas(64) double temp_w2_re[8], temp_w2_im[8];
        alignas(64) double temp_w3_re[8], temp_w3_im[8];
        alignas(64) double temp_proth_re[8], temp_proth_im[8];

        for (int k = 0; k < 8; ++k) {
            double i_d = static_cast<double>(i + k);
            double len_d = static_cast<double>(fft_len);

            double ang = -2.0 * M_PI * i_d / len_d;
            temp_w1_re[k] = std::cos(ang);       temp_w1_im[k] = std::sin(ang);
            temp_w2_re[k] = std::cos(2.0 * ang); temp_w2_im[k] = std::sin(2.0 * ang);
            temp_w3_re[k] = std::cos(3.0 * ang); temp_w3_im[k] = std::sin(3.0 * ang);

            double p_ang = -M_PI * i_d / len_d;
            temp_proth_re[k] = std::cos(p_ang);  temp_proth_im[k] = std::sin(p_ang);
        }

        _mm512_stream_pd(&ctx.w1_re[i], _mm512_load_pd(temp_w1_re));
        _mm512_stream_pd(&ctx.w1_im[i], _mm512_load_pd(temp_w1_im));
        _mm512_stream_pd(&ctx.w2_re[i], _mm512_load_pd(temp_w2_re));
        _mm512_stream_pd(&ctx.w2_im[i], _mm512_load_pd(temp_w2_im));
        _mm512_stream_pd(&ctx.w3_re[i], _mm512_load_pd(temp_w3_re));
        _mm512_stream_pd(&ctx.w3_im[i], _mm512_load_pd(temp_w3_im));
        _mm512_stream_pd(&ctx.proth_w_re[i], _mm512_load_pd(temp_proth_re));
        _mm512_stream_pd(&ctx.proth_w_im[i], _mm512_load_pd(temp_proth_im));
    }

    _mm_sfence();
    
    for (; i < fft_len; ++i) {
        double i_d = static_cast<double>(i);
        double len_d = static_cast<double>(fft_len);
        double ang = -2.0 * M_PI * i_d / len_d;
        ctx.w1_re[i] = std::cos(ang);       ctx.w1_im[i] = std::sin(ang);
        ctx.w2_re[i] = std::cos(2.0 * ang); ctx.w2_im[i] = std::sin(2.0 * ang);
        ctx.w3_re[i] = std::cos(3.0 * ang); ctx.w3_im[i] = std::sin(3.0 * ang);
        double p_ang = -M_PI * i_d / len_d;
        ctx.proth_w_re[i] = std::cos(p_ang);  ctx.proth_w_im[i] = std::sin(p_ang);
    }

    std::string db_file = "proth_state_N" + std::to_string(fft_len) + ".bin";
    uint64_t recovered_k = 0;
    load_checkpoint(recovered_k, fft_len, reinterpret_cast<uint64_t*>(ctx.re), db_file);
    return true;
}



void fft_destroy(FFTContext& ctx) {
    if (ctx.raw_allocation && ctx.raw_allocation != MAP_FAILED) {
        munmap(ctx.raw_allocation, ctx.total_allocation_size);
        ctx.raw_allocation = nullptr;
    }
}


std::string get_fft_engine_name(const FFTContext& ctx) {
    bool is_pure_power_of_two = true;
    for (size_t f : ctx.factor_plan.factors) {
        if (f != 2) { is_pure_power_of_two = false; break; }
    }

    if (is_pure_power_of_two) {
        unsigned p = 0;
        size_t temp = ctx.fft_len;
        while (temp > 1) { temp >>= 1; p++; }

        // Engine choice: see use_bailey_engine (Bailey 2D for even powers, Stockham 1D for some odd ones).
        if (use_bailey_engine(p)) {
            return "Unified Bailey 2D (Radix-4/2) [FFT_Len = 2^" + std::to_string(p) + "]";
        }
        else {
            return "Stockham 1D (Radix-8/4/2) [FFT_Len = 2^" + std::to_string(p) + "]";
        }
    }
    else {
        return "Mixed-Radix 2D (Factors 2/3/5) [FFT_Len = " + std::to_string(ctx.fft_len) + "]";
    }
}

// =================================================================
// FACTORIZER FOR MIXED-RADIX
// =================================================================
inline FFTFactorPlan generate_fft_plan_inline(size_t n) {
    FFTFactorPlan plan;
    plan.total_n = n;
    size_t temp = n;
    plan.has_unsupported_primes = false;

    // 5, 3 and 2
    while (temp % 5 == 0) { plan.factors.push_back(5); temp /= 5; }
    while (temp % 3 == 0) { plan.factors.push_back(3); temp /= 3; }
    while (temp % 2 == 0) { plan.factors.push_back(2); temp /= 2; }

    for (size_t d = 7; d * d <= temp; d += 2) {
        while (temp % d == 0) {
            plan.factors.push_back(d);
            temp /= d;
            plan.has_unsupported_primes = true;
        }
    }
    if (temp > 1) {
        plan.factors.push_back(temp);
        plan.has_unsupported_primes = true;
    }

    return plan;
}





// =================================================================
// 2D MIXED-RADIX KERNELS (CACHE-LOCAL ROW FFTs - AVX-512) - ILP UNROLLED (test)
// =================================================================
inline void pass_radix2_seq(const double* in_re, const double* in_im, double* out_re, double* out_im,
    size_t n, size_t m, const FFTContext& ctx, bool inverse, size_t tw_step) {
    size_t remaining = n / m;
    size_t stride = remaining / 2;

    for (size_t j = 0; j < m; ++j) {
        size_t tw_idx = j * stride * tw_step;
        double tw_re = ctx.w1_re[tw_idx];
        double tw_im = inverse ? -ctx.w1_im[tw_idx] : ctx.w1_im[tw_idx];

        __m512d v_tw_re = _mm512_set1_pd(tw_re);
        __m512d v_tw_im = _mm512_set1_pd(tw_im);

        for (size_t k = 0; k < stride; ) {
            // UNROLLx2 (16 elements)
            if (stride - k >= 16) {
                size_t I0_A = k + j * remaining; size_t I1_A = I0_A + stride;
                size_t I0_B = I0_A + 8;          size_t I1_B = I1_A + 8;

                __m512d r0_A = _mm512_loadu_pd(&in_re[I0_A]), i0_A = _mm512_loadu_pd(&in_im[I0_A]);
                __m512d r1_A = _mm512_loadu_pd(&in_re[I1_A]), i1_A = _mm512_loadu_pd(&in_im[I1_A]);
                __m512d r0_B = _mm512_loadu_pd(&in_re[I0_B]), i0_B = _mm512_loadu_pd(&in_im[I0_B]);
                __m512d r1_B = _mm512_loadu_pd(&in_re[I1_B]), i1_B = _mm512_loadu_pd(&in_im[I1_B]);

                __m512d t1_re_A = _mm512_fmsub_pd(r1_A, v_tw_re, _mm512_mul_pd(i1_A, v_tw_im));
                __m512d t1_re_B = _mm512_fmsub_pd(r1_B, v_tw_re, _mm512_mul_pd(i1_B, v_tw_im));
                __m512d t1_im_A = _mm512_fmadd_pd(r1_A, v_tw_im, _mm512_mul_pd(i1_A, v_tw_re));
                __m512d t1_im_B = _mm512_fmadd_pd(r1_B, v_tw_im, _mm512_mul_pd(i1_B, v_tw_re));

                size_t O0_A = k + j * stride; size_t O1_A = O0_A + m * stride;
                size_t O0_B = O0_A + 8;       size_t O1_B = O1_A + 8;

                _mm512_storeu_pd(&out_re[O0_A], _mm512_add_pd(r0_A, t1_re_A));
                _mm512_storeu_pd(&out_im[O0_A], _mm512_add_pd(i0_A, t1_im_A));
                _mm512_storeu_pd(&out_re[O0_B], _mm512_add_pd(r0_B, t1_re_B));
                _mm512_storeu_pd(&out_im[O0_B], _mm512_add_pd(i0_B, t1_im_B));

                _mm512_storeu_pd(&out_re[O1_A], _mm512_sub_pd(r0_A, t1_re_A));
                _mm512_storeu_pd(&out_im[O1_A], _mm512_sub_pd(i0_A, t1_im_A));
                _mm512_storeu_pd(&out_re[O1_B], _mm512_sub_pd(r0_B, t1_re_B));
                _mm512_storeu_pd(&out_im[O1_B], _mm512_sub_pd(i0_B, t1_im_B));
                k += 16;
            }
            else if (stride - k >= 8) {
                size_t I0 = k + j * remaining; size_t I1 = I0 + stride;
                __m512d r0 = _mm512_loadu_pd(&in_re[I0]), i0 = _mm512_loadu_pd(&in_im[I0]);
                __m512d r1 = _mm512_loadu_pd(&in_re[I1]), i1 = _mm512_loadu_pd(&in_im[I1]);
                __m512d t1_re = _mm512_fmsub_pd(r1, v_tw_re, _mm512_mul_pd(i1, v_tw_im));
                __m512d t1_im = _mm512_fmadd_pd(r1, v_tw_im, _mm512_mul_pd(i1, v_tw_re));
                size_t O0 = k + j * stride; size_t O1 = O0 + m * stride;
                _mm512_storeu_pd(&out_re[O0], _mm512_add_pd(r0, t1_re));
                _mm512_storeu_pd(&out_im[O0], _mm512_add_pd(i0, t1_im));
                _mm512_storeu_pd(&out_re[O1], _mm512_sub_pd(r0, t1_re));
                _mm512_storeu_pd(&out_im[O1], _mm512_sub_pd(i0, t1_im));
                k += 8;
            }
            else {
                size_t I0 = k + j * remaining, I1 = I0 + stride;
                size_t O0 = k + j * stride, O1 = O0 + m * stride;
                double r0 = in_re[I0], i0 = in_im[I0], r1 = in_re[I1], i1 = in_im[I1];
                double t1_re = r1 * tw_re - i1 * tw_im, t1_im = r1 * tw_im + i1 * tw_re;
                out_re[O0] = r0 + t1_re; out_im[O0] = i0 + t1_im;
                out_re[O1] = r0 - t1_re; out_im[O1] = i0 - t1_im;
                k++;
            }
        }
    }
}

inline void pass_radix3_seq(const double* in_re, const double* in_im, double* out_re, double* out_im,
    size_t n, size_t m, const FFTContext& ctx, bool inverse, size_t tw_step) {
    size_t remaining = n / m; size_t stride = remaining / 3;
    const double SQRT3_2 = 0.86602540378443864676;
    double is_inv = inverse ? -1.0 : 1.0;
    __m512d v_half = _mm512_set1_pd(0.5), v_s3 = _mm512_set1_pd(SQRT3_2), v_is_inv = _mm512_set1_pd(is_inv);

    for (size_t j = 0; j < m; ++j) {
        size_t tw_idx = j * stride * tw_step;
        double c1 = ctx.w1_re[tw_idx], s1 = inverse ? -ctx.w1_im[tw_idx] : ctx.w1_im[tw_idx];
        double c2 = ctx.w2_re[tw_idx], s2 = inverse ? -ctx.w2_im[tw_idx] : ctx.w2_im[tw_idx];

        __m512d vc1 = _mm512_set1_pd(c1), vs1 = _mm512_set1_pd(s1);
        __m512d vc2 = _mm512_set1_pd(c2), vs2 = _mm512_set1_pd(s2);

        for (size_t k = 0; k < stride; ) {
            
            if (stride - k >= 16) {
                size_t I0_A = k + j * remaining, I1_A = I0_A + stride, I2_A = I1_A + stride;
                size_t I0_B = I0_A + 8, I1_B = I1_A + 8, I2_B = I2_A + 8;

                __m512d r0_A = _mm512_loadu_pd(&in_re[I0_A]), i0_A = _mm512_loadu_pd(&in_im[I0_A]);
                __m512d r0_B = _mm512_loadu_pd(&in_re[I0_B]), i0_B = _mm512_loadu_pd(&in_im[I0_B]);
                __m512d x1_re_A = _mm512_loadu_pd(&in_re[I1_A]), x1_im_A = _mm512_loadu_pd(&in_im[I1_A]);
                __m512d x1_re_B = _mm512_loadu_pd(&in_re[I1_B]), x1_im_B = _mm512_loadu_pd(&in_im[I1_B]);
                __m512d x2_re_A = _mm512_loadu_pd(&in_re[I2_A]), x2_im_A = _mm512_loadu_pd(&in_im[I2_A]);
                __m512d x2_re_B = _mm512_loadu_pd(&in_re[I2_B]), x2_im_B = _mm512_loadu_pd(&in_im[I2_B]);

                __m512d t1_re_A = _mm512_fmsub_pd(x1_re_A, vc1, _mm512_mul_pd(x1_im_A, vs1));
                __m512d t1_re_B = _mm512_fmsub_pd(x1_re_B, vc1, _mm512_mul_pd(x1_im_B, vs1));
                __m512d t1_im_A = _mm512_fmadd_pd(x1_re_A, vs1, _mm512_mul_pd(x1_im_A, vc1));
                __m512d t1_im_B = _mm512_fmadd_pd(x1_re_B, vs1, _mm512_mul_pd(x1_im_B, vc1));

                __m512d t2_re_A = _mm512_fmsub_pd(x2_re_A, vc2, _mm512_mul_pd(x2_im_A, vs2));
                __m512d t2_re_B = _mm512_fmsub_pd(x2_re_B, vc2, _mm512_mul_pd(x2_im_B, vs2));
                __m512d t2_im_A = _mm512_fmadd_pd(x2_re_A, vs2, _mm512_mul_pd(x2_im_A, vc2));
                __m512d t2_im_B = _mm512_fmadd_pd(x2_re_B, vs2, _mm512_mul_pd(x2_im_B, vc2));

                __m512d ta_re_A = _mm512_add_pd(t1_re_A, t2_re_A), ta_im_A = _mm512_add_pd(t1_im_A, t2_im_A);
                __m512d ta_re_B = _mm512_add_pd(t1_re_B, t2_re_B), ta_im_B = _mm512_add_pd(t1_im_B, t2_im_B);
                __m512d tb_re_A = _mm512_sub_pd(t1_re_A, t2_re_A), tb_im_A = _mm512_sub_pd(t1_im_A, t2_im_A);
                __m512d tb_re_B = _mm512_sub_pd(t1_re_B, t2_re_B), tb_im_B = _mm512_sub_pd(t1_im_B, t2_im_B);

                size_t O0_A = k + j * stride, O1_A = O0_A + m * stride, O2_A = O1_A + m * stride;
                size_t O0_B = O0_A + 8, O1_B = O1_A + 8, O2_B = O2_A + 8;

                _mm512_storeu_pd(&out_re[O0_A], _mm512_add_pd(r0_A, ta_re_A));
                _mm512_storeu_pd(&out_im[O0_A], _mm512_add_pd(i0_A, ta_im_A));
                _mm512_storeu_pd(&out_re[O0_B], _mm512_add_pd(r0_B, ta_re_B));
                _mm512_storeu_pd(&out_im[O0_B], _mm512_add_pd(i0_B, ta_im_B));

                __m512d half_ta_re_A = _mm512_mul_pd(v_half, ta_re_A), half_ta_im_A = _mm512_mul_pd(v_half, ta_im_A);
                __m512d half_ta_re_B = _mm512_mul_pd(v_half, ta_re_B), half_ta_im_B = _mm512_mul_pd(v_half, ta_im_B);

                __m512d M_re_A = _mm512_mul_pd(v_is_inv, _mm512_mul_pd(v_s3, tb_re_A));
                __m512d M_re_B = _mm512_mul_pd(v_is_inv, _mm512_mul_pd(v_s3, tb_re_B));
                __m512d M_im_A = _mm512_mul_pd(v_is_inv, _mm512_mul_pd(v_s3, tb_im_A));
                __m512d M_im_B = _mm512_mul_pd(v_is_inv, _mm512_mul_pd(v_s3, tb_im_B));

                _mm512_storeu_pd(&out_re[O1_A], _mm512_add_pd(_mm512_sub_pd(r0_A, half_ta_re_A), M_im_A));
                _mm512_storeu_pd(&out_im[O1_A], _mm512_sub_pd(_mm512_sub_pd(i0_A, half_ta_im_A), M_re_A));
                _mm512_storeu_pd(&out_re[O1_B], _mm512_add_pd(_mm512_sub_pd(r0_B, half_ta_re_B), M_im_B));
                _mm512_storeu_pd(&out_im[O1_B], _mm512_sub_pd(_mm512_sub_pd(i0_B, half_ta_im_B), M_re_B));

                _mm512_storeu_pd(&out_re[O2_A], _mm512_sub_pd(_mm512_sub_pd(r0_A, half_ta_re_A), M_im_A));
                _mm512_storeu_pd(&out_im[O2_A], _mm512_add_pd(_mm512_sub_pd(i0_A, half_ta_im_A), M_re_A));
                _mm512_storeu_pd(&out_re[O2_B], _mm512_sub_pd(_mm512_sub_pd(r0_B, half_ta_re_B), M_im_B));
                _mm512_storeu_pd(&out_im[O2_B], _mm512_add_pd(_mm512_sub_pd(i0_B, half_ta_im_B), M_re_B));
                k += 16;
            }
            else if (stride - k >= 8) {
                // FALLBACK 1 vector
                size_t I0 = k + j * remaining, I1 = I0 + stride, I2 = I1 + stride;
                __m512d r0 = _mm512_loadu_pd(&in_re[I0]), i0 = _mm512_loadu_pd(&in_im[I0]);
                __m512d x1_re = _mm512_loadu_pd(&in_re[I1]), x1_im = _mm512_loadu_pd(&in_im[I1]);
                __m512d x2_re = _mm512_loadu_pd(&in_re[I2]), x2_im = _mm512_loadu_pd(&in_im[I2]);

                __m512d t1_re = _mm512_fmsub_pd(x1_re, vc1, _mm512_mul_pd(x1_im, vs1));
                __m512d t1_im = _mm512_fmadd_pd(x1_re, vs1, _mm512_mul_pd(x1_im, vc1));
                __m512d t2_re = _mm512_fmsub_pd(x2_re, vc2, _mm512_mul_pd(x2_im, vs2));
                __m512d t2_im = _mm512_fmadd_pd(x2_re, vs2, _mm512_mul_pd(x2_im, vc2));

                __m512d ta_re = _mm512_add_pd(t1_re, t2_re), ta_im = _mm512_add_pd(t1_im, t2_im);
                __m512d tb_re = _mm512_sub_pd(t1_re, t2_re), tb_im = _mm512_sub_pd(t1_im, t2_im);

                size_t O0 = k + j * stride, O1 = O0 + m * stride, O2 = O1 + m * stride;
                _mm512_storeu_pd(&out_re[O0], _mm512_add_pd(r0, ta_re));
                _mm512_storeu_pd(&out_im[O0], _mm512_add_pd(i0, ta_im));

                __m512d half_ta_re = _mm512_mul_pd(v_half, ta_re), half_ta_im = _mm512_mul_pd(v_half, ta_im);
                __m512d M_re = _mm512_mul_pd(v_is_inv, _mm512_mul_pd(v_s3, tb_re));
                __m512d M_im = _mm512_mul_pd(v_is_inv, _mm512_mul_pd(v_s3, tb_im));

                _mm512_storeu_pd(&out_re[O1], _mm512_add_pd(_mm512_sub_pd(r0, half_ta_re), M_im));
                _mm512_storeu_pd(&out_im[O1], _mm512_sub_pd(_mm512_sub_pd(i0, half_ta_im), M_re));
                _mm512_storeu_pd(&out_re[O2], _mm512_sub_pd(_mm512_sub_pd(r0, half_ta_re), M_im));
                _mm512_storeu_pd(&out_im[O2], _mm512_add_pd(_mm512_sub_pd(i0, half_ta_im), M_re));
                k += 8;
            }
            else {
                // FALLBACK scalar
                size_t I0 = k + j * remaining, I1 = I0 + stride, I2 = I1 + stride;
                size_t O0 = k + j * stride, O1 = O0 + m * stride, O2 = O1 + m * stride;
                double r0 = in_re[I0], i0 = in_im[I0], x1_re = in_re[I1], x1_im = in_im[I1], x2_re = in_re[I2], x2_im = in_im[I2];
                double t1_re = x1_re * c1 - x1_im * s1, t1_im = x1_re * s1 + x1_im * c1;
                double t2_re = x2_re * c2 - x2_im * s2, t2_im = x2_re * s2 + x2_im * c2;
                double ta_re = t1_re + t2_re, ta_im = t1_im + t2_im, tb_re = t1_re - t2_re, tb_im = t1_im - t2_im;
                out_re[O0] = r0 + ta_re; out_im[O0] = i0 + ta_im;
                double half_ta_re = 0.5 * ta_re, half_ta_im = 0.5 * ta_im;
                double M_re = is_inv * SQRT3_2 * tb_re, M_im = is_inv * SQRT3_2 * tb_im;
                out_re[O1] = r0 - half_ta_re + M_im; out_im[O1] = i0 - half_ta_im - M_re;
                out_re[O2] = r0 - half_ta_re - M_im; out_im[O2] = i0 - half_ta_im + M_re;
                k++;
            }
        }
    }
}

inline void pass_radix5_seq(const double* in_re, const double* in_im, double* out_re, double* out_im,
    size_t n, size_t m, const FFTContext& ctx, bool inverse, size_t tw_step) {
    size_t remaining = n / m; size_t stride = remaining / 5;
    const double C72 = 0.30901699437494742410, S72 = 0.95105651629515357211;
    const double C144 = -0.80901699437494742410, S144 = 0.58778525229247312916;
    double is_inv = inverse ? -1.0 : 1.0;
    __m512d v_c72 = _mm512_set1_pd(C72), v_s72 = _mm512_set1_pd(S72);
    __m512d v_c144 = _mm512_set1_pd(C144), v_s144 = _mm512_set1_pd(S144);
    __m512d v_is_inv = _mm512_set1_pd(is_inv);

    for (size_t j = 0; j < m; ++j) {
        size_t tw_idx = j * stride * tw_step;
        double c1 = ctx.w1_re[tw_idx], s1 = inverse ? -ctx.w1_im[tw_idx] : ctx.w1_im[tw_idx];
        double c2 = ctx.w2_re[tw_idx], s2 = inverse ? -ctx.w2_im[tw_idx] : ctx.w2_im[tw_idx];
        double c3 = ctx.w3_re[tw_idx], s3 = inverse ? -ctx.w3_im[tw_idx] : ctx.w3_im[tw_idx];
        size_t tw4_idx = 4 * tw_idx;
        double c4 = ctx.w1_re[tw4_idx], s4 = inverse ? -ctx.w1_im[tw4_idx] : ctx.w1_im[tw4_idx];

        __m512d vc1 = _mm512_set1_pd(c1), vs1 = _mm512_set1_pd(s1);
        __m512d vc2 = _mm512_set1_pd(c2), vs2 = _mm512_set1_pd(s2);
        __m512d vc3 = _mm512_set1_pd(c3), vs3 = _mm512_set1_pd(s3);
        __m512d vc4 = _mm512_set1_pd(c4), vs4 = _mm512_set1_pd(s4);

        for (size_t k = 0; k < stride; ) {
            // UNROLLx2 (16 elements)
            if (stride - k >= 16) {
                size_t I0_A = k + j * remaining, I1_A = I0_A + stride, I2_A = I1_A + stride, I3_A = I2_A + stride, I4_A = I3_A + stride;
                size_t I0_B = I0_A + 8, I1_B = I1_A + 8, I2_B = I2_A + 8, I3_B = I3_A + 8, I4_B = I4_A + 8;

                __m512d v0_re_A = _mm512_loadu_pd(&in_re[I0_A]), v0_im_A = _mm512_loadu_pd(&in_im[I0_A]);
                __m512d v0_re_B = _mm512_loadu_pd(&in_re[I0_B]), v0_im_B = _mm512_loadu_pd(&in_im[I0_B]);
                __m512d x1_r_A = _mm512_loadu_pd(&in_re[I1_A]), x1_i_A = _mm512_loadu_pd(&in_im[I1_A]);
                __m512d x1_r_B = _mm512_loadu_pd(&in_re[I1_B]), x1_i_B = _mm512_loadu_pd(&in_im[I1_B]);
                __m512d x2_r_A = _mm512_loadu_pd(&in_re[I2_A]), x2_i_A = _mm512_loadu_pd(&in_im[I2_A]);
                __m512d x2_r_B = _mm512_loadu_pd(&in_re[I2_B]), x2_i_B = _mm512_loadu_pd(&in_im[I2_B]);
                __m512d x3_r_A = _mm512_loadu_pd(&in_re[I3_A]), x3_i_A = _mm512_loadu_pd(&in_im[I3_A]);
                __m512d x3_r_B = _mm512_loadu_pd(&in_re[I3_B]), x3_i_B = _mm512_loadu_pd(&in_im[I3_B]);
                __m512d x4_r_A = _mm512_loadu_pd(&in_re[I4_A]), x4_i_A = _mm512_loadu_pd(&in_im[I4_A]);
                __m512d x4_r_B = _mm512_loadu_pd(&in_re[I4_B]), x4_i_B = _mm512_loadu_pd(&in_im[I4_B]);

                __m512d t1_re_A = _mm512_fmsub_pd(x1_r_A, vc1, _mm512_mul_pd(x1_i_A, vs1));
                __m512d t1_re_B = _mm512_fmsub_pd(x1_r_B, vc1, _mm512_mul_pd(x1_i_B, vs1));
                __m512d t1_im_A = _mm512_fmadd_pd(x1_r_A, vs1, _mm512_mul_pd(x1_i_A, vc1));
                __m512d t1_im_B = _mm512_fmadd_pd(x1_r_B, vs1, _mm512_mul_pd(x1_i_B, vc1));
                __m512d t2_re_A = _mm512_fmsub_pd(x2_r_A, vc2, _mm512_mul_pd(x2_i_A, vs2));
                __m512d t2_re_B = _mm512_fmsub_pd(x2_r_B, vc2, _mm512_mul_pd(x2_i_B, vs2));
                __m512d t2_im_A = _mm512_fmadd_pd(x2_r_A, vs2, _mm512_mul_pd(x2_i_A, vc2));
                __m512d t2_im_B = _mm512_fmadd_pd(x2_r_B, vs2, _mm512_mul_pd(x2_i_B, vc2));
                __m512d t3_re_A = _mm512_fmsub_pd(x3_r_A, vc3, _mm512_mul_pd(x3_i_A, vs3));
                __m512d t3_re_B = _mm512_fmsub_pd(x3_r_B, vc3, _mm512_mul_pd(x3_i_B, vs3));
                __m512d t3_im_A = _mm512_fmadd_pd(x3_r_A, vs3, _mm512_mul_pd(x3_i_A, vc3));
                __m512d t3_im_B = _mm512_fmadd_pd(x3_r_B, vs3, _mm512_mul_pd(x3_i_B, vc3));
                __m512d t4_re_A = _mm512_fmsub_pd(x4_r_A, vc4, _mm512_mul_pd(x4_i_A, vs4));
                __m512d t4_re_B = _mm512_fmsub_pd(x4_r_B, vc4, _mm512_mul_pd(x4_i_B, vs4));
                __m512d t4_im_A = _mm512_fmadd_pd(x4_r_A, vs4, _mm512_mul_pd(x4_i_A, vc4));
                __m512d t4_im_B = _mm512_fmadd_pd(x4_r_B, vs4, _mm512_mul_pd(x4_i_B, vc4));

                __m512d ta_re_A = _mm512_add_pd(t1_re_A, t4_re_A), ta_im_A = _mm512_add_pd(t1_im_A, t4_im_A);
                __m512d ta_re_B = _mm512_add_pd(t1_re_B, t4_re_B), ta_im_B = _mm512_add_pd(t1_im_B, t4_im_B);
                __m512d tb_re_A = _mm512_sub_pd(t1_re_A, t4_re_A), tb_im_A = _mm512_sub_pd(t1_im_A, t4_im_A);
                __m512d tb_re_B = _mm512_sub_pd(t1_re_B, t4_re_B), tb_im_B = _mm512_sub_pd(t1_im_B, t4_im_B);
                __m512d tc_re_A = _mm512_add_pd(t2_re_A, t3_re_A), tc_im_A = _mm512_add_pd(t2_im_A, t3_im_A);
                __m512d tc_re_B = _mm512_add_pd(t2_re_B, t3_re_B), tc_im_B = _mm512_add_pd(t2_im_B, t3_im_B);
                __m512d td_re_A = _mm512_sub_pd(t2_re_A, t3_re_A), td_im_A = _mm512_sub_pd(t2_im_A, t3_im_A);
                __m512d td_re_B = _mm512_sub_pd(t2_re_B, t3_re_B), td_im_B = _mm512_sub_pd(t2_im_B, t3_im_B);

                size_t O0_A = k + j * stride;
                size_t O1_A = O0_A + m * stride, O2_A = O1_A + m * stride, O3_A = O2_A + m * stride, O4_A = O3_A + m * stride;
                size_t O0_B = O0_A + 8, O1_B = O1_A + 8, O2_B = O2_A + 8, O3_B = O3_A + 8, O4_B = O4_A + 8;

                _mm512_storeu_pd(&out_re[O0_A], _mm512_add_pd(v0_re_A, _mm512_add_pd(ta_re_A, tc_re_A)));
                _mm512_storeu_pd(&out_im[O0_A], _mm512_add_pd(v0_im_A, _mm512_add_pd(ta_im_A, tc_im_A)));
                _mm512_storeu_pd(&out_re[O0_B], _mm512_add_pd(v0_re_B, _mm512_add_pd(ta_re_B, tc_re_B)));
                _mm512_storeu_pd(&out_im[O0_B], _mm512_add_pd(v0_im_B, _mm512_add_pd(ta_im_B, tc_im_B)));

                __m512d xr1_A = _mm512_add_pd(v0_re_A, _mm512_add_pd(_mm512_mul_pd(v_c72, ta_re_A), _mm512_mul_pd(v_c144, tc_re_A)));
                __m512d xr1_B = _mm512_add_pd(v0_re_B, _mm512_add_pd(_mm512_mul_pd(v_c72, ta_re_B), _mm512_mul_pd(v_c144, tc_re_B)));
                __m512d xi1_A = _mm512_add_pd(v0_im_A, _mm512_add_pd(_mm512_mul_pd(v_c72, ta_im_A), _mm512_mul_pd(v_c144, tc_im_A)));
                __m512d xi1_B = _mm512_add_pd(v0_im_B, _mm512_add_pd(_mm512_mul_pd(v_c72, ta_im_B), _mm512_mul_pd(v_c144, tc_im_B)));
                __m512d yr1_A = _mm512_mul_pd(v_is_inv, _mm512_add_pd(_mm512_mul_pd(v_s72, tb_im_A), _mm512_mul_pd(v_s144, td_im_A)));
                __m512d yr1_B = _mm512_mul_pd(v_is_inv, _mm512_add_pd(_mm512_mul_pd(v_s72, tb_im_B), _mm512_mul_pd(v_s144, td_im_B)));
                __m512d yi1_A = _mm512_mul_pd(v_is_inv, _mm512_add_pd(_mm512_mul_pd(v_s72, tb_re_A), _mm512_mul_pd(v_s144, td_re_A)));
                __m512d yi1_B = _mm512_mul_pd(v_is_inv, _mm512_add_pd(_mm512_mul_pd(v_s72, tb_re_B), _mm512_mul_pd(v_s144, td_re_B)));

                __m512d xr2_A = _mm512_add_pd(v0_re_A, _mm512_add_pd(_mm512_mul_pd(v_c144, ta_re_A), _mm512_mul_pd(v_c72, tc_re_A)));
                __m512d xr2_B = _mm512_add_pd(v0_re_B, _mm512_add_pd(_mm512_mul_pd(v_c144, ta_re_B), _mm512_mul_pd(v_c72, tc_re_B)));
                __m512d xi2_A = _mm512_add_pd(v0_im_A, _mm512_add_pd(_mm512_mul_pd(v_c144, ta_im_A), _mm512_mul_pd(v_c72, tc_im_A)));
                __m512d xi2_B = _mm512_add_pd(v0_im_B, _mm512_add_pd(_mm512_mul_pd(v_c144, ta_im_B), _mm512_mul_pd(v_c72, tc_im_B)));
                __m512d yr2_A = _mm512_mul_pd(v_is_inv, _mm512_sub_pd(_mm512_mul_pd(v_s144, tb_im_A), _mm512_mul_pd(v_s72, td_im_A)));
                __m512d yr2_B = _mm512_mul_pd(v_is_inv, _mm512_sub_pd(_mm512_mul_pd(v_s144, tb_im_B), _mm512_mul_pd(v_s72, td_im_B)));
                __m512d yi2_A = _mm512_mul_pd(v_is_inv, _mm512_sub_pd(_mm512_mul_pd(v_s144, tb_re_A), _mm512_mul_pd(v_s72, td_re_A)));
                __m512d yi2_B = _mm512_mul_pd(v_is_inv, _mm512_sub_pd(_mm512_mul_pd(v_s144, tb_re_B), _mm512_mul_pd(v_s72, td_re_B)));

                _mm512_storeu_pd(&out_re[O1_A], _mm512_add_pd(xr1_A, yr1_A)); _mm512_storeu_pd(&out_im[O1_A], _mm512_sub_pd(xi1_A, yi1_A));
                _mm512_storeu_pd(&out_re[O1_B], _mm512_add_pd(xr1_B, yr1_B)); _mm512_storeu_pd(&out_im[O1_B], _mm512_sub_pd(xi1_B, yi1_B));
                _mm512_storeu_pd(&out_re[O2_A], _mm512_add_pd(xr2_A, yr2_A)); _mm512_storeu_pd(&out_im[O2_A], _mm512_sub_pd(xi2_A, yi2_A));
                _mm512_storeu_pd(&out_re[O2_B], _mm512_add_pd(xr2_B, yr2_B)); _mm512_storeu_pd(&out_im[O2_B], _mm512_sub_pd(xi2_B, yi2_B));
                _mm512_storeu_pd(&out_re[O3_A], _mm512_sub_pd(xr2_A, yr2_A)); _mm512_storeu_pd(&out_im[O3_A], _mm512_add_pd(xi2_A, yi2_A));
                _mm512_storeu_pd(&out_re[O3_B], _mm512_sub_pd(xr2_B, yr2_B)); _mm512_storeu_pd(&out_im[O3_B], _mm512_add_pd(xi2_B, yi2_B));
                _mm512_storeu_pd(&out_re[O4_A], _mm512_sub_pd(xr1_A, yr1_A)); _mm512_storeu_pd(&out_im[O4_A], _mm512_add_pd(xi1_A, yi1_A));
                _mm512_storeu_pd(&out_re[O4_B], _mm512_sub_pd(xr1_B, yr1_B)); _mm512_storeu_pd(&out_im[O4_B], _mm512_add_pd(xi1_B, yi1_B));
                k += 16;
            }
            else if (stride - k >= 8) {
                // FALLBACK 1 vector
                size_t I0 = k + j * remaining, I1 = I0 + stride, I2 = I1 + stride, I3 = I2 + stride, I4 = I3 + stride;
                __m512d v0_re = _mm512_loadu_pd(&in_re[I0]), v0_im = _mm512_loadu_pd(&in_im[I0]);
                __m512d x1_r = _mm512_loadu_pd(&in_re[I1]), x1_i = _mm512_loadu_pd(&in_im[I1]);
                __m512d x2_r = _mm512_loadu_pd(&in_re[I2]), x2_i = _mm512_loadu_pd(&in_im[I2]);
                __m512d x3_r = _mm512_loadu_pd(&in_re[I3]), x3_i = _mm512_loadu_pd(&in_im[I3]);
                __m512d x4_r = _mm512_loadu_pd(&in_re[I4]), x4_i = _mm512_loadu_pd(&in_im[I4]);

                __m512d t1_re = _mm512_fmsub_pd(x1_r, vc1, _mm512_mul_pd(x1_i, vs1));
                __m512d t1_im = _mm512_fmadd_pd(x1_r, vs1, _mm512_mul_pd(x1_i, vc1));
                __m512d t2_re = _mm512_fmsub_pd(x2_r, vc2, _mm512_mul_pd(x2_i, vs2));
                __m512d t2_im = _mm512_fmadd_pd(x2_r, vs2, _mm512_mul_pd(x2_i, vc2));
                __m512d t3_re = _mm512_fmsub_pd(x3_r, vc3, _mm512_mul_pd(x3_i, vs3));
                __m512d t3_im = _mm512_fmadd_pd(x3_r, vs3, _mm512_mul_pd(x3_i, vc3));
                __m512d t4_re = _mm512_fmsub_pd(x4_r, vc4, _mm512_mul_pd(x4_i, vs4));
                __m512d t4_im = _mm512_fmadd_pd(x4_r, vs4, _mm512_mul_pd(x4_i, vc4));

                __m512d ta_re = _mm512_add_pd(t1_re, t4_re), ta_im = _mm512_add_pd(t1_im, t4_im);
                __m512d tb_re = _mm512_sub_pd(t1_re, t4_re), tb_im = _mm512_sub_pd(t1_im, t4_im);
                __m512d tc_re = _mm512_add_pd(t2_re, t3_re), tc_im = _mm512_add_pd(t2_im, t3_im);
                __m512d td_re = _mm512_sub_pd(t2_re, t3_re), td_im = _mm512_sub_pd(t2_im, t3_im);

                size_t O0 = k + j * stride;
                size_t O1 = O0 + m * stride, O2 = O1 + m * stride, O3 = O2 + m * stride, O4 = O3 + m * stride;

                _mm512_storeu_pd(&out_re[O0], _mm512_add_pd(v0_re, _mm512_add_pd(ta_re, tc_re)));
                _mm512_storeu_pd(&out_im[O0], _mm512_add_pd(v0_im, _mm512_add_pd(ta_im, tc_im)));

                __m512d xr1 = _mm512_add_pd(v0_re, _mm512_add_pd(_mm512_mul_pd(v_c72, ta_re), _mm512_mul_pd(v_c144, tc_re)));
                __m512d xi1 = _mm512_add_pd(v0_im, _mm512_add_pd(_mm512_mul_pd(v_c72, ta_im), _mm512_mul_pd(v_c144, tc_im)));
                __m512d yr1 = _mm512_mul_pd(v_is_inv, _mm512_add_pd(_mm512_mul_pd(v_s72, tb_im), _mm512_mul_pd(v_s144, td_im)));
                __m512d yi1 = _mm512_mul_pd(v_is_inv, _mm512_add_pd(_mm512_mul_pd(v_s72, tb_re), _mm512_mul_pd(v_s144, td_re)));

                __m512d xr2 = _mm512_add_pd(v0_re, _mm512_add_pd(_mm512_mul_pd(v_c144, ta_re), _mm512_mul_pd(v_c72, tc_re)));
                __m512d xi2 = _mm512_add_pd(v0_im, _mm512_add_pd(_mm512_mul_pd(v_c144, ta_im), _mm512_mul_pd(v_c72, tc_im)));
                __m512d yr2 = _mm512_mul_pd(v_is_inv, _mm512_sub_pd(_mm512_mul_pd(v_s144, tb_im), _mm512_mul_pd(v_s72, td_im)));
                __m512d yi2 = _mm512_mul_pd(v_is_inv, _mm512_sub_pd(_mm512_mul_pd(v_s144, tb_re), _mm512_mul_pd(v_s72, td_re)));

                _mm512_storeu_pd(&out_re[O1], _mm512_add_pd(xr1, yr1)); _mm512_storeu_pd(&out_im[O1], _mm512_sub_pd(xi1, yi1));
                _mm512_storeu_pd(&out_re[O2], _mm512_add_pd(xr2, yr2)); _mm512_storeu_pd(&out_im[O2], _mm512_sub_pd(xi2, yi2));
                _mm512_storeu_pd(&out_re[O3], _mm512_sub_pd(xr2, yr2)); _mm512_storeu_pd(&out_im[O3], _mm512_add_pd(xi2, yi2));
                _mm512_storeu_pd(&out_re[O4], _mm512_sub_pd(xr1, yr1)); _mm512_storeu_pd(&out_im[O4], _mm512_add_pd(xi1, yi1));
                k += 8;
            }
            else {
                // FALLBACK scalar
                size_t I0 = k + j * remaining, I1 = I0 + stride, I2 = I1 + stride, I3 = I2 + stride, I4 = I3 + stride;
                size_t O0 = k + j * stride, O1 = O0 + m * stride, O2 = O1 + m * stride, O3 = O2 + m * stride, O4 = O3 + m * stride;
                double v0_re = in_re[I0], v0_im = in_im[I0], x1_r = in_re[I1], x1_i = in_im[I1];
                double x2_r = in_re[I2], x2_i = in_im[I2], x3_r = in_re[I3], x3_i = in_im[I3], x4_r = in_re[I4], x4_i = in_im[I4];
                double t1_re = x1_r * c1 - x1_i * s1, t1_im = x1_r * s1 + x1_i * c1;
                double t2_re = x2_r * c2 - x2_i * s2, t2_im = x2_r * s2 + x2_i * c2;
                double t3_re = x3_r * c3 - x3_i * s3, t3_im = x3_r * s3 + x3_i * c3;
                double t4_re = x4_r * c4 - x4_i * s4, t4_im = x4_r * s4 + x4_i * c4;
                double ta_re = t1_re + t4_re, ta_im = t1_im + t4_im, tb_re = t1_re - t4_re, tb_im = t1_im - t4_im;
                double tc_re = t2_re + t3_re, tc_im = t2_im + t3_im, td_re = t2_re - t3_re, td_im = t2_im - t3_im;
                out_re[O0] = v0_re + ta_re + tc_re; out_im[O0] = v0_im + ta_im + tc_im;
                double xr1 = v0_re + C72 * ta_re + C144 * tc_re, xi1 = v0_im + C72 * ta_im + C144 * tc_im;
                double yr1 = is_inv * (S72 * tb_im + S144 * td_im), yi1 = is_inv * (S72 * tb_re + S144 * td_re);
                double xr2 = v0_re + C144 * ta_re + C72 * tc_re, xi2 = v0_im + C144 * ta_im + C72 * tc_im;
                double yr2 = is_inv * (S144 * tb_im - S72 * td_im), yi2 = is_inv * (S144 * tb_re - S72 * td_re);
                out_re[O1] = xr1 + yr1; out_im[O1] = xi1 - yi1;
                out_re[O2] = xr2 + yr2; out_im[O2] = xi2 - yi2;
                out_re[O3] = xr2 - yr2; out_im[O3] = xi2 + yi2;
                out_re[O4] = xr1 - yr1; out_im[O4] = xi1 + yi1;
                k++;
            }
        }
    }
}

inline void fft_mixed_radix_1d_seq(double* in_re, double* in_im, double* out_re, double* out_im,
    size_t local_n, size_t total_n, const FFTFactorPlan& plan, bool inverse, const FFTContext& ctx) {
    double* cur_re = in_re; double* cur_im = in_im;
    double* nxt_re = out_re; double* nxt_im = out_im;
    size_t m = 1;
    size_t tw_step = total_n / local_n;

    for (size_t radix : plan.factors) {
        if (radix == 2) pass_radix2_seq(cur_re, cur_im, nxt_re, nxt_im, local_n, m, ctx, inverse, tw_step);
        else if (radix == 3) pass_radix3_seq(cur_re, cur_im, nxt_re, nxt_im, local_n, m, ctx, inverse, tw_step);
        else if (radix == 5) pass_radix5_seq(cur_re, cur_im, nxt_re, nxt_im, local_n, m, ctx, inverse, tw_step);

        m *= radix;
        std::swap(cur_re, nxt_re);
        std::swap(cur_im, nxt_im);
    }

    
    if (cur_re != in_re) {
        for (size_t i = 0; i < local_n; ++i) {
            in_re[i] = cur_re[i];
            in_im[i] = cur_im[i];
        }
    }
}




void transpose_blocked_complex(const double* __restrict src_re, const double* __restrict src_im, double* __restrict dst_re, double* __restrict dst_im, size_t n1, size_t n2);
void transpose_and_twiddle(const double* __restrict src_re, const double* __restrict src_im, double* __restrict dst_re, double* __restrict dst_im, size_t n2, size_t n1, FFTContext& ctx, bool inverse);

// =================================================================
//  MIXED-RADIX
// =================================================================
inline void fft_mixed_radix_execute(FFTContext& ctx, const FFTFactorPlan& plan, bool inverse) {
    if (plan.has_unsupported_primes) {
        std::cerr << "[ERROR] Unsupported prime factors in plan. Fallback needed.\n"; return;
    }

    size_t N = plan.total_n;
    size_t n1 = 1, n2 = 1;

    
    for (size_t radix : plan.factors) {
        if (n1 <= n2) n1 *= radix;
        else n2 *= radix;
    }
        
    if (n1 == 1 || n2 == 1) {
        fft_mixed_radix_1d_seq(ctx.re, ctx.im, ctx.re2, ctx.im2, N, N, plan, inverse, ctx);
        return;
    }

    FFTFactorPlan plan1 = generate_fft_plan_inline(n1);
    FFTFactorPlan plan2 = generate_fft_plan_inline(n2);
        
    if (!inverse) {
        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2);
        std::swap(ctx.re, ctx.re2); std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; i++) {
            fft_mixed_radix_1d_seq(&ctx.re[i * n1], &ctx.im[i * n1], &ctx.re2[i * n1], &ctx.im2[i * n1], n1, N, plan1, false, ctx);
        }

        transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1, ctx, false);
        std::swap(ctx.re, ctx.re2); std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; i++) {
            fft_mixed_radix_1d_seq(&ctx.re[i * n2], &ctx.im[i * n2], &ctx.re2[i * n2], &ctx.im2[i * n2], n2, N, plan2, false, ctx);
        }

        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2);
        std::swap(ctx.re, ctx.re2); std::swap(ctx.im, ctx.im2);
    }
    else {
        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1);
        std::swap(ctx.re, ctx.re2); std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; i++) {
            fft_mixed_radix_1d_seq(&ctx.re[i * n2], &ctx.im[i * n2], &ctx.re2[i * n2], &ctx.im2[i * n2], n2, N, plan2, true, ctx);
        }

        transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, ctx, true);
        std::swap(ctx.re, ctx.re2); std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; i++) {
            fft_mixed_radix_1d_seq(&ctx.re[i * n1], &ctx.im[i * n1], &ctx.re2[i * n1], &ctx.im2[i * n1], n1, N, plan1, true, ctx);
        }

        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1);
        std::swap(ctx.re, ctx.re2); std::swap(ctx.im, ctx.im2);
    }
}





// =================================================================
// AVX-512 TRANSPOSER (8x8  ZMM, ZERO MEMORY SPILL)
// =================================================================
inline __attribute__((always_inline)) void transpose8x8_pd(
    __m512d& r0, __m512d& r1, __m512d& r2, __m512d& r3,
    __m512d& r4, __m512d& r5, __m512d& r6, __m512d& r7)
{
    // Step 1: shuffle 2x2 blocks (hardware unpack instructions)
    __m512d t0 = _mm512_unpacklo_pd(r0, r1);
    __m512d t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    // Step 2: shuffle 4x4 blocks (precise vector permutes)
    const __m512i idx_u0 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idx_u2 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);

    __m512d u0 = _mm512_permutex2var_pd(t0, idx_u0, t2);
    __m512d u2 = _mm512_permutex2var_pd(t0, idx_u2, t2);
    __m512d u1 = _mm512_permutex2var_pd(t1, idx_u0, t3);
    __m512d u3 = _mm512_permutex2var_pd(t1, idx_u2, t3);
    __m512d u4 = _mm512_permutex2var_pd(t4, idx_u0, t6);
    __m512d u6 = _mm512_permutex2var_pd(t4, idx_u2, t6);
    __m512d u5 = _mm512_permutex2var_pd(t5, idx_u0, t7);
    __m512d u7 = _mm512_permutex2var_pd(t5, idx_u2, t7);

    // Step 3: final 8x8 stitch
    const __m512i idx_r0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    const __m512i idx_r4 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    r0 = _mm512_permutex2var_pd(u0, idx_r0, u4);
    r4 = _mm512_permutex2var_pd(u0, idx_r4, u4);
    r1 = _mm512_permutex2var_pd(u1, idx_r0, u5);
    r5 = _mm512_permutex2var_pd(u1, idx_r4, u5);
    r2 = _mm512_permutex2var_pd(u2, idx_r0, u6);
    r6 = _mm512_permutex2var_pd(u2, idx_r4, u6);
    r3 = _mm512_permutex2var_pd(u3, idx_r0, u7);
    r7 = _mm512_permutex2var_pd(u3, idx_r4, u7);
}


inline __attribute__((always_inline)) void transpose_twiddle_leaf_8x8(
    const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t r1, size_t c1, size_t rows, size_t cols, 
    const FFTContext& ctx, bool inverse) 
{
size_t prefetch_rows_ahead = 4; 

if (r1 + prefetch_rows_ahead < rows) {
    // Prefetch the row 4 positions further down, taking 'cols' into account
    _mm_prefetch((const char*)&src_re[(r1 + prefetch_rows_ahead) * cols + c1], _MM_HINT_T1);
    _mm_prefetch((const char*)&src_im[(r1 + prefetch_rows_ahead) * cols + c1], _MM_HINT_T1);
}
    // 1. Load a block of rows from src memory
    __m512d r0 = _mm512_loadu_pd(&src_re[r1 * cols + c1]);
    __m512d r1_ = _mm512_loadu_pd(&src_re[(r1 + 1) * cols + c1]);
    __m512d r2 = _mm512_loadu_pd(&src_re[(r1 + 2) * cols + c1]);
    __m512d r3 = _mm512_loadu_pd(&src_re[(r1 + 3) * cols + c1]);
    __m512d r4 = _mm512_loadu_pd(&src_re[(r1 + 4) * cols + c1]);
    __m512d r5 = _mm512_loadu_pd(&src_re[(r1 + 5) * cols + c1]);
    __m512d r6 = _mm512_loadu_pd(&src_re[(r1 + 6) * cols + c1]);
    __m512d r7 = _mm512_loadu_pd(&src_re[(r1 + 7) * cols + c1]);

    __m512d i0 = _mm512_loadu_pd(&src_im[r1 * cols + c1]);
    __m512d i1 = _mm512_loadu_pd(&src_im[(r1 + 1) * cols + c1]);
    __m512d i2 = _mm512_loadu_pd(&src_im[(r1 + 2) * cols + c1]);
    __m512d i3 = _mm512_loadu_pd(&src_im[(r1 + 3) * cols + c1]);
    __m512d i4 = _mm512_loadu_pd(&src_im[(r1 + 4) * cols + c1]);
    __m512d i5 = _mm512_loadu_pd(&src_im[(r1 + 5) * cols + c1]);
    __m512d i6 = _mm512_loadu_pd(&src_im[(r1 + 6) * cols + c1]);
    __m512d i7 = _mm512_loadu_pd(&src_im[(r1 + 7) * cols + c1]);

    // 2. Transpose the ZMM registers
    transpose8x8_pd(r0, r1_, r2, r3, r4, r5, r6, r7);
    transpose8x8_pd(i0, i1, i2, i3, i4, i5, i6, i7);

    __m512d sign_mask = _mm512_set1_pd(inverse ? -1.0 : 1.0);
    size_t N = ctx.fft_len;
    // After the transpose, registers r0..r7 become consecutive rows of the destination matrix dst.
    // Each register corresponds to input column `current_col = c1 + j`.
    __m512d r_arr[8] = { r0, r1_, r2, r3, r4, r5, r6, r7 };
    __m512d i_arr[8] = { i0, i1, i2, i3, i4, i5, i6, i7 };

    // Twiddle of element (r1+i, c1+j): w^((r1+i)*(c1+j)) = S_j * U_i * V_ij  (exponents mod N), where
    //   S_j  = w^(r1*(c1+j))  (one table read per column),
    //   U_i  = w^(i*c1)       (eight reads for the whole block),
    //   V_ij = w^(i*j)        (8x8 constant, computed once per thread and N; index i*j <= 49 < N).
    // Instead of 64 random reads from w1 per block there are 16; the rest are vector multiplications.
    alignas(64) static thread_local double V_re[8][8];
    alignas(64) static thread_local double V_im[8][8];
    static thread_local size_t V_N = 0;
    if (V_N != N) {
        for (size_t jj = 0; jj < 8; ++jj) {
            for (size_t ii = 0; ii < 8; ++ii) {
                V_re[jj][ii] = ctx.w1_re[ii * jj];
                V_im[jj][ii] = ctx.w1_im[ii * jj];
            }
        }
        V_N = N;
    }

    alignas(64) double u_re[8];
    alignas(64) double u_im[8];
    for (size_t ii = 0; ii < 8; ++ii) {
        const size_t u_idx = (ii * c1) & (N - 1);
        u_re[ii] = ctx.w1_re[u_idx];
        u_im[ii] = ctx.w1_im[u_idx];
    }
    const __m512d Ur = _mm512_load_pd(u_re);
    const __m512d Ui = _mm512_load_pd(u_im);
    const Tw2 tw2(ctx);

    for (size_t j = 0; j < 8; ++j) {
        size_t current_col = c1 + j;

        const __m512d Vr = _mm512_load_pd(V_re[j]);
        const __m512d Vi = _mm512_load_pd(V_im[j]);
        // P = U * V_j (over i)
        const __m512d Pr = _mm512_fmsub_pd(Ur, Vr, _mm512_mul_pd(Ui, Vi));
        const __m512d Pi = _mm512_fmadd_pd(Ur, Vi, _mm512_mul_pd(Ui, Vr));
        // Q = P * S_j (S_j scalar, broadcast)
        double s_re, s_im;
        tw2.get((r1 * current_col) & (N - 1), s_re, s_im);
        const __m512d Sr = _mm512_set1_pd(s_re);
        const __m512d Si = _mm512_set1_pd(s_im);
        __m512d wr = _mm512_fmsub_pd(Pr, Sr, _mm512_mul_pd(Pi, Si));
        __m512d wi = _mm512_fmadd_pd(Pr, Si, _mm512_mul_pd(Pi, Sr));
        wi = _mm512_mul_pd(wi, sign_mask);

        __m512d A = r_arr[j];
        __m512d B = i_arr[j];

        // Complex multiplication (Karatsuba style)
        __m512d A_plus_B = _mm512_add_pd(A, B);
        __m512d wi_minus_wr = _mm512_sub_pd(wi, wr);
        __m512d wr_plus_wi = _mm512_add_pd(wr, wi);

        __m512d k1 = _mm512_mul_pd(wr, A_plus_B);
        __m512d k2 = _mm512_mul_pd(A, wi_minus_wr);
        __m512d k3 = _mm512_mul_pd(B, wr_plus_wi);

        __m512d out_re = _mm512_sub_pd(k1, k3);
        __m512d out_im = _mm512_add_pd(k1, k2);

        // Stride-1 store into the destination matrix
        _mm512_storeu_pd(&dst_re[current_col * rows + r1], out_re);
        _mm512_storeu_pd(&dst_im[current_col * rows + r1], out_im);
    }
}
inline __attribute__((always_inline)) void transpose_pure_leaf_8x8(
    const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t r1, size_t c1, size_t n1, size_t n2)
{
    __m512d r0 = _mm512_loadu_pd(&src_re[(r1 + 0) * n2 + c1]);
    __m512d r1_ = _mm512_loadu_pd(&src_re[(r1 + 1) * n2 + c1]);
    __m512d r2 = _mm512_loadu_pd(&src_re[(r1 + 2) * n2 + c1]);
    __m512d r3 = _mm512_loadu_pd(&src_re[(r1 + 3) * n2 + c1]);
    __m512d r4 = _mm512_loadu_pd(&src_re[(r1 + 4) * n2 + c1]);
    __m512d r5 = _mm512_loadu_pd(&src_re[(r1 + 5) * n2 + c1]);
    __m512d r6 = _mm512_loadu_pd(&src_re[(r1 + 6) * n2 + c1]);
    __m512d r7 = _mm512_loadu_pd(&src_re[(r1 + 7) * n2 + c1]);

    __m512d i0 = _mm512_loadu_pd(&src_im[(r1 + 0) * n2 + c1]);
    __m512d i1 = _mm512_loadu_pd(&src_im[(r1 + 1) * n2 + c1]);
    __m512d i2 = _mm512_loadu_pd(&src_im[(r1 + 2) * n2 + c1]);
    __m512d i3 = _mm512_loadu_pd(&src_im[(r1 + 3) * n2 + c1]);
    __m512d i4 = _mm512_loadu_pd(&src_im[(r1 + 4) * n2 + c1]);
    __m512d i5 = _mm512_loadu_pd(&src_im[(r1 + 5) * n2 + c1]);
    __m512d i6 = _mm512_loadu_pd(&src_im[(r1 + 6) * n2 + c1]);
    __m512d i7 = _mm512_loadu_pd(&src_im[(r1 + 7) * n2 + c1]);

    transpose8x8_pd(r0, r1_, r2, r3, r4, r5, r6, r7);
    transpose8x8_pd(i0, i1, i2, i3, i4, i5, i6, i7);

    _mm512_storeu_pd(&dst_re[(c1 + 0) * n1 + r1], r0);
    _mm512_storeu_pd(&dst_re[(c1 + 1) * n1 + r1], r1_);
    _mm512_storeu_pd(&dst_re[(c1 + 2) * n1 + r1], r2);
    _mm512_storeu_pd(&dst_re[(c1 + 3) * n1 + r1], r3);
    _mm512_storeu_pd(&dst_re[(c1 + 4) * n1 + r1], r4);
    _mm512_storeu_pd(&dst_re[(c1 + 5) * n1 + r1], r5);
    _mm512_storeu_pd(&dst_re[(c1 + 6) * n1 + r1], r6);
    _mm512_storeu_pd(&dst_re[(c1 + 7) * n1 + r1], r7);

    _mm512_storeu_pd(&dst_im[(c1 + 0) * n1 + r1], i0);
    _mm512_storeu_pd(&dst_im[(c1 + 1) * n1 + r1], i1);
    _mm512_storeu_pd(&dst_im[(c1 + 2) * n1 + r1], i2);
    _mm512_storeu_pd(&dst_im[(c1 + 3) * n1 + r1], i3);
    _mm512_storeu_pd(&dst_im[(c1 + 4) * n1 + r1], i4);
    _mm512_storeu_pd(&dst_im[(c1 + 5) * n1 + r1], i5);
    _mm512_storeu_pd(&dst_im[(c1 + 6) * n1 + r1], i6);
    _mm512_storeu_pd(&dst_im[(c1 + 7) * n1 + r1], i7);
}
// =================================================================
// 1. INVERTED LOOPS (GATHER)
// =================================================================
void transpose_recursive(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t r1, size_t r2, size_t c1, size_t c2,
    size_t n1, size_t n2) {
    size_t r_len = r2 - r1;
    size_t c_len = c2 - c1;

    // Direct AVX-512 call for full 8x8 blocks (ZERO L1 cache spills)
    if (r_len == 8 && c_len == 8) {
        transpose_pure_leaf_8x8(src_re, src_im, dst_re, dst_im, r1, c1, n1, n2);
        return;
    }

    if (r_len <= 8 && c_len <= 8) {
        for (size_t j = c1; j < c2; ++j) {
            for (size_t i = r1; i < r2; ++i) {
                dst_re[j * n1 + i] = src_re[i * n2 + j];
                dst_im[j * n1 + i] = src_im[i * n2 + j];
            }
        }
    }
    else if (r_len >= c_len) {
        size_t r_mid = r1 + r_len / 2;
        transpose_recursive(src_re, src_im, dst_re, dst_im, r1, r_mid, c1, c2, n1, n2);
        transpose_recursive(src_re, src_im, dst_re, dst_im, r_mid, r2, c1, c2, n1, n2);
    }
    else {
        size_t c_mid = c1 + c_len / 2;
        transpose_recursive(src_re, src_im, dst_re, dst_im, r1, r2, c1, c_mid, n1, n2);
        transpose_recursive(src_re, src_im, dst_re, dst_im, r1, r2, c_mid, c2, n1, n2);
    }
}

void transpose_blocked_complex(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t n1, size_t n2) {

    const size_t THREAD_BLOCK = 64;

#pragma omp parallel for schedule(dynamic) collapse(2)
    for (size_t i = 0; i < n1; i += THREAD_BLOCK) {
        for (size_t j = 0; j < n2; j += THREAD_BLOCK) {
            size_t max_i = std::min(i + THREAD_BLOCK, n1);
            size_t max_j = std::min(j + THREAD_BLOCK, n2);
            transpose_recursive(src_re, src_im, dst_re, dst_im, i, max_i, j, max_j, n1, n2);
        }
    }
}

void transpose_twiddle_recursive(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t r1, size_t r2, size_t c1, size_t c2,
    size_t rows, size_t cols, const FFTContext& ctx, bool inverse) {

    size_t r_len = r2 - r1;
    size_t c_len = c2 - c1;

    // Direct AVX-512 call for full 8x8 blocks
    if (r_len == 8 && c_len == 8) {
        transpose_twiddle_leaf_8x8(src_re, src_im, dst_re, dst_im, r1, c1, rows, cols, ctx, inverse);
        return;
    }

    if (r_len <= 8 && c_len <= 8) {
        double sign = inverse ? -1.0 : 1.0;
        size_t N = ctx.fft_len;

        for (size_t j = c1; j < c2; ++j) {
            size_t dst_base = j * rows;
#pragma omp simd
            for (size_t i = r1; i < r2; ++i) {
                double r = src_re[i * cols + j];
                double im = src_im[i * cols + j];

                size_t tw_idx = (i * j) & (N - 1);

                double wr = ctx.w1_re[tw_idx];
                double wi = ctx.w1_im[tw_idx] * sign;

                dst_re[dst_base + i] = r * wr - im * wi;
                dst_im[dst_base + i] = r * wi + im * wr;
            }
        }
        return;
    }

    if (r_len >= c_len) {
        size_t r_mid = r1 + r_len / 2;
        transpose_twiddle_recursive(src_re, src_im, dst_re, dst_im, r1, r_mid, c1, c2, rows, cols, ctx, inverse);
        transpose_twiddle_recursive(src_re, src_im, dst_re, dst_im, r_mid, r2, c1, c2, rows, cols, ctx, inverse);
    }
    else {
        size_t c_mid = c1 + c_len / 2;
        transpose_twiddle_recursive(src_re, src_im, dst_re, dst_im, r1, r2, c1, c_mid, rows, cols, ctx, inverse);
        transpose_twiddle_recursive(src_re, src_im, dst_re, dst_im, r1, r2, c_mid, c2, rows, cols, ctx, inverse);
    }
}

void transpose_and_twiddle(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t n2, size_t n1, FFTContext& ctx, bool inverse) {

    const size_t THREAD_BLOCK = 64;

#pragma omp parallel for schedule(dynamic) collapse(2)
    for (size_t i = 0; i < n2; i += THREAD_BLOCK) {
        for (size_t j = 0; j < n1; j += THREAD_BLOCK) {
            size_t max_i = std::min(i + THREAD_BLOCK, n2);
            size_t max_j = std::min(j + THREAD_BLOCK, n1);
            transpose_twiddle_recursive(src_re, src_im, dst_re, dst_im, i, max_i, j, max_j, n2, n1, ctx, inverse);
        }
    }
}

// ---------------------------------------------------------
// VERSION 1.1: MULTI-THREADED STOCKHAM 1D (DYNAMIC THREADING)
// ---------------------------------------------------------
static inline __attribute__((always_inline)) void st_transpose8x8(__m512d* r) {
    __m512d t0 = _mm512_unpacklo_pd(r[0], r[1]), t1 = _mm512_unpackhi_pd(r[0], r[1]);
    __m512d t2 = _mm512_unpacklo_pd(r[2], r[3]), t3 = _mm512_unpackhi_pd(r[2], r[3]);
    __m512d t4 = _mm512_unpacklo_pd(r[4], r[5]), t5 = _mm512_unpackhi_pd(r[4], r[5]);
    __m512d t6 = _mm512_unpacklo_pd(r[6], r[7]), t7 = _mm512_unpackhi_pd(r[6], r[7]);
    __m512d s0 = _mm512_shuffle_f64x2(t0, t2, 0x88), s1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    __m512d s2 = _mm512_shuffle_f64x2(t0, t2, 0xDD), s3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d s4 = _mm512_shuffle_f64x2(t4, t6, 0x88), s5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    __m512d s6 = _mm512_shuffle_f64x2(t4, t6, 0xDD), s7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    r[0] = _mm512_shuffle_f64x2(s0, s4, 0x88); r[1] = _mm512_shuffle_f64x2(s1, s5, 0x88);
    r[2] = _mm512_shuffle_f64x2(s2, s6, 0x88); r[3] = _mm512_shuffle_f64x2(s3, s7, 0x88);
    r[4] = _mm512_shuffle_f64x2(s0, s4, 0xDD); r[5] = _mm512_shuffle_f64x2(s1, s5, 0xDD);
    r[6] = _mm512_shuffle_f64x2(s2, s6, 0xDD); r[7] = _mm512_shuffle_f64x2(s3, s7, 0xDD);
}

// Radix-8 Stockham for 8 independent blocks at once (one vector lane = one block).
// Same formulas as the main loop of fft_stockham_radix8; input r[q]/i[q], twiddle wr[q]/wi[q] (q=1..7), output yr[q]/yi[q].
static inline __attribute__((always_inline)) void st_radix8_vec(const __m512d* r, const __m512d* i,
    const __m512d* wr, const __m512d* wi, __m512d* yr, __m512d* yi,
    __m512d v_sign, __m512d v_neg_sign, __m512d v_C, __m512d v_neg_C) {
    __m512d tr[8], ti[8];
    tr[0] = r[0]; ti[0] = i[0];
#pragma GCC unroll 8
    for (int q = 1; q < 8; ++q) {
        tr[q] = _mm512_fmsub_pd(r[q], wr[q], _mm512_mul_pd(i[q], wi[q]));
        ti[q] = _mm512_fmadd_pd(r[q], wi[q], _mm512_mul_pd(i[q], wr[q]));
    }
    __m512d u0r = _mm512_add_pd(tr[0], tr[4]), u0i = _mm512_add_pd(ti[0], ti[4]); __m512d u4r = _mm512_sub_pd(tr[0], tr[4]), u4i = _mm512_sub_pd(ti[0], ti[4]);
    __m512d u1r = _mm512_add_pd(tr[1], tr[5]), u1i = _mm512_add_pd(ti[1], ti[5]); __m512d u5r = _mm512_sub_pd(tr[1], tr[5]), u5i = _mm512_sub_pd(ti[1], ti[5]);
    __m512d u2r = _mm512_add_pd(tr[2], tr[6]), u2i = _mm512_add_pd(ti[2], ti[6]); __m512d u6r = _mm512_sub_pd(tr[2], tr[6]), u6i = _mm512_sub_pd(ti[2], ti[6]);
    __m512d u3r = _mm512_add_pd(tr[3], tr[7]), u3i = _mm512_add_pd(ti[3], ti[7]); __m512d u7r = _mm512_sub_pd(tr[3], tr[7]), u7i = _mm512_sub_pd(ti[3], ti[7]);

    __m512d v5r = _mm512_mul_pd(v_C, _mm512_add_pd(u5r, _mm512_mul_pd(u5i, v_sign)));
    __m512d v5i = _mm512_mul_pd(v_C, _mm512_sub_pd(u5i, _mm512_mul_pd(u5r, v_sign)));
    __m512d v6r = _mm512_mul_pd(u6i, v_sign);
    __m512d v6i = _mm512_mul_pd(u6r, v_neg_sign);
    __m512d v7r = _mm512_mul_pd(v_C, _mm512_sub_pd(_mm512_mul_pd(u7i, v_sign), u7r));
    __m512d v7i = _mm512_mul_pd(v_neg_C, _mm512_add_pd(_mm512_mul_pd(u7r, v_sign), u7i));

    __m512d a0r = _mm512_add_pd(u0r, u2r), a0i = _mm512_add_pd(u0i, u2i); __m512d a1r = _mm512_sub_pd(u0r, u2r), a1i = _mm512_sub_pd(u0i, u2i);
    __m512d a2r = _mm512_add_pd(u1r, u3r), a2i = _mm512_add_pd(u1i, u3i); __m512d a3r = _mm512_sub_pd(u1r, u3r), a3i = _mm512_sub_pd(u1i, u3i);
    __m512d d3r = _mm512_mul_pd(a3i, v_sign), d3i = _mm512_mul_pd(a3r, v_neg_sign);

    yr[0] = _mm512_add_pd(a0r, a2r); yi[0] = _mm512_add_pd(a0i, a2i);
    yr[4] = _mm512_sub_pd(a0r, a2r); yi[4] = _mm512_sub_pd(a0i, a2i);
    yr[2] = _mm512_add_pd(a1r, d3r); yi[2] = _mm512_add_pd(a1i, d3i);
    yr[6] = _mm512_sub_pd(a1r, d3r); yi[6] = _mm512_sub_pd(a1i, d3i);

    __m512d b0r = _mm512_add_pd(u4r, v6r), b0i = _mm512_add_pd(u4i, v6i); __m512d b1r = _mm512_sub_pd(u4r, v6r), b1i = _mm512_sub_pd(u4i, v6i);
    __m512d b2r = _mm512_add_pd(v5r, v7r), b2i = _mm512_add_pd(v5i, v7i); __m512d b3r = _mm512_sub_pd(v5r, v7r), b3i = _mm512_sub_pd(v5i, v7i);
    __m512d e3r = _mm512_mul_pd(b3i, v_sign), e3i = _mm512_mul_pd(b3r, v_neg_sign);

    yr[1] = _mm512_add_pd(b0r, b2r); yi[1] = _mm512_add_pd(b0i, b2i);
    yr[5] = _mm512_sub_pd(b0r, b2r); yi[5] = _mm512_sub_pd(b0i, b2i);
    yr[3] = _mm512_add_pd(b1r, e3r); yi[3] = _mm512_add_pd(b1i, e3i);
    yr[7] = _mm512_sub_pd(b1r, e3r); yi[7] = _mm512_sub_pd(b1i, e3i);
}

// The last two Stockham passes (radix-8 at remaining==16, stride==2, then radix-2) in a single sweep.
// Input block j (16 consecutive elements) produces results only for pairs (k=0,k=1) with the same j+q*m, so everything is local.
// The result lands in out_*. Requires m = n/16 divisible by 8.
static void stockham_fused_last16(const double* in_re, const double* in_im, double* out_re, double* out_im,
    const double* w1re, const double* w1im, size_t n, bool inverse) {
    const size_t m = n >> 4;
    const size_t half = n >> 1;
    const double sign = inverse ? -1.0 : 1.0;
    const __m512d v_sign = _mm512_set1_pd(sign);
    const __m512d v_neg_sign = _mm512_set1_pd(-sign);
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    const __m512d v_neg_C = _mm512_set1_pd(-0.70710678118654752440);
    const __m512d v_isg = _mm512_set1_pd(inverse ? -1.0 : 1.0);   // sign of the twiddle imaginary part
    const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

#pragma omp parallel for schedule(static)
    for (size_t g = 0; g < m / 8; ++g) {
        const size_t j0 = g * 8;
        __m512d R[16], I[16];
        for (int l = 0; l < 8; ++l) {
            R[l]     = _mm512_load_pd(in_re + 16 * (j0 + l));
            R[8 + l] = _mm512_load_pd(in_re + 16 * (j0 + l) + 8);
            I[l]     = _mm512_load_pd(in_im + 16 * (j0 + l));
            I[8 + l] = _mm512_load_pd(in_im + 16 * (j0 + l) + 8);
        }
        st_transpose8x8(R); st_transpose8x8(R + 8);
        st_transpose8x8(I); st_transpose8x8(I + 8);   // now R[c]/I[c] = element c of blocks j0..j0+7 (c = k + 2q)

        __m512i idx1 = _mm512_slli_epi64(_mm512_add_epi64(_mm512_set1_epi64((long long)j0), lane), 1);   // tw = 2*j
        __m512i idx2 = _mm512_slli_epi64(idx1, 1), idx4 = _mm512_slli_epi64(idx2, 1);
        __m512d wr[8], wi[8];
        wr[1] = _mm512_i64gather_pd(idx1, w1re, 8); wi[1] = _mm512_mul_pd(_mm512_i64gather_pd(idx1, w1im, 8), v_isg);
        wr[2] = _mm512_i64gather_pd(idx2, w1re, 8); wi[2] = _mm512_mul_pd(_mm512_i64gather_pd(idx2, w1im, 8), v_isg);
        wr[4] = _mm512_i64gather_pd(idx4, w1re, 8); wi[4] = _mm512_mul_pd(_mm512_i64gather_pd(idx4, w1im, 8), v_isg);
        // w3 = w2*w1, w5 = w4*w1, w6 = w4*w2, w7 = w6*w1
        wr[3] = _mm512_fmsub_pd(wr[2], wr[1], _mm512_mul_pd(wi[2], wi[1])); wi[3] = _mm512_fmadd_pd(wr[2], wi[1], _mm512_mul_pd(wi[2], wr[1]));
        wr[5] = _mm512_fmsub_pd(wr[4], wr[1], _mm512_mul_pd(wi[4], wi[1])); wi[5] = _mm512_fmadd_pd(wr[4], wi[1], _mm512_mul_pd(wi[4], wr[1]));
        wr[6] = _mm512_fmsub_pd(wr[4], wr[2], _mm512_mul_pd(wi[4], wi[2])); wi[6] = _mm512_fmadd_pd(wr[4], wi[2], _mm512_mul_pd(wi[4], wr[2]));
        wr[7] = _mm512_fmsub_pd(wr[6], wr[1], _mm512_mul_pd(wi[6], wi[1])); wi[7] = _mm512_fmadd_pd(wr[6], wi[1], _mm512_mul_pd(wi[6], wr[1]));

        __m512d Yr[2][8], Yi[2][8];
        for (int k = 0; k < 2; ++k) {
            __m512d r[8], i[8];
#pragma GCC unroll 8
            for (int q = 0; q < 8; ++q) { r[q] = R[k + 2 * q]; i[q] = I[k + 2 * q]; }
            st_radix8_vec(r, i, wr, wi, Yr[k], Yi[k], v_sign, v_neg_sign, v_C, v_neg_C);
        }
#pragma GCC unroll 8
        for (int q = 0; q < 8; ++q) {
            const size_t jp = j0 + (size_t)q * m;
            __m512d tr = _mm512_loadu_pd(w1re + jp);
            __m512d ti = _mm512_mul_pd(_mm512_loadu_pd(w1im + jp), v_isg);
            __m512d xr = _mm512_fmsub_pd(Yr[1][q], tr, _mm512_mul_pd(Yi[1][q], ti));
            __m512d xi = _mm512_fmadd_pd(Yr[1][q], ti, _mm512_mul_pd(Yi[1][q], tr));
            _mm512_store_pd(out_re + jp, _mm512_add_pd(Yr[0][q], xr));
            _mm512_store_pd(out_im + jp, _mm512_add_pd(Yi[0][q], xi));
            _mm512_store_pd(out_re + jp + half, _mm512_sub_pd(Yr[0][q], xr));
            _mm512_store_pd(out_im + jp + half, _mm512_sub_pd(Yi[0][q], xi));
        }
    }
}

// One radix-8 Stockham stage on one block: 8 input streams every in_stride, 8 output streams every out_stride,
// k = 0..cnt-1 (cnt a multiple of 8). Same formulas as the main loop (st_radix8_vec), twiddle w^(q*tw).
static inline __attribute__((always_inline)) void st_radix8_stage(
    const double* __restrict in_re, const double* __restrict in_im, size_t in_stride,
    double* __restrict out_re, double* __restrict out_im, size_t out_stride,
    size_t cnt, size_t tw, const double* __restrict w1re, const double* __restrict w1im, bool inverse,
    __m512d v_sign, __m512d v_neg_sign, __m512d v_C, __m512d v_neg_C) {
    __m512d wr[8], wi[8];
    wr[0] = _mm512_setzero_pd(); wi[0] = _mm512_setzero_pd();   // unused (q=0 has no twiddle)
#pragma GCC unroll 8
    for (int q = 1; q < 8; ++q) {
        wr[q] = _mm512_set1_pd(w1re[q * tw]);
        wi[q] = _mm512_set1_pd(inverse ? -w1im[q * tw] : w1im[q * tw]);
    }
    for (size_t k = 0; k < cnt; k += 8) {
        __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int q = 0; q < 8; ++q) {
            r[q] = _mm512_load_pd(in_re + k + q * in_stride);
            i[q] = _mm512_load_pd(in_im + k + q * in_stride);
        }
        st_radix8_vec(r, i, wr, wi, yr, yi, v_sign, v_neg_sign, v_C, v_neg_C);
#pragma GCC unroll 8
        for (int q = 0; q < 8; ++q) {
            _mm512_store_pd(out_re + k + q * out_stride, yr[q]);
            _mm512_store_pd(out_im + k + q * out_stride, yi[q]);
        }
    }
}

// A chain of S consecutive radix-8 passes (from level m0) computed in chunks of R0 = n/m0 elements:
// chunk j reads input j*R0..., keeps intermediate results in two thread-local buffers (L2), and writes
// only the result of the last stage to memory. Saves S-1 full passes over memory.
// Block b (0..8^l-1) at level l has global j_l = j + m0*b; output q of stage l goes to block b + 8^l*q.
static void stockham_fused_chain(const double* in_re, const double* in_im, double* out_re, double* out_im,
    const double* w1re, const double* w1im, size_t n, size_t m0, int S, bool inverse) {
    const size_t R0 = n / m0;
    const double sign = inverse ? -1.0 : 1.0;
    const __m512d v_sign = _mm512_set1_pd(sign);
    const __m512d v_neg_sign = _mm512_set1_pd(-sign);
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    const __m512d v_neg_C = _mm512_set1_pd(-0.70710678118654752440);

#pragma omp parallel
    {
        static thread_local double* scr = nullptr;
        static thread_local size_t scr_cap = 0;
        const size_t need = 4 * R0;
        if (scr_cap < need) {
            std::free(scr);
            scr = static_cast<double*>(std::aligned_alloc(64, ((need * sizeof(double) + 63) / 64) * 64));
            scr_cap = need;
        }
        double* const b0_re = scr;          double* const b0_im = scr + R0;
        double* const b1_re = scr + 2 * R0; double* const b1_im = scr + 3 * R0;

#pragma omp for schedule(static)
        for (size_t j = 0; j < m0; ++j) {
            size_t R = R0;
            size_t nb = 1;   // number of blocks at level l (8^l)
            for (int l = 0; l < S; ++l) {
                const size_t stride = R / 8;     // input stream length (= output block size)
                const bool last = (l == S - 1);
                const double* src_re = (l == 0) ? (in_re + j * R0) : ((l & 1) ? b0_re : b1_re);
                const double* src_im = (l == 0) ? (in_im + j * R0) : ((l & 1) ? b0_im : b1_im);
                double* dst_re = (l & 1) ? b1_re : b0_re;
                double* dst_im = (l & 1) ? b1_im : b0_im;
                for (size_t b = 0; b < nb; ++b) {
                    const size_t jl = j + m0 * b;
                    const size_t tw = jl * stride;
                    const double* sr = src_re + ((l == 0) ? 0 : b * R);
                    const double* si = src_im + ((l == 0) ? 0 : b * R);
                    if (last) {
                        st_radix8_stage(sr, si, stride, out_re + jl * stride, out_im + jl * stride, n / 8,
                            stride, tw, w1re, w1im, inverse, v_sign, v_neg_sign, v_C, v_neg_C);
                    }
                    else {
                        st_radix8_stage(sr, si, stride, dst_re + b * stride, dst_im + b * stride, nb * stride,
                            stride, tw, w1re, w1im, inverse, v_sign, v_neg_sign, v_C, v_neg_C);
                    }
                }
                R = stride;
                nb *= 8;
            }
        }
    }
}

static bool stockham_chain_enabled() {
    // OFF by default: measured 2026-10-03 at 2^19, 1.37 ms/iter with the chain vs 1.04 without (32% slower).
    // LOTR_CHAIN=1 enables it (for further experiments).
    static const bool v = []() { const char* e = std::getenv("LOTR_CHAIN"); return (e && e[0] == '1'); }();
    return v;
}

static bool stockham_fuse_enabled() {
    static const bool v = []() { const char* e = std::getenv("LOTR_NO_FUSE"); return !(e && e[0] == '1'); }();
    return v;
}

double* fft_stockham_radix8(FFTContext& ctx, bool inverse) {
    const size_t n = ctx.fft_len;
    double* in_re = ctx.re; double* in_im = ctx.im;
    double* out_re = ctx.re2; double* out_im = ctx.im2;
    const double sign = inverse ? -1.0 : 1.0;

    const __m512d v_sign = _mm512_set1_pd(sign);
    const __m512d v_neg_sign = _mm512_set1_pd(-sign);
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    const __m512d v_neg_C = _mm512_set1_pd(-0.70710678118654752440);
#ifdef LOTR_PROFILE
    int pass_no = 0;
#endif

    for (size_t m = 1; m < n; ) {
#ifdef LOTR_PROFILE
        LOTR_PASS(ctx, (pass_no < 7 ? pass_no : 7));
        ++pass_no;
#endif
        size_t remaining = n / m;

        if (m >= 16 && remaining <= 8192 && stockham_chain_enabled()) {
            // Consecutive radix-8 passes with vector k (stride >= 8) in one chain over chunks that fit in L2
            int S = 0;
            for (size_t R = remaining; R >= 64; R >>= 3) ++S;
            if (S >= 2) {
                stockham_fused_chain(in_re, in_im, out_re, out_im, &ctx.w1_re[0], &ctx.w1_im[0], n, m, S, inverse);
                for (int s = 0; s < S; ++s) m <<= 3;
                std::swap(in_re, out_re);
                std::swap(in_im, out_im);
                continue;
            }
        }

        if (remaining == 16 && m >= 8 && stockham_fuse_enabled()) {
            // The last two passes (radix-8 at stride 2 + radix-2) in a single sweep
            stockham_fused_last16(in_re, in_im, out_re, out_im, &ctx.w1_re[0], &ctx.w1_im[0], n, inverse);
            std::swap(in_re, out_re);
            std::swap(in_im, out_im);
            break;
        }

        if (remaining >= 8) {
            size_t stride = remaining >> 3;
            if (stride >= 8) {
                // ========================================================
                // AVX-512 OPTIMIZATION: dynamic parallelization
                // ========================================================
                if (m < 16) {
                    // STEP 1: few outer iterations (m=1, m=8).
                    // One OpenMP region over (j, chunk of k) instead of a separate region (with a barrier) for each j.
                    const size_t CH = 2048;
                    const size_t nch = (stride + CH - 1) / CH;
#pragma omp parallel for schedule(static)
                    for (size_t idx = 0; idx < m * nch; idx++) {
                        const size_t j = idx / nch;
                        const size_t k0 = (idx % nch) * CH;
                        const size_t kend = std::min(k0 + CH, stride);
                        size_t tw = j * stride;
                        __m512d v_w1r = _mm512_set1_pd(ctx.w1_re[tw]);       __m512d v_w1i = _mm512_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);
                        __m512d v_w2r = _mm512_set1_pd(ctx.w1_re[2 * tw]);   __m512d v_w2i = _mm512_set1_pd(inverse ? -ctx.w1_im[2 * tw] : ctx.w1_im[2 * tw]);
                        __m512d v_w3r = _mm512_set1_pd(ctx.w1_re[3 * tw]);   __m512d v_w3i = _mm512_set1_pd(inverse ? -ctx.w1_im[3 * tw] : ctx.w1_im[3 * tw]);
                        __m512d v_w4r = _mm512_set1_pd(ctx.w1_re[4 * tw]);   __m512d v_w4i = _mm512_set1_pd(inverse ? -ctx.w1_im[4 * tw] : ctx.w1_im[4 * tw]);
                        __m512d v_w5r = _mm512_set1_pd(ctx.w1_re[5 * tw]);   __m512d v_w5i = _mm512_set1_pd(inverse ? -ctx.w1_im[5 * tw] : ctx.w1_im[5 * tw]);
                        __m512d v_w6r = _mm512_set1_pd(ctx.w1_re[6 * tw]);   __m512d v_w6i = _mm512_set1_pd(inverse ? -ctx.w1_im[6 * tw] : ctx.w1_im[6 * tw]);
                        __m512d v_w7r = _mm512_set1_pd(ctx.w1_re[7 * tw]);   __m512d v_w7i = _mm512_set1_pd(inverse ? -ctx.w1_im[7 * tw] : ctx.w1_im[7 * tw]);

                        for (size_t k = k0; k < kend; k += 8) {
                            size_t i0 = k + j * (stride << 3), i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
                            size_t i4 = i3 + stride, i5 = i4 + stride, i6 = i5 + stride, i7 = i6 + stride;

                            __m512d r0 = _mm512_load_pd(&in_re[i0]), i0v = _mm512_load_pd(&in_im[i0]);
                            __m512d r1 = _mm512_load_pd(&in_re[i1]), i1v = _mm512_load_pd(&in_im[i1]);
                            __m512d r2 = _mm512_load_pd(&in_re[i2]), i2v = _mm512_load_pd(&in_im[i2]);
                            __m512d r3 = _mm512_load_pd(&in_re[i3]), i3v = _mm512_load_pd(&in_im[i3]);
                            __m512d r4 = _mm512_load_pd(&in_re[i4]), i4v = _mm512_load_pd(&in_im[i4]);
                            __m512d r5 = _mm512_load_pd(&in_re[i5]), i5v = _mm512_load_pd(&in_im[i5]);
                            __m512d r6 = _mm512_load_pd(&in_re[i6]), i6v = _mm512_load_pd(&in_im[i6]);
                            __m512d r7 = _mm512_load_pd(&in_re[i7]), i7v = _mm512_load_pd(&in_im[i7]);

                            __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i)); __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                            __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i)); __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r));
                            __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i)); __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r));
                            __m512d t4r = _mm512_fmsub_pd(r4, v_w4r, _mm512_mul_pd(i4v, v_w4i)); __m512d t4i = _mm512_fmadd_pd(r4, v_w4i, _mm512_mul_pd(i4v, v_w4r));
                            __m512d t5r = _mm512_fmsub_pd(r5, v_w5r, _mm512_mul_pd(i5v, v_w5i)); __m512d t5i = _mm512_fmadd_pd(r5, v_w5i, _mm512_mul_pd(i5v, v_w5r));
                            __m512d t6r = _mm512_fmsub_pd(r6, v_w6r, _mm512_mul_pd(i6v, v_w6i)); __m512d t6i = _mm512_fmadd_pd(r6, v_w6i, _mm512_mul_pd(i6v, v_w6r));
                            __m512d t7r = _mm512_fmsub_pd(r7, v_w7r, _mm512_mul_pd(i7v, v_w7i)); __m512d t7i = _mm512_fmadd_pd(r7, v_w7i, _mm512_mul_pd(i7v, v_w7r));

                            __m512d u0r = _mm512_add_pd(r0, t4r), u0i = _mm512_add_pd(i0v, t4i); __m512d u4r = _mm512_sub_pd(r0, t4r), u4i = _mm512_sub_pd(i0v, t4i);
                            __m512d u1r = _mm512_add_pd(t1r, t5r), u1i = _mm512_add_pd(t1i, t5i); __m512d u5r = _mm512_sub_pd(t1r, t5r), u5i = _mm512_sub_pd(t1i, t5i);
                            __m512d u2r = _mm512_add_pd(t2r, t6r), u2i = _mm512_add_pd(t2i, t6i); __m512d u6r = _mm512_sub_pd(t2r, t6r), u6i = _mm512_sub_pd(t2i, t6i);
                            __m512d u3r = _mm512_add_pd(t3r, t7r), u3i = _mm512_add_pd(t3i, t7i); __m512d u7r = _mm512_sub_pd(t3r, t7r), u7i = _mm512_sub_pd(t3i, t7i);

                            __m512d v5r = _mm512_mul_pd(v_C, _mm512_add_pd(u5r, _mm512_mul_pd(u5i, v_sign)));
                            __m512d v5i = _mm512_mul_pd(v_C, _mm512_sub_pd(u5i, _mm512_mul_pd(u5r, v_sign)));
                            __m512d v6r = _mm512_mul_pd(u6i, v_sign);
                            __m512d v6i = _mm512_mul_pd(u6r, v_neg_sign);
                            __m512d v7r = _mm512_mul_pd(v_C, _mm512_sub_pd(_mm512_mul_pd(u7i, v_sign), u7r));
                            __m512d v7i = _mm512_mul_pd(v_neg_C, _mm512_add_pd(_mm512_mul_pd(u7r, v_sign), u7i));

                            __m512d a0r = _mm512_add_pd(u0r, u2r), a0i = _mm512_add_pd(u0i, u2i); __m512d a1r = _mm512_sub_pd(u0r, u2r), a1i = _mm512_sub_pd(u0i, u2i);
                            __m512d a2r = _mm512_add_pd(u1r, u3r), a2i = _mm512_add_pd(u1i, u3i); __m512d a3r = _mm512_sub_pd(u1r, u3r), a3i = _mm512_sub_pd(u1i, u3i);
                            __m512d d3r = _mm512_mul_pd(a3i, v_sign), d3i = _mm512_mul_pd(a3r, v_neg_sign);

                            size_t o0 = k + j * stride, o1 = o0 + m * stride, o2 = o1 + m * stride, o3 = o2 + m * stride;
                            size_t o4 = o3 + m * stride, o5 = o4 + m * stride, o6 = o5 + m * stride, o7 = o6 + m * stride;

                            _mm512_store_pd(&out_re[o0], _mm512_add_pd(a0r, a2r)); _mm512_store_pd(&out_im[o0], _mm512_add_pd(a0i, a2i));
                            _mm512_store_pd(&out_re[o4], _mm512_sub_pd(a0r, a2r)); _mm512_store_pd(&out_im[o4], _mm512_sub_pd(a0i, a2i));
                            _mm512_store_pd(&out_re[o2], _mm512_add_pd(a1r, d3r)); _mm512_store_pd(&out_im[o2], _mm512_add_pd(a1i, d3i));
                            _mm512_store_pd(&out_re[o6], _mm512_sub_pd(a1r, d3r)); _mm512_store_pd(&out_im[o6], _mm512_sub_pd(a1i, d3i));

                            __m512d b0r = _mm512_add_pd(u4r, v6r), b0i = _mm512_add_pd(u4i, v6i); __m512d b1r = _mm512_sub_pd(u4r, v6r), b1i = _mm512_sub_pd(u4i, v6i);
                            __m512d b2r = _mm512_add_pd(v5r, v7r), b2i = _mm512_add_pd(v5i, v7i); __m512d b3r = _mm512_sub_pd(v5r, v7r), b3i = _mm512_sub_pd(v5i, v7i);
                            __m512d e3r = _mm512_mul_pd(b3i, v_sign), e3i = _mm512_mul_pd(b3r, v_neg_sign);

                            _mm512_store_pd(&out_re[o1], _mm512_add_pd(b0r, b2r)); _mm512_store_pd(&out_im[o1], _mm512_add_pd(b0i, b2i));
                            _mm512_store_pd(&out_re[o5], _mm512_sub_pd(b0r, b2r)); _mm512_store_pd(&out_im[o5], _mm512_sub_pd(b0i, b2i));
                            _mm512_store_pd(&out_re[o3], _mm512_add_pd(b1r, e3r)); _mm512_store_pd(&out_im[o3], _mm512_add_pd(b1i, e3i));
                            _mm512_store_pd(&out_re[o7], _mm512_sub_pd(b1r, e3r)); _mm512_store_pd(&out_im[o7], _mm512_sub_pd(b1i, e3i));
                        }
                    }
                }
                else {
                    // STEP 2: m is large enough. Parallelize over 'j' again,
                    // which avoids creating too many small blocks for the threads.
#pragma omp parallel for schedule(static)
                    for (size_t j = 0; j < m; j++) {
                        size_t tw = j * stride;
                        __m512d v_w1r = _mm512_set1_pd(ctx.w1_re[tw]);       __m512d v_w1i = _mm512_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);
                        __m512d v_w2r = _mm512_set1_pd(ctx.w1_re[2 * tw]);   __m512d v_w2i = _mm512_set1_pd(inverse ? -ctx.w1_im[2 * tw] : ctx.w1_im[2 * tw]);
                        __m512d v_w3r = _mm512_set1_pd(ctx.w1_re[3 * tw]);   __m512d v_w3i = _mm512_set1_pd(inverse ? -ctx.w1_im[3 * tw] : ctx.w1_im[3 * tw]);
                        __m512d v_w4r = _mm512_set1_pd(ctx.w1_re[4 * tw]);   __m512d v_w4i = _mm512_set1_pd(inverse ? -ctx.w1_im[4 * tw] : ctx.w1_im[4 * tw]);
                        __m512d v_w5r = _mm512_set1_pd(ctx.w1_re[5 * tw]);   __m512d v_w5i = _mm512_set1_pd(inverse ? -ctx.w1_im[5 * tw] : ctx.w1_im[5 * tw]);
                        __m512d v_w6r = _mm512_set1_pd(ctx.w1_re[6 * tw]);   __m512d v_w6i = _mm512_set1_pd(inverse ? -ctx.w1_im[6 * tw] : ctx.w1_im[6 * tw]);
                        __m512d v_w7r = _mm512_set1_pd(ctx.w1_re[7 * tw]);   __m512d v_w7i = _mm512_set1_pd(inverse ? -ctx.w1_im[7 * tw] : ctx.w1_im[7 * tw]);

                        for (size_t k = 0; k < stride; k += 8) {
                            size_t i0 = k + j * (stride << 3), i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
                            size_t i4 = i3 + stride, i5 = i4 + stride, i6 = i5 + stride, i7 = i6 + stride;

                            __m512d r0 = _mm512_load_pd(&in_re[i0]), i0v = _mm512_load_pd(&in_im[i0]);
                            __m512d r1 = _mm512_load_pd(&in_re[i1]), i1v = _mm512_load_pd(&in_im[i1]);
                            __m512d r2 = _mm512_load_pd(&in_re[i2]), i2v = _mm512_load_pd(&in_im[i2]);
                            __m512d r3 = _mm512_load_pd(&in_re[i3]), i3v = _mm512_load_pd(&in_im[i3]);
                            __m512d r4 = _mm512_load_pd(&in_re[i4]), i4v = _mm512_load_pd(&in_im[i4]);
                            __m512d r5 = _mm512_load_pd(&in_re[i5]), i5v = _mm512_load_pd(&in_im[i5]);
                            __m512d r6 = _mm512_load_pd(&in_re[i6]), i6v = _mm512_load_pd(&in_im[i6]);
                            __m512d r7 = _mm512_load_pd(&in_re[i7]), i7v = _mm512_load_pd(&in_im[i7]);

                            __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i)); __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                            __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i)); __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r));
                            __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i)); __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r));
                            __m512d t4r = _mm512_fmsub_pd(r4, v_w4r, _mm512_mul_pd(i4v, v_w4i)); __m512d t4i = _mm512_fmadd_pd(r4, v_w4i, _mm512_mul_pd(i4v, v_w4r));
                            __m512d t5r = _mm512_fmsub_pd(r5, v_w5r, _mm512_mul_pd(i5v, v_w5i)); __m512d t5i = _mm512_fmadd_pd(r5, v_w5i, _mm512_mul_pd(i5v, v_w5r));
                            __m512d t6r = _mm512_fmsub_pd(r6, v_w6r, _mm512_mul_pd(i6v, v_w6i)); __m512d t6i = _mm512_fmadd_pd(r6, v_w6i, _mm512_mul_pd(i6v, v_w6r));
                            __m512d t7r = _mm512_fmsub_pd(r7, v_w7r, _mm512_mul_pd(i7v, v_w7i)); __m512d t7i = _mm512_fmadd_pd(r7, v_w7i, _mm512_mul_pd(i7v, v_w7r));

                            __m512d u0r = _mm512_add_pd(r0, t4r), u0i = _mm512_add_pd(i0v, t4i); __m512d u4r = _mm512_sub_pd(r0, t4r), u4i = _mm512_sub_pd(i0v, t4i);
                            __m512d u1r = _mm512_add_pd(t1r, t5r), u1i = _mm512_add_pd(t1i, t5i); __m512d u5r = _mm512_sub_pd(t1r, t5r), u5i = _mm512_sub_pd(t1i, t5i);
                            __m512d u2r = _mm512_add_pd(t2r, t6r), u2i = _mm512_add_pd(t2i, t6i); __m512d u6r = _mm512_sub_pd(t2r, t6r), u6i = _mm512_sub_pd(t2i, t6i);
                            __m512d u3r = _mm512_add_pd(t3r, t7r), u3i = _mm512_add_pd(t3i, t7i); __m512d u7r = _mm512_sub_pd(t3r, t7r), u7i = _mm512_sub_pd(t3i, t7i);

                            __m512d v5r = _mm512_mul_pd(v_C, _mm512_add_pd(u5r, _mm512_mul_pd(u5i, v_sign)));
                            __m512d v5i = _mm512_mul_pd(v_C, _mm512_sub_pd(u5i, _mm512_mul_pd(u5r, v_sign)));
                            __m512d v6r = _mm512_mul_pd(u6i, v_sign);
                            __m512d v6i = _mm512_mul_pd(u6r, v_neg_sign);
                            __m512d v7r = _mm512_mul_pd(v_C, _mm512_sub_pd(_mm512_mul_pd(u7i, v_sign), u7r));
                            __m512d v7i = _mm512_mul_pd(v_neg_C, _mm512_add_pd(_mm512_mul_pd(u7r, v_sign), u7i));

                            __m512d a0r = _mm512_add_pd(u0r, u2r), a0i = _mm512_add_pd(u0i, u2i); __m512d a1r = _mm512_sub_pd(u0r, u2r), a1i = _mm512_sub_pd(u0i, u2i);
                            __m512d a2r = _mm512_add_pd(u1r, u3r), a2i = _mm512_add_pd(u1i, u3i); __m512d a3r = _mm512_sub_pd(u1r, u3r), a3i = _mm512_sub_pd(u1i, u3i);
                            __m512d d3r = _mm512_mul_pd(a3i, v_sign), d3i = _mm512_mul_pd(a3r, v_neg_sign);

                            size_t o0 = k + j * stride, o1 = o0 + m * stride, o2 = o1 + m * stride, o3 = o2 + m * stride;
                            size_t o4 = o3 + m * stride, o5 = o4 + m * stride, o6 = o5 + m * stride, o7 = o6 + m * stride;

                            _mm512_store_pd(&out_re[o0], _mm512_add_pd(a0r, a2r)); _mm512_store_pd(&out_im[o0], _mm512_add_pd(a0i, a2i));
                            _mm512_store_pd(&out_re[o4], _mm512_sub_pd(a0r, a2r)); _mm512_store_pd(&out_im[o4], _mm512_sub_pd(a0i, a2i));
                            _mm512_store_pd(&out_re[o2], _mm512_add_pd(a1r, d3r)); _mm512_store_pd(&out_im[o2], _mm512_add_pd(a1i, d3i));
                            _mm512_store_pd(&out_re[o6], _mm512_sub_pd(a1r, d3r)); _mm512_store_pd(&out_im[o6], _mm512_sub_pd(a1i, d3i));

                            __m512d b0r = _mm512_add_pd(u4r, v6r), b0i = _mm512_add_pd(u4i, v6i); __m512d b1r = _mm512_sub_pd(u4r, v6r), b1i = _mm512_sub_pd(u4i, v6i);
                            __m512d b2r = _mm512_add_pd(v5r, v7r), b2i = _mm512_add_pd(v5i, v7i); __m512d b3r = _mm512_sub_pd(v5r, v7r), b3i = _mm512_sub_pd(v5i, v7i);
                            __m512d e3r = _mm512_mul_pd(b3i, v_sign), e3i = _mm512_mul_pd(b3r, v_neg_sign);

                            _mm512_store_pd(&out_re[o1], _mm512_add_pd(b0r, b2r)); _mm512_store_pd(&out_im[o1], _mm512_add_pd(b0i, b2i));
                            _mm512_store_pd(&out_re[o5], _mm512_sub_pd(b0r, b2r)); _mm512_store_pd(&out_im[o5], _mm512_sub_pd(b0i, b2i));
                            _mm512_store_pd(&out_re[o3], _mm512_add_pd(b1r, e3r)); _mm512_store_pd(&out_im[o3], _mm512_add_pd(b1i, e3i));
                            _mm512_store_pd(&out_re[o7], _mm512_sub_pd(b1r, e3r)); _mm512_store_pd(&out_im[o7], _mm512_sub_pd(b1i, e3i));
                        }
                    }
                }
            }
            else {
                double C = 0.70710678118654752440;
#pragma omp parallel for schedule(static)
                for (size_t j = 0; j < m; j++) {
                    size_t tw = j * stride;
                    double w1r = ctx.w1_re[tw], w1i = inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw];
                    double w2r = ctx.w1_re[2 * tw], w2i = inverse ? -ctx.w1_im[2 * tw] : ctx.w1_im[2 * tw];
                    double w3r = ctx.w1_re[3 * tw], w3i = inverse ? -ctx.w1_im[3 * tw] : ctx.w1_im[3 * tw];
                    double w4r = ctx.w1_re[4 * tw], w4i = inverse ? -ctx.w1_im[4 * tw] : ctx.w1_im[4 * tw];
                    double w5r = ctx.w1_re[5 * tw], w5i = inverse ? -ctx.w1_im[5 * tw] : ctx.w1_im[5 * tw];
                    double w6r = ctx.w1_re[6 * tw], w6i = inverse ? -ctx.w1_im[6 * tw] : ctx.w1_im[6 * tw];
                    double w7r = ctx.w1_re[7 * tw], w7i = inverse ? -ctx.w1_im[7 * tw] : ctx.w1_im[7 * tw];

                    for (size_t k = 0; k < stride; ++k) {
                        size_t i0 = k + j * 8 * stride, i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
                        size_t i4 = i3 + stride, i5 = i4 + stride, i6 = i5 + stride, i7 = i6 + stride;

                        double r0 = in_re[i0], i0v = in_im[i0], r1 = in_re[i1], i1v = in_im[i1], r2 = in_re[i2], i2v = in_im[i2], r3 = in_re[i3], i3v = in_im[i3];
                        double r4 = in_re[i4], i4v = in_im[i4], r5 = in_re[i5], i5v = in_im[i5], r6 = in_re[i6], i6v = in_im[i6], r7 = in_re[i7], i7v = in_im[i7];

                        double t1r = r1 * w1r - i1v * w1i, t1i = r1 * w1i + i1v * w1r; double t2r = r2 * w2r - i2v * w2i, t2i = r2 * w2i + i2v * w2r;
                        double t3r = r3 * w3r - i3v * w3i, t3i = r3 * w3i + i3v * w3r; double t4r = r4 * w4r - i4v * w4i, t4i = r4 * w4i + i4v * w4r;
                        double t5r = r5 * w5r - i5v * w5i, t5i = r5 * w5i + i5v * w5r; double t6r = r6 * w6r - i6v * w6i, t6i = r6 * w6i + i6v * w6r;
                        double t7r = r7 * w7r - i7v * w7i, t7i = r7 * w7i + i7v * w7r;

                        double u0r = r0 + t4r, u0i = i0v + t4i; double u4r = r0 - t4r, u4i = i0v - t4i;
                        double u1r = t1r + t5r, u1i = t1i + t5i; double u5r = t1r - t5r, u5i = t1i - t5i;
                        double u2r = t2r + t6r, u2i = t2i + t6i; double u6r = t2r - t6r, u6i = t2i - t6i;
                        double u3r = t3r + t7r, u3i = t3i + t7i; double u7r = t3r - t7r, u7i = t3i - t7i;

                        double v5r = C * (u5r + u5i * sign), v5i = C * (u5i - u5r * sign);
                        double v6r = u6i * sign, v6i = -u6r * sign;
                        double v7r = C * (-u7r + u7i * sign), v7i = C * (-u7r * sign - u7i);

                        double a0r = u0r + u2r, a0i = u0i + u2i; double a1r = u0r - u2r, a1i = u0i - u2i;
                        double a2r = u1r + u3r, a2i = u1i + u3i; double a3r = u1r - u3r, a3i = u1i - u3i;
                        double d3r = a3i * sign, d3i = -a3r * sign;

                        double b0r = u4r + v6r, b0i = u4i + v6i; double b1r = u4r - v6r, b1i = u4i - v6i;
                        double b2r = v5r + v7r, b2i = v5i + v7i; double b3r = v5r - v7r, b3i = v5i - v7i;
                        double e3r = b3i * sign, e3i = -b3r * sign;

                        size_t o0 = k + j * stride, o1 = o0 + m * stride, o2 = o1 + m * stride, o3 = o2 + m * stride;
                        size_t o4 = o3 + m * stride, o5 = o4 + m * stride, o6 = o5 + m * stride, o7 = o6 + m * stride;

                        out_re[o0] = a0r + a2r; out_im[o0] = a0i + a2i; out_re[o4] = a0r - a2r; out_im[o4] = a0i - a2i;
                        out_re[o2] = a1r + d3r; out_im[o2] = a1i + d3i; out_re[o6] = a1r - d3r; out_im[o6] = a1i - d3i;
                        out_re[o1] = b0r + b2r; out_im[o1] = b0i + b2i; out_re[o5] = b0r - b2r; out_im[o5] = b0i - b2i;
                        out_re[o3] = b1r + e3r; out_im[o3] = b1i + e3i; out_re[o7] = b1r - e3r; out_im[o7] = b1i - e3i;
                    }
                }
            }
            m <<= 3;
        }
        else if (remaining >= 4) {
            size_t stride = remaining >> 2;
#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;
                double w1r = ctx.w1_re[tw], w1i = inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw];
                double w2r = ctx.w1_re[2 * tw], w2i = inverse ? -ctx.w1_im[2 * tw] : ctx.w1_im[2 * tw];
                double w3r = ctx.w1_re[3 * tw], w3i = inverse ? -ctx.w1_im[3 * tw] : ctx.w1_im[3 * tw];

                for (size_t k = 0; k < stride; ++k) {
                    size_t i0 = k + j * 4 * stride, i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
                    double r0 = in_re[i0], i0v = in_im[i0], r1 = in_re[i1], i1v = in_im[i1], r2 = in_re[i2], i2v = in_im[i2], r3 = in_re[i3], i3v = in_im[i3];
                    double t1r = r1 * w1r - i1v * w1i, t1i = r1 * w1i + i1v * w1r;
                    double t2r = r2 * w2r - i2v * w2i, t2i = r2 * w2i + i2v * w2r;
                    double t3r = r3 * w3r - i3v * w3i, t3i = r3 * w3i + i3v * w3r;
                    double s0r = r0 + t2r, s0i = i0v + t2i, s1r = r0 - t2r, s1i = i0v - t2i, s2r = t1r + t3r, s2i = t1i + t3i, s3r = t1r - t3r, s3i = t1i - t3i;
                    double d3r = s3i * sign, d3i = -s3r * sign;
                    size_t o0 = k + j * stride, o1 = o0 + m * stride, o2 = o1 + m * stride, o3 = o2 + m * stride;
                    out_re[o0] = s0r + s2r; out_im[o0] = s0i + s2i; out_re[o1] = s1r + d3r; out_im[o1] = s1i + d3i;
                    out_re[o2] = s0r - s2r; out_im[o2] = s0i - s2i; out_re[o3] = s1r - d3r; out_im[o3] = s1i - d3i;
                }
            }
            m <<= 2;
        }
        else if (remaining >= 2) {
            size_t stride = remaining >> 1;
#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;
                double w1r = ctx.w1_re[tw], w1i = inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw];
                for (size_t k = 0; k < stride; ++k) {
                    size_t i0 = k + j * 2 * stride, i1 = i0 + stride;
                    double r0 = in_re[i0], i0v = in_im[i0], r1 = in_re[i1], i1v = in_im[i1];
                    double t1r = r1 * w1r - i1v * w1i, t1i = r1 * w1i + i1v * w1r;
                    size_t o0 = k + j * stride, o1 = o0 + m * stride;
                    out_re[o0] = r0 + t1r; out_im[o0] = i0v + t1i;
                    out_re[o1] = r0 - t1r; out_im[o1] = i0v - t1i;
                }
            }
            m <<= 1;
        }
        std::swap(in_re, out_re);
        std::swap(in_im, out_im);
    }
    return in_re;
}

// ---------------------------------------------------------
// CACHE-FRIENDLY RADIX-4 (Bailey 2D) - WITH RADIX-2 SUPPORT AT THE END
// ---------------------------------------------------------
double* fft_stockham_radix4_blocked(double* in_re, double* in_im, double* out_re, double* out_im,
    size_t n,
    const std::vector<double>& sw1r, const std::vector<double>& sw1i,
    const std::vector<double>& sw2r, const std::vector<double>& sw2i,
    const std::vector<double>& sw3r, const std::vector<double>& sw3i,
    bool inverse) {

    const double sign = inverse ? -1.0 : 1.0;
    const __m512d v_sign = _mm512_set1_pd(sign);
    const __m512d v_neg_sign = _mm512_set1_pd(-sign);

    double* cur_in_re = in_re; double* cur_in_im = in_im;
    double* cur_out_re = out_re; double* cur_out_im = out_im;

    for (size_t m = 1; m < n; ) {
        size_t remaining = n / m;

        // ===============================================================
        // RADIX-4 STEP (for all full quad blocks)
        // ===============================================================
        if (remaining >= 4) {
            size_t stride = remaining >> 2;

            if (stride >= 8) {
                for (size_t j = 0; j < m; j++) {
                    size_t tw = j * stride;
                    __m512d v_w1r = _mm512_set1_pd(sw1r[tw]);
                    __m512d v_w1i = _mm512_set1_pd(inverse ? -sw1i[tw] : sw1i[tw]);
                    __m512d v_w2r = _mm512_set1_pd(sw2r[tw]);
                    __m512d v_w2i = _mm512_set1_pd(inverse ? -sw2i[tw] : sw2i[tw]);
                    __m512d v_w3r = _mm512_set1_pd(sw3r[tw]);
                    __m512d v_w3i = _mm512_set1_pd(inverse ? -sw3i[tw] : sw3i[tw]);

                    size_t k = 0;

                    for (; k + 15 < stride; k += 16) {
                        size_t i0_A = k + j * (stride << 2);
                        size_t i1_A = i0_A + stride; size_t i2_A = i1_A + stride; size_t i3_A = i2_A + stride;

                        size_t i0_B = i0_A + 8;
                        size_t i1_B = i1_A + 8; size_t i2_B = i2_A + 8; size_t i3_B = i3_A + 8;

                        // CACHE OPTIMIZATION: T0 loads directly into the L1 cache
                        size_t prefetch_offset = 128;
                        _mm_prefetch((const char*)&cur_in_re[i0_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i0_B + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i1_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i1_B + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i2_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i2_B + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i3_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i3_B + prefetch_offset], _MM_HINT_T0);

                        _mm_prefetch((const char*)&cur_in_im[i0_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i0_B + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i1_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i1_B + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i2_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i2_B + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i3_A + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i3_B + prefetch_offset], _MM_HINT_T0);

                        __m512d r0_A = _mm512_load_pd(&cur_in_re[i0_A]), i0v_A = _mm512_load_pd(&cur_in_im[i0_A]);
                        __m512d r0_B = _mm512_load_pd(&cur_in_re[i0_B]), i0v_B = _mm512_load_pd(&cur_in_im[i0_B]);
                        __m512d r1_A = _mm512_load_pd(&cur_in_re[i1_A]), i1v_A = _mm512_load_pd(&cur_in_im[i1_A]);
                        __m512d r1_B = _mm512_load_pd(&cur_in_re[i1_B]), i1v_B = _mm512_load_pd(&cur_in_im[i1_B]);
                        __m512d r2_A = _mm512_load_pd(&cur_in_re[i2_A]), i2v_A = _mm512_load_pd(&cur_in_im[i2_A]);
                        __m512d r2_B = _mm512_load_pd(&cur_in_re[i2_B]), i2v_B = _mm512_load_pd(&cur_in_im[i2_B]);
                        __m512d r3_A = _mm512_load_pd(&cur_in_re[i3_A]), i3v_A = _mm512_load_pd(&cur_in_im[i3_A]);
                        __m512d r3_B = _mm512_load_pd(&cur_in_re[i3_B]), i3v_B = _mm512_load_pd(&cur_in_im[i3_B]);

                        __m512d m1_A = _mm512_mul_pd(i1v_A, v_w1i);
                        __m512d m1_B = _mm512_mul_pd(i1v_B, v_w1i);
                        __m512d t1r_A = _mm512_fmsub_pd(r1_A, v_w1r, m1_A);
                        __m512d t1r_B = _mm512_fmsub_pd(r1_B, v_w1r, m1_B);

                        __m512d m2_A = _mm512_mul_pd(i1v_A, v_w1r);
                        __m512d m2_B = _mm512_mul_pd(i1v_B, v_w1r);
                        __m512d t1i_A = _mm512_fmadd_pd(r1_A, v_w1i, m2_A);
                        __m512d t1i_B = _mm512_fmadd_pd(r1_B, v_w1i, m2_B);

                        __m512d m3_A = _mm512_mul_pd(i2v_A, v_w2i);
                        __m512d m3_B = _mm512_mul_pd(i2v_B, v_w2i);
                        __m512d t2r_A = _mm512_fmsub_pd(r2_A, v_w2r, m3_A);
                        __m512d t2r_B = _mm512_fmsub_pd(r2_B, v_w2r, m3_B);

                        __m512d m4_A = _mm512_mul_pd(i2v_A, v_w2r);
                        __m512d m4_B = _mm512_mul_pd(i2v_B, v_w2r);
                        __m512d t2i_A = _mm512_fmadd_pd(r2_A, v_w2i, m4_A);
                        __m512d t2i_B = _mm512_fmadd_pd(r2_B, v_w2i, m4_B);

                        __m512d m5_A = _mm512_mul_pd(i3v_A, v_w3i);
                        __m512d m5_B = _mm512_mul_pd(i3v_B, v_w3i);
                        __m512d t3r_A = _mm512_fmsub_pd(r3_A, v_w3r, m5_A);
                        __m512d t3r_B = _mm512_fmsub_pd(r3_B, v_w3r, m5_B);

                        __m512d m6_A = _mm512_mul_pd(i3v_A, v_w3r);
                        __m512d m6_B = _mm512_mul_pd(i3v_B, v_w3r);
                        __m512d t3i_A = _mm512_fmadd_pd(r3_A, v_w3i, m6_A);
                        __m512d t3i_B = _mm512_fmadd_pd(r3_B, v_w3i, m6_B);

                        __m512d s0r_A = _mm512_add_pd(r0_A, t2r_A), s0i_A = _mm512_add_pd(i0v_A, t2i_A);
                        __m512d s0r_B = _mm512_add_pd(r0_B, t2r_B), s0i_B = _mm512_add_pd(i0v_B, t2i_B);
                        __m512d s1r_A = _mm512_sub_pd(r0_A, t2r_A), s1i_A = _mm512_sub_pd(i0v_A, t2i_A);
                        __m512d s1r_B = _mm512_sub_pd(r0_B, t2r_B), s1i_B = _mm512_sub_pd(i0v_B, t2i_B);

                        __m512d s2r_A = _mm512_add_pd(t1r_A, t3r_A), s2i_A = _mm512_add_pd(t1i_A, t3i_A);
                        __m512d s2r_B = _mm512_add_pd(t1r_B, t3r_B), s2i_B = _mm512_add_pd(t1i_B, t3i_B);
                        __m512d s3r_A = _mm512_sub_pd(t1r_A, t3r_A), s3i_A = _mm512_sub_pd(t1i_A, t3i_A);
                        __m512d s3r_B = _mm512_sub_pd(t1r_B, t3r_B), s3i_B = _mm512_sub_pd(t1i_B, t3i_B);

                        __m512d d3r_A = _mm512_mul_pd(s3i_A, v_sign), d3i_A = _mm512_mul_pd(s3r_A, v_neg_sign);
                        __m512d d3r_B = _mm512_mul_pd(s3i_B, v_sign), d3i_B = _mm512_mul_pd(s3r_B, v_neg_sign);

                        size_t o0_A = k + j * stride;           size_t o1_A = o0_A + m * stride;
                        size_t o2_A = o1_A + m * stride;        size_t o3_A = o2_A + m * stride;
                        size_t o0_B = o0_A + 8;                 size_t o1_B = o1_A + 8;
                        size_t o2_B = o2_A + 8;                 size_t o3_B = o3_A + 8;

                        _mm512_storeu_pd(&cur_out_re[o0_A], _mm512_add_pd(s0r_A, s2r_A)); _mm512_storeu_pd(&cur_out_im[o0_A], _mm512_add_pd(s0i_A, s2i_A));
                        _mm512_storeu_pd(&cur_out_re[o0_B], _mm512_add_pd(s0r_B, s2r_B)); _mm512_storeu_pd(&cur_out_im[o0_B], _mm512_add_pd(s0i_B, s2i_B));
                        _mm512_storeu_pd(&cur_out_re[o1_A], _mm512_add_pd(s1r_A, d3r_A)); _mm512_storeu_pd(&cur_out_im[o1_A], _mm512_add_pd(s1i_A, d3i_A));
                        _mm512_storeu_pd(&cur_out_re[o1_B], _mm512_add_pd(s1r_B, d3r_B)); _mm512_storeu_pd(&cur_out_im[o1_B], _mm512_add_pd(s1i_B, d3i_B));
                        _mm512_storeu_pd(&cur_out_re[o2_A], _mm512_sub_pd(s0r_A, s2r_A)); _mm512_storeu_pd(&cur_out_im[o2_A], _mm512_sub_pd(s0i_A, s2i_A));
                        _mm512_storeu_pd(&cur_out_re[o2_B], _mm512_sub_pd(s0r_B, s2r_B)); _mm512_storeu_pd(&cur_out_im[o2_B], _mm512_sub_pd(s0i_B, s2i_B));
                        _mm512_storeu_pd(&cur_out_re[o3_A], _mm512_sub_pd(s1r_A, d3r_A)); _mm512_storeu_pd(&cur_out_im[o3_A], _mm512_sub_pd(s1i_A, d3i_A));
                        _mm512_storeu_pd(&cur_out_re[o3_B], _mm512_sub_pd(s1r_B, d3r_B)); _mm512_storeu_pd(&cur_out_im[o3_B], _mm512_sub_pd(s1i_B, d3i_B));
                    }


                    for (; k < stride; k += 8) {
                        size_t i0 = k + j * (stride << 2);
                        size_t i1 = i0 + stride; size_t i2 = i1 + stride; size_t i3 = i2 + stride;
                        size_t prefetch_offset = 128;

                        _mm_prefetch((const char*)&cur_in_re[i0 + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i1 + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i2 + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_re[i3 + prefetch_offset], _MM_HINT_T0);

                        _mm_prefetch((const char*)&cur_in_im[i0 + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i1 + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i2 + prefetch_offset], _MM_HINT_T0);
                        _mm_prefetch((const char*)&cur_in_im[i3 + prefetch_offset], _MM_HINT_T0);

                        __m512d r0 = _mm512_load_pd(&cur_in_re[i0]), i0v = _mm512_load_pd(&cur_in_im[i0]);
                        __m512d r1 = _mm512_load_pd(&cur_in_re[i1]), i1v = _mm512_load_pd(&cur_in_im[i1]);
                        __m512d r2 = _mm512_load_pd(&cur_in_re[i2]), i2v = _mm512_load_pd(&cur_in_im[i2]);
                        __m512d r3 = _mm512_load_pd(&cur_in_re[i3]), i3v = _mm512_load_pd(&cur_in_im[i3]);

                        __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i));
                        __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                        __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i));
                        __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r));
                        __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i));
                        __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r));

                        __m512d s0r = _mm512_add_pd(r0, t2r), s0i = _mm512_add_pd(i0v, t2i);
                        __m512d s1r = _mm512_sub_pd(r0, t2r), s1i = _mm512_sub_pd(i0v, t2i);
                        __m512d s2r = _mm512_add_pd(t1r, t3r), s2i = _mm512_add_pd(t1i, t3i);
                        __m512d s3r = _mm512_sub_pd(t1r, t3r), s3i = _mm512_sub_pd(t1i, t3i);

                        __m512d d3r = _mm512_mul_pd(s3i, v_sign);
                        __m512d d3i = _mm512_mul_pd(s3r, v_neg_sign);

                        size_t o0 = k + j * stride; size_t o1 = o0 + m * stride;
                        size_t o2 = o1 + m * stride; size_t o3 = o2 + m * stride;
                        _mm512_storeu_pd(&cur_out_re[o0], _mm512_add_pd(s0r, s2r));
                        _mm512_storeu_pd(&cur_out_im[o0], _mm512_add_pd(s0i, s2i));
                        _mm512_storeu_pd(&cur_out_re[o1], _mm512_add_pd(s1r, d3r));
                        _mm512_storeu_pd(&cur_out_im[o1], _mm512_add_pd(s1i, d3i));
                        _mm512_storeu_pd(&cur_out_re[o2], _mm512_sub_pd(s0r, s2r));
                        _mm512_storeu_pd(&cur_out_im[o2], _mm512_sub_pd(s0i, s2i));
                        _mm512_storeu_pd(&cur_out_re[o3], _mm512_sub_pd(s1r, d3r));
                        _mm512_storeu_pd(&cur_out_im[o3], _mm512_sub_pd(s1i, d3i));
                    }
                }
            }
            else if (stride == 4 && m >= 2) {
                for (size_t j = 0; j < m; j += 2) {
                    size_t tw0 = j * 4, tw1 = (j + 1) * 4;
                    auto tw_v = [](double t0, double t1) {
                        return _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_set1_pd(t0)), _mm256_set1_pd(t1), 1);
                        };
                    __m512d v_w1r = tw_v(sw1r[tw0], sw1r[tw1]);
                    __m512d v_w1i = tw_v(inverse ? -sw1i[tw0] : sw1i[tw0], inverse ? -sw1i[tw1] : sw1i[tw1]);
                    __m512d v_w2r = tw_v(sw2r[tw0], sw2r[tw1]);
                    __m512d v_w2i = tw_v(inverse ? -sw2i[tw0] : sw2i[tw0], inverse ? -sw2i[tw1] : sw2i[tw1]);
                    __m512d v_w3r = tw_v(sw3r[tw0], sw3r[tw1]);
                    __m512d v_w3i = tw_v(inverse ? -sw3i[tw0] : sw3i[tw0], inverse ? -sw3i[tw1] : sw3i[tw1]);

                    auto load2x4 = [&](const double* ptr, size_t off) {
                        return _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(&ptr[j * 16 + off])), _mm256_loadu_pd(&ptr[(j + 1) * 16 + off]), 1);
                        };
                    __m512d r0 = load2x4(cur_in_re, 0);  __m512d i0v = load2x4(cur_in_im, 0);
                    __m512d r1 = load2x4(cur_in_re, 4);  __m512d i1v = load2x4(cur_in_im, 4);
                    __m512d r2 = load2x4(cur_in_re, 8);  __m512d i2v = load2x4(cur_in_im, 8);
                    __m512d r3 = load2x4(cur_in_re, 12); __m512d i3v = load2x4(cur_in_im, 12);

                    __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i));
                    __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                    __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i));
                    __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r));
                    __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i));
                    __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r));
                    __m512d s0r = _mm512_add_pd(r0, t2r), s0i = _mm512_add_pd(i0v, t2i);
                    __m512d s1r = _mm512_sub_pd(r0, t2r), s1i = _mm512_sub_pd(i0v, t2i);
                    __m512d s2r = _mm512_add_pd(t1r, t3r), s2i = _mm512_add_pd(t1i, t3i);
                    __m512d s3r = _mm512_sub_pd(t1r, t3r), s3i = _mm512_sub_pd(t1i, t3i);
                    __m512d d3r = _mm512_mul_pd(s3i, v_sign), d3i = _mm512_mul_pd(s3r, v_neg_sign);

                    size_t o0 = j * 4; size_t o1 = o0 + m * 4; size_t o2 = o1 + m * 4; size_t o3 = o2 + m * 4;
                    _mm512_storeu_pd(&cur_out_re[o0], _mm512_add_pd(s0r, s2r)); _mm512_storeu_pd(&cur_out_im[o0], _mm512_add_pd(s0i, s2i));
                    _mm512_storeu_pd(&cur_out_re[o1], _mm512_add_pd(s1r, d3r)); _mm512_storeu_pd(&cur_out_im[o1], _mm512_add_pd(s1i, d3i));
                    _mm512_storeu_pd(&cur_out_re[o2], _mm512_sub_pd(s0r, s2r)); _mm512_storeu_pd(&cur_out_im[o2], _mm512_sub_pd(s0i, s2i));
                    _mm512_storeu_pd(&cur_out_re[o3], _mm512_sub_pd(s1r, d3r)); _mm512_storeu_pd(&cur_out_im[o3], _mm512_sub_pd(s1i, d3i));
                }
            }
            else if (stride == 2 && m >= 4) {
                for (size_t j = 0; j < m; j += 4) {
                    auto tw_im = [&](const std::vector<double>& arr, size_t idx) { return inverse ? -arr[idx] : arr[idx]; };
                    auto tw_v2 = [](double t0, double t1, double t2, double t3) {
                        return _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_setr_pd(t0, t0, t1, t1)), _mm256_setr_pd(t2, t2, t3, t3), 1);
                        };
                    size_t t0 = j * 2, t1 = (j + 1) * 2, t2 = (j + 2) * 2, t3 = (j + 3) * 2;
                    __m512d v_w1r = tw_v2(sw1r[t0], sw1r[t1], sw1r[t2], sw1r[t3]);
                    __m512d v_w1i = tw_v2(tw_im(sw1i, t0), tw_im(sw1i, t1), tw_im(sw1i, t2), tw_im(sw1i, t3));
                    __m512d v_w2r = tw_v2(sw2r[t0], sw2r[t1], sw2r[t2], sw2r[t3]);
                    __m512d v_w2i = tw_v2(tw_im(sw2i, t0), tw_im(sw2i, t1), tw_im(sw2i, t2), tw_im(sw2i, t3));
                    __m512d v_w3r = tw_v2(sw3r[t0], sw3r[t1], sw3r[t2], sw3r[t3]);
                    __m512d v_w3i = tw_v2(tw_im(sw3i, t0), tw_im(sw3i, t1), tw_im(sw3i, t2), tw_im(sw3i, t3));

                    auto load4x2 = [&](const double* ptr, size_t off) {
                        __m256d lo = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(&ptr[(j + 0) * 8 + off])), _mm_loadu_pd(&ptr[(j + 1) * 8 + off]), 1);
                        __m256d hi = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(&ptr[(j + 2) * 8 + off])), _mm_loadu_pd(&ptr[(j + 3) * 8 + off]), 1);
                        return _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1);
                        };
                    __m512d r0 = load4x2(cur_in_re, 0); __m512d i0v = load4x2(cur_in_im, 0);
                    __m512d r1 = load4x2(cur_in_re, 2); __m512d i1v = load4x2(cur_in_im, 2);
                    __m512d r2 = load4x2(cur_in_re, 4); __m512d i2v = load4x2(cur_in_im, 4);
                    __m512d r3 = load4x2(cur_in_re, 6); __m512d i3v = load4x2(cur_in_im, 6);

                    __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i));
                    __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                    __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i));
                    __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r));
                    __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i));
                    __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r));
                    __m512d s0r = _mm512_add_pd(r0, t2r), s0i = _mm512_add_pd(i0v, t2i);
                    __m512d s1r = _mm512_sub_pd(r0, t2r), s1i = _mm512_sub_pd(i0v, t2i);
                    __m512d s2r = _mm512_add_pd(t1r, t3r), s2i = _mm512_add_pd(t1i, t3i);
                    __m512d s3r = _mm512_sub_pd(t1r, t3r), s3i = _mm512_sub_pd(t1i, t3i);
                    __m512d d3r = _mm512_mul_pd(s3i, v_sign), d3i = _mm512_mul_pd(s3r, v_neg_sign);

                    size_t o0 = j * 2; size_t o1 = o0 + m * 2; size_t o2 = o1 + m * 2; size_t o3 = o2 + m * 2;
                    _mm512_storeu_pd(&cur_out_re[o0], _mm512_add_pd(s0r, s2r)); _mm512_storeu_pd(&cur_out_im[o0], _mm512_add_pd(s0i, s2i));
                    _mm512_storeu_pd(&cur_out_re[o1], _mm512_add_pd(s1r, d3r)); _mm512_storeu_pd(&cur_out_im[o1], _mm512_add_pd(s1i, d3i));
                    _mm512_storeu_pd(&cur_out_re[o2], _mm512_sub_pd(s0r, s2r)); _mm512_storeu_pd(&cur_out_im[o2], _mm512_sub_pd(s0i, s2i));
                    _mm512_storeu_pd(&cur_out_re[o3], _mm512_sub_pd(s1r, d3r)); _mm512_storeu_pd(&cur_out_im[o3], _mm512_sub_pd(s1i, d3i));
                }
            }
            else if (stride == 1 && m >= 8) {
                for (size_t j = 0; j < m; j += 8) {
                    __m512d v_w1r = _mm512_loadu_pd(&sw1r[j]);
                    __m512d v_w1i = _mm512_mul_pd(_mm512_loadu_pd(&sw1i[j]), v_sign);
                    __m512d v_w2r = _mm512_loadu_pd(&sw2r[j]);
                    __m512d v_w2i = _mm512_mul_pd(_mm512_loadu_pd(&sw2i[j]), v_sign);
                    __m512d v_w3r = _mm512_loadu_pd(&sw3r[j]);
                    __m512d v_w3i = _mm512_mul_pd(_mm512_loadu_pd(&sw3i[j]), v_sign);

                    auto load8x1 = [&](const double* ptr, size_t off) {
                        return _mm512_setr_pd(ptr[(j + 0) * 4 + off], ptr[(j + 1) * 4 + off], ptr[(j + 2) * 4 + off], ptr[(j + 3) * 4 + off],
                            ptr[(j + 4) * 4 + off], ptr[(j + 5) * 4 + off], ptr[(j + 6) * 4 + off], ptr[(j + 7) * 4 + off]);
                        };
                    __m512d r0 = load8x1(cur_in_re, 0); __m512d i0v = load8x1(cur_in_im, 0);
                    __m512d r1 = load8x1(cur_in_re, 1); __m512d i1v = load8x1(cur_in_im, 1);
                    __m512d r2 = load8x1(cur_in_re, 2); __m512d i2v = load8x1(cur_in_im, 2);
                    __m512d r3 = load8x1(cur_in_re, 3); __m512d i3v = load8x1(cur_in_im, 3);

                    __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i));
                    __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                    __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i));
                    __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r));
                    __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i));
                    __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r));
                    __m512d s0r = _mm512_add_pd(r0, t2r), s0i = _mm512_add_pd(i0v, t2i);
                    __m512d s1r = _mm512_sub_pd(r0, t2r), s1i = _mm512_sub_pd(i0v, t2i);
                    __m512d s2r = _mm512_add_pd(t1r, t3r), s2i = _mm512_add_pd(t1i, t3i);
                    __m512d s3r = _mm512_sub_pd(t1r, t3r), s3i = _mm512_sub_pd(t1i, t3i);
                    __m512d d3r = _mm512_mul_pd(s3i, v_sign), d3i = _mm512_mul_pd(s3r, v_neg_sign);

                    size_t o0 = j; size_t o1 = o0 + m; size_t o2 = o1 + m; size_t o3 = o2 + m;
                    _mm512_storeu_pd(&cur_out_re[o0], _mm512_add_pd(s0r, s2r)); _mm512_storeu_pd(&cur_out_im[o0], _mm512_add_pd(s0i, s2i));
                    _mm512_storeu_pd(&cur_out_re[o1], _mm512_add_pd(s1r, d3r)); _mm512_storeu_pd(&cur_out_im[o1], _mm512_add_pd(s1i, d3i));
                    _mm512_storeu_pd(&cur_out_re[o2], _mm512_sub_pd(s0r, s2r)); _mm512_storeu_pd(&cur_out_im[o2], _mm512_sub_pd(s0i, s2i));
                    _mm512_storeu_pd(&cur_out_re[o3], _mm512_sub_pd(s1r, d3r)); _mm512_storeu_pd(&cur_out_im[o3], _mm512_sub_pd(s1i, d3i));
                }
            }
            else {
                for (size_t j = 0; j < m; j++) {
                    size_t tw = j * stride;
                    double w1r = sw1r[tw], w1i = inverse ? -sw1i[tw] : sw1i[tw];
                    double w2r = sw2r[tw], w2i = inverse ? -sw2i[tw] : sw2i[tw];
                    double w3r = sw3r[tw], w3i = inverse ? -sw3i[tw] : sw3i[tw];

                    for (size_t k = 0; k < stride; ++k) {
                        size_t i0 = k + j * 4 * stride, i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
                        double r0 = cur_in_re[i0], i0v = cur_in_im[i0], r1 = cur_in_re[i1], i1v = cur_in_im[i1];
                        double r2 = cur_in_re[i2], i2v = cur_in_im[i2], r3 = cur_in_re[i3], i3v = cur_in_im[i3];
                        double t1r = r1 * w1r - i1v * w1i, t1i = r1 * w1i + i1v * w1r;
                        double t2r = r2 * w2r - i2v * w2i, t2i = r2 * w2i + i2v * w2r;
                        double t3r = r3 * w3r - i3v * w3i, t3i = r3 * w3i + i3v * w3r;
                        double s0r = r0 + t2r, s0i = i0v + t2i, s1r = r0 - t2r, s1i = i0v - t2i;
                        double s2r = t1r + t3r, s2i = t1i + t3i, s3r = t1r - t3r, s3i = t1i - t3i;
                        double d3r = s3i * sign, d3i = -s3r * sign;
                        size_t o0 = k + j * stride, o1 = o0 + m * stride, o2 = o1 + m * stride, o3 = o2 + m * stride;
                        cur_out_re[o0] = s0r + s2r; cur_out_im[o0] = s0i + s2i;
                        cur_out_re[o1] = s1r + d3r; cur_out_im[o1] = s1i + d3i;
                        cur_out_re[o2] = s0r - s2r; cur_out_im[o2] = s0i - s2i;
                        cur_out_re[o3] = s1r - d3r; cur_out_im[o3] = s1i - d3i;
                    }
                }
            }

            m <<= 2; 
        }
        // ===============================================================
        // RADIX-2 
        // ===============================================================
        else if (remaining == 2) {
            // We use setr_epi64 (natural order of elements: 0, 1, 2... 7)
// This guarantees no hardware errors with the permutex2var instruction!
            const __m512i idx_even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
            const __m512i idx_odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);

            // We define the sign multiplier outside the loop for IFFT
            double sign_inv = inverse ? -1.0 : 1.0;
            const __m512d v_sign_inv = _mm512_set1_pd(sign_inv);

            size_t j = 0;
            for (; j + 7 < m; j += 8) {
                // Smooth reading of 8 scales at once
                __m512d v_w1r = _mm512_loadu_pd(&sw1r[j]);
                __m512d v_w1i = _mm512_mul_pd(_mm512_loadu_pd(&sw1i[j]), v_sign_inv);

                size_t i0 = j * 2;

                // Prefetch - we warn the L1/L2 cache well in advance
                size_t prefetch_offset = 128;
                _mm_prefetch((const char*)&cur_in_re[i0 + prefetch_offset], _MM_HINT_NTA);
                _mm_prefetch((const char*)&cur_in_im[i0 + prefetch_offset], _MM_HINT_NTA);

                // Loading 16 inputs at once 
                __m512d re_a = _mm512_loadu_pd(&cur_in_re[i0]);
                __m512d re_b = _mm512_loadu_pd(&cur_in_re[i0 + 8]);
                __m512d im_a = _mm512_loadu_pd(&cur_in_im[i0]);
                __m512d im_b = _mm512_loadu_pd(&cur_in_im[i0 + 8]);

                
                __m512d r0 = _mm512_permutex2var_pd(re_a, idx_even, re_b);
                __m512d r1 = _mm512_permutex2var_pd(re_a, idx_odd, re_b);
                __m512d i0v = _mm512_permutex2var_pd(im_a, idx_even, im_b);
                __m512d i1v = _mm512_permutex2var_pd(im_a, idx_odd, im_b);

                // Pure Vector Butterfly Mathematics
                __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i));
                __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));

                __m512d o0_re = _mm512_add_pd(r0, t1r);
                __m512d o0_im = _mm512_add_pd(i0v, t1i);
                __m512d o1_re = _mm512_sub_pd(r0, t1r);
                __m512d o1_im = _mm512_sub_pd(i0v, t1i);

                // RAM writes
                size_t o0 = j;
                size_t o1 = j + m;
                _mm512_storeu_pd(&cur_out_re[o0], o0_re);
                _mm512_storeu_pd(&cur_out_im[o0], o0_im);
                _mm512_storeu_pd(&cur_out_re[o1], o1_re);
                _mm512_storeu_pd(&cur_out_im[o1], o1_im);
            }

            // Loop tail (scalar fallback in case of leftovers)
            for (; j < m; j++) {
                double w1r = sw1r[j];
                double w1i = sw1i[j] * sign_inv;

                size_t i0 = j * 2, i1 = i0 + 1;
                double r0 = cur_in_re[i0], i0v = cur_in_im[i0];
                double r1 = cur_in_re[i1], i1v = cur_in_im[i1];

                double t1r = r1 * w1r - i1v * w1i;
                double t1i = r1 * w1i + i1v * w1r;

                size_t o0 = j, o1 = j + m;
                cur_out_re[o0] = r0 + t1r;
                cur_out_im[o0] = i0v + t1i;
                cur_out_re[o1] = r0 - t1r;
                cur_out_im[o1] = i0v - t1i;
            }
            m <<= 1;
        }

        std::swap(cur_in_re, cur_out_re);
        std::swap(cur_in_im, cur_out_im);
    }

    return cur_in_re;
}

// ---------------------------------------------------------
// CACHE-FRIENDLY BAILEY 2D
// ---------------------------------------------------------
// =====================================================================
// TWO-PASS BAILEY: transposes built into the FFT passes (2 passes over memory instead of 5).
//
// "Column" pass: matrix src with R rows and C columns (row = C consecutive doubles, re/im separate).
// For every block of 8 columns:
//   1) load 8 rows x 8 columns at once and transpose 8x8 in registers -> local buffer of 8 rows of length R,
//   2) (optional twiddle) multiply element (r, col) by w^(r*col) (S_j * U_i * V_ij scheme as in the transpose leaf),
//   3) length-R FFT on each of the 8 rows (Stockham radix-4/2, the whole row in L1),
//   4) store: ROWS mode   -> dst[col*R + k]  (rows, same layout as the old row_fft_pass),
//             TRANSP mode -> dst[k*C + col]  (8x8 transpose in registers, 64 B stores).
// Blocks are independent, so they are distributed among threads. LOTR_BAILEY_OLD=1 enables the old path (5 passes).
// =====================================================================
static bool bailey_old_enabled() {
    static const bool v = []() { const char* e = std::getenv("LOTR_BAILEY_OLD"); return (e && e[0] == '1'); }();
    return v;
}

// Two-pass Bailey vs old (5 passes). 2026-10-08: with real FFT, fusion and the new unpack, two-pass wins
// at 2^18 too (6 threads, CCD1: loop .480 vs .50 ms/iter). 2^19 unchanged (Stockham or old - to be measured).
// LOTR_BAILEY_NEW=1 / LOTR_BAILEY_OLD=1 force the choice.
static bool bailey_twopass_for(unsigned p) {
    if (bailey_old_enabled()) return false;
    static const bool force_new = (std::getenv("LOTR_BAILEY_NEW") != nullptr);
    if (force_new) return true;
    return p != 19;
}
static void bailey_colfft_pass(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t R, size_t C, bool twiddle, bool transposed_out, FFTContext& ctx, bool inverse,
    const std::vector<double>& w1r, const std::vector<double>& w1i,
    const std::vector<double>& w2r, const std::vector<double>& w2i,
    const std::vector<double>& w3r, const std::vector<double>& w3i) {

    const size_t N = ctx.fft_len;
    const double* const tw_re = &ctx.w1_re[0];
    const double* const tw_im = &ctx.w1_im[0];
    const Tw2 tw2(ctx);
    const __m512d v_tsign = _mm512_set1_pd(inverse ? -1.0 : 1.0);

#pragma omp parallel
    {
        // thread-local buffers: 8 rows of length R: FFT input and output (re and im)
        static thread_local double* scr = nullptr;
        static thread_local size_t scr_cap = 0;
        // Buffer rows are padded (pitch R+16) so that the 8 rows do not map to the same L1 set (4 KB stride)
        const size_t P = R + 16;
        const size_t need = 32 * P;
        if (scr_cap < need) {
            std::free(scr);
            scr = static_cast<double*>(std::aligned_alloc(64, ((need * sizeof(double) + 63) / 64) * 64));
            scr_cap = need;
        }
        double* const in_re = scr;
        double* const in_im = scr + 8 * P;
        double* const out_re = scr + 16 * P;
        double* const out_im = scr + 24 * P;

        // V[bb][t] = w^(t*bb), t,bb < 8 (index t*bb <= 49 < N); computed once per thread for a given N
        alignas(64) static thread_local double V_re[8][8];
        alignas(64) static thread_local double V_im[8][8];
        static thread_local size_t V_N = 0;
        if (V_N != N) {
            for (size_t bb = 0; bb < 8; ++bb)
                for (size_t t = 0; t < 8; ++t) { V_re[bb][t] = tw_re[t * bb]; V_im[bb][t] = tw_im[t * bb]; }
            V_N = N;
        }

#pragma omp for schedule(static)
        for (size_t cb = 0; cb < C / 8; ++cb) {
            const size_t c0 = cb * 8;

            // ---- twiddle: P_bb[t] = U[t] * V[bb][t], U[t] = w^(t*c0) ----
            __m512d Pr[8], Pi[8];
            if (twiddle) {
                alignas(64) double u_re[8];
                alignas(64) double u_im[8];
                for (size_t t = 0; t < 8; ++t) {
                    const size_t idx = (t * c0) & (N - 1);
                    u_re[t] = tw_re[idx]; u_im[t] = tw_im[idx];
                }
                const __m512d Ur = _mm512_load_pd(u_re), Ui = _mm512_load_pd(u_im);
                for (size_t bb = 0; bb < 8; ++bb) {
                    const __m512d Vr = _mm512_load_pd(V_re[bb]), Vi = _mm512_load_pd(V_im[bb]);
                    Pr[bb] = _mm512_fmsub_pd(Ur, Vr, _mm512_mul_pd(Ui, Vi));
                    Pi[bb] = _mm512_fmadd_pd(Ur, Vi, _mm512_mul_pd(Ui, Vr));
                }
            }

            // ---- gather: 8 rows x 8 columns -> 8x8 transpose -> 8 rows of length R ----
            for (size_t r = 0; r < R; r += 8) {
                // Prefetch 16 rows ahead (a C*8 B stride crosses pages - the hardware prefetcher does not help here)
                if (r + 16 < R) {
#pragma GCC unroll 8
                    for (int t = 0; t < 8; ++t) {
                        _mm_prefetch((const char*)(src_re + (r + 16 + t) * C + c0), _MM_HINT_T1);   // into L2: a 4 KB stride would overflow an L1 set
                        _mm_prefetch((const char*)(src_im + (r + 16 + t) * C + c0), _MM_HINT_T1);
                    }
                }
                __m512d vr[8], vi[8];
#pragma GCC unroll 8
                for (int t = 0; t < 8; ++t) {
                    vr[t] = _mm512_loadu_pd(src_re + (r + t) * C + c0);
                    vi[t] = _mm512_loadu_pd(src_im + (r + t) * C + c0);
                }
                st_transpose8x8(vr);
                st_transpose8x8(vi);   // now vr[bb][t] = element (row r+t, column c0+bb)
                if (twiddle) {
#pragma GCC unroll 8
                    for (int bb = 0; bb < 8; ++bb) {
                        double s_re, s_im;
                        tw2.get((r * (c0 + bb)) & (N - 1), s_re, s_im);   // S = w^(r*col)
                        const __m512d Sr = _mm512_set1_pd(s_re);
                        const __m512d Si = _mm512_set1_pd(s_im);
                        const __m512d wr = _mm512_fmsub_pd(Pr[bb], Sr, _mm512_mul_pd(Pi[bb], Si));
                        const __m512d wi = _mm512_mul_pd(_mm512_fmadd_pd(Pr[bb], Si, _mm512_mul_pd(Pi[bb], Sr)), v_tsign);
                        const __m512d xr = _mm512_fmsub_pd(vr[bb], wr, _mm512_mul_pd(vi[bb], wi));
                        const __m512d xi = _mm512_fmadd_pd(vr[bb], wi, _mm512_mul_pd(vi[bb], wr));
                        vr[bb] = xr; vi[bb] = xi;
                    }
                }
#pragma GCC unroll 8
                for (int bb = 0; bb < 8; ++bb) {
                    _mm512_store_pd(in_re + bb * P + r, vr[bb]);
                    _mm512_store_pd(in_im + bb * P + r, vi[bb]);
                }
            }

            // ---- length-R FFT of each of the 8 rows ----
            const double* res_re[8];
            const double* res_im[8];
            for (size_t bb = 0; bb < 8; ++bb) {
                double* a_re = in_re + bb * P; double* a_im = in_im + bb * P;
                double* o_re = out_re + bb * P; double* o_im = out_im + bb * P;
                double* res = fft_stockham_radix4_blocked(a_re, a_im, o_re, o_im, R, w1r, w1i, w2r, w2i, w3r, w3i, inverse);
                const bool moved = (res != a_re);
                res_re[bb] = moved ? o_re : a_re;
                res_im[bb] = moved ? o_im : a_im;
            }

            // ---- store ----
            if (!transposed_out) {
                for (size_t bb = 0; bb < 8; ++bb) {
                    std::memcpy(dst_re + (c0 + bb) * R, res_re[bb], R * sizeof(double));
                    std::memcpy(dst_im + (c0 + bb) * R, res_im[bb], R * sizeof(double));
                }
            }
            else {
                for (size_t k = 0; k < R; k += 8) {
                    __m512d xr[8], xi[8];
#pragma GCC unroll 8
                    for (int bb = 0; bb < 8; ++bb) {
                        xr[bb] = _mm512_load_pd(res_re[bb] + k);
                        xi[bb] = _mm512_load_pd(res_im[bb] + k);
                    }
                    st_transpose8x8(xr);
                    st_transpose8x8(xi);   // xr[t][bb] = result (row bb, index k+t) -> dst[(k+t)*C + c0 + bb]
#pragma GCC unroll 8
                    for (int t = 0; t < 8; ++t) {
                        _mm512_storeu_pd(dst_re + (k + t) * C + c0, xr[t]);
                        _mm512_storeu_pd(dst_im + (k + t) * C + c0, xi[t]);
                    }
                }
            }
        }
    }
}
// =====================================================================
// LENGTH-R FFT FOR 8 COLUMNS AT ONCE ("batch-8"): vector r (8 doubles at offset 8*r) = element r of the eight columns.
// Stockham radix-8 (radix-4 or radix-2 tail), same index formulas as fft_stockham_radix4_blocked:
//   input  k + j*r*stride + q*stride,  twiddle W_R^(q*j*stride),  output k + j*stride + p*m*stride.
// Twiddles are shared by the 8 columns (broadcast), so no stage needs in-register shuffles.
// w_re/w_im: table of R-th roots of unity (length R, forward direction). Returns true when the result is in b.
// =====================================================================
static bool fft_batch8(double* __restrict a_re, double* __restrict a_im, double* __restrict b_re, double* __restrict b_im,
                       size_t R, const double* w_re0, const double* w_im0, bool inverse, size_t wstride = 1) {
    // wstride > 1: root table of a larger order (R*wstride), we use every wstride-th element
    struct Strided {
        const double* p; size_t s;
        double operator[](size_t i) const { return p[i * s]; }
    };
    const Strided w_re{ w_re0, wstride }, w_im{ w_im0, wstride };
    const double sg = inverse ? -1.0 : 1.0;
    const __m512d v_sign = _mm512_set1_pd(sg);
    const __m512d v_neg_sign = _mm512_set1_pd(-sg);
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    const __m512d v_neg_C = _mm512_set1_pd(-0.70710678118654752440);

    double* ir = a_re; double* ii = a_im;
    double* orr = b_re; double* oi = b_im;
    bool in_b = false;
    size_t m = 1;
    while (m < R) {
        const size_t rem = R / m;
        if (rem >= 8) {
            const size_t stride = rem >> 3;
            const size_t os = m * stride;
            for (size_t j = 0; j < m; ++j) {
                const size_t tw = j * stride;
                __m512d wr[8], wi[8];
                wr[0] = _mm512_set1_pd(1.0); wi[0] = _mm512_setzero_pd();
#pragma GCC unroll 7
                for (int q = 1; q < 8; ++q) {
                    wr[q] = _mm512_set1_pd(w_re[q * tw]);
                    wi[q] = _mm512_set1_pd(sg * w_im[q * tw]);
                }
                const size_t ib = j * 8 * stride;
                const size_t ob = j * stride;
                for (size_t k = 0; k < stride; ++k) {
                    __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
                    for (int q = 0; q < 8; ++q) {
                        r[q] = _mm512_load_pd(ir + 8 * (ib + k + q * stride));
                        i[q] = _mm512_load_pd(ii + 8 * (ib + k + q * stride));
                    }
                    st_radix8_vec(r, i, wr, wi, yr, yi, v_sign, v_neg_sign, v_C, v_neg_C);
#pragma GCC unroll 8
                    for (int p = 0; p < 8; ++p) {
                        _mm512_store_pd(orr + 8 * (ob + k + p * os), yr[p]);
                        _mm512_store_pd(oi + 8 * (ob + k + p * os), yi[p]);
                    }
                }
            }
            m <<= 3;
        }
        else if (rem == 4) {
            // radix-4, stride = 1
            for (size_t j = 0; j < m; ++j) {
                const __m512d w1r = _mm512_set1_pd(w_re[j]), w1i = _mm512_set1_pd(sg * w_im[j]);
                const __m512d w2r = _mm512_set1_pd(w_re[2 * j]), w2i = _mm512_set1_pd(sg * w_im[2 * j]);
                const __m512d w3r = _mm512_set1_pd(w_re[3 * j]), w3i = _mm512_set1_pd(sg * w_im[3 * j]);
                const size_t ib = 4 * j;
                const __m512d r0 = _mm512_load_pd(ir + 8 * ib),       i0 = _mm512_load_pd(ii + 8 * ib);
                const __m512d r1 = _mm512_load_pd(ir + 8 * (ib + 1)), i1 = _mm512_load_pd(ii + 8 * (ib + 1));
                const __m512d r2 = _mm512_load_pd(ir + 8 * (ib + 2)), i2 = _mm512_load_pd(ii + 8 * (ib + 2));
                const __m512d r3 = _mm512_load_pd(ir + 8 * (ib + 3)), i3 = _mm512_load_pd(ii + 8 * (ib + 3));
                const __m512d t1r = _mm512_fmsub_pd(r1, w1r, _mm512_mul_pd(i1, w1i)), t1i = _mm512_fmadd_pd(r1, w1i, _mm512_mul_pd(i1, w1r));
                const __m512d t2r = _mm512_fmsub_pd(r2, w2r, _mm512_mul_pd(i2, w2i)), t2i = _mm512_fmadd_pd(r2, w2i, _mm512_mul_pd(i2, w2r));
                const __m512d t3r = _mm512_fmsub_pd(r3, w3r, _mm512_mul_pd(i3, w3i)), t3i = _mm512_fmadd_pd(r3, w3i, _mm512_mul_pd(i3, w3r));
                const __m512d s0r = _mm512_add_pd(r0, t2r), s0i = _mm512_add_pd(i0, t2i);
                const __m512d s1r = _mm512_sub_pd(r0, t2r), s1i = _mm512_sub_pd(i0, t2i);
                const __m512d s2r = _mm512_add_pd(t1r, t3r), s2i = _mm512_add_pd(t1i, t3i);
                const __m512d s3r = _mm512_sub_pd(t1r, t3r), s3i = _mm512_sub_pd(t1i, t3i);
                const __m512d d3r = _mm512_mul_pd(s3i, v_sign), d3i = _mm512_mul_pd(s3r, v_neg_sign);
                _mm512_store_pd(orr + 8 * j,           _mm512_add_pd(s0r, s2r)); _mm512_store_pd(oi + 8 * j,           _mm512_add_pd(s0i, s2i));
                _mm512_store_pd(orr + 8 * (j + m),     _mm512_add_pd(s1r, d3r)); _mm512_store_pd(oi + 8 * (j + m),     _mm512_add_pd(s1i, d3i));
                _mm512_store_pd(orr + 8 * (j + 2 * m), _mm512_sub_pd(s0r, s2r)); _mm512_store_pd(oi + 8 * (j + 2 * m), _mm512_sub_pd(s0i, s2i));
                _mm512_store_pd(orr + 8 * (j + 3 * m), _mm512_sub_pd(s1r, d3r)); _mm512_store_pd(oi + 8 * (j + 3 * m), _mm512_sub_pd(s1i, d3i));
            }
            m <<= 2;
        }
        else {
            // radix-2, stride = 1
            for (size_t j = 0; j < m; ++j) {
                const __m512d wr = _mm512_set1_pd(w_re[j]), wi = _mm512_set1_pd(sg * w_im[j]);
                const __m512d r0 = _mm512_load_pd(ir + 8 * (2 * j)),     i0 = _mm512_load_pd(ii + 8 * (2 * j));
                const __m512d r1 = _mm512_load_pd(ir + 8 * (2 * j + 1)), i1 = _mm512_load_pd(ii + 8 * (2 * j + 1));
                const __m512d tr = _mm512_fmsub_pd(r1, wr, _mm512_mul_pd(i1, wi));
                const __m512d ti = _mm512_fmadd_pd(r1, wi, _mm512_mul_pd(i1, wr));
                _mm512_store_pd(orr + 8 * j,       _mm512_add_pd(r0, tr)); _mm512_store_pd(oi + 8 * j,       _mm512_add_pd(i0, ti));
                _mm512_store_pd(orr + 8 * (j + m), _mm512_sub_pd(r0, tr)); _mm512_store_pd(oi + 8 * (j + m), _mm512_sub_pd(i0, ti));
            }
            m <<= 1;
        }
        std::swap(ir, orr);
        std::swap(ii, oi);
        in_b = !in_b;
    }
    return in_b;
}

// Number of fft_batch8 stages for length R (radix-8, radix-4/2 tail).
static int batch8_stages(size_t R) {
    int s = 0;
    for (size_t m = 1; m < R; ++s) {
        const size_t rem = R / m;
        m <<= (rem >= 8) ? 3 : ((rem == 4) ? 2 : 1);
    }
    return s;
}

// From this column length on, batch-8 computes the FFT in two levels (a block of 8 columns x R no longer fits in L1).
static inline bool batch8_two_level(size_t R) { return R >= 512; }

// Position of vector X[K] in the "blocked" output layout of fft_batch8_2l (R2 = R/8): block p = K & 7, position K >> 3.
static inline size_t batch8_posL(size_t K, size_t R2) { return (K & 7) * R2 + (K >> 3); }

// =====================================================================
// TWO-LEVEL batch-8 FFT (DIF), R = 8 * R2:
//   1) for k < R2: y_p = DFT8 over q of x[k + R2*q], then y_p *= W_R^(p*k)  -> block p, position k
//   2) in each of the 8 blocks a length-R2 FFT (block ~8-16 KB, computed in L1) -> X[p + 8*k'] at position k' of block p.
// Output always in out_*, in the blocked layout (batch8_posL). in_L = true: the input is in the blocked layout too (e.g. the
// spectrum after the square in the fusion). in_* must not overlap out_* or tmp_*. w_re/w_im: R-th roots of unity (length R).
// =====================================================================
static void fft_batch8_2l(const double* __restrict in_re, const double* __restrict in_im, bool in_L,
                          double* __restrict out_re, double* __restrict out_im, double* __restrict tmp_re, double* __restrict tmp_im,
                          size_t R, const double* w_re, const double* w_im, bool inverse) {
    const size_t R2 = R >> 3;
    const double sg = inverse ? -1.0 : 1.0;
    const __m512d v_sign = _mm512_set1_pd(sg);
    const __m512d v_neg_sign = _mm512_set1_pd(-sg);
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    const __m512d v_neg_C = _mm512_set1_pd(-0.70710678118654752440);
    __m512d one_r[8], one_i[8];
    for (int q = 0; q < 8; ++q) { one_r[q] = _mm512_set1_pd(1.0); one_i[q] = _mm512_setzero_pd(); }

    // stage 1 writes to the buffer from which the sub-FFTs (ping-pong) end up in out_*
    const bool odd = (batch8_stages(R2) & 1) != 0;
    double* const s1_re = odd ? tmp_re : out_re;
    double* const s1_im = odd ? tmp_im : out_im;
    double* const s2_re = odd ? out_re : tmp_re;
    double* const s2_im = odd ? out_im : tmp_im;

    for (size_t k = 0; k < R2; ++k) {
        __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int q = 0; q < 8; ++q) {
            const size_t nidx = k + R2 * (size_t)q;
            const size_t pos = in_L ? batch8_posL(nidx, R2) : nidx;
            r[q] = _mm512_load_pd(in_re + 8 * pos);
            i[q] = _mm512_load_pd(in_im + 8 * pos);
        }
        st_radix8_vec(r, i, one_r, one_i, yr, yi, v_sign, v_neg_sign, v_C, v_neg_C);
        _mm512_store_pd(s1_re + 8 * k, yr[0]);
        _mm512_store_pd(s1_im + 8 * k, yi[0]);
#pragma GCC unroll 7
        for (int p = 1; p < 8; ++p) {
            const __m512d wr = _mm512_set1_pd(w_re[(size_t)p * k]);
            const __m512d wi = _mm512_set1_pd(sg * w_im[(size_t)p * k]);
            _mm512_store_pd(s1_re + 8 * (p * R2 + k), _mm512_fmsub_pd(yr[p], wr, _mm512_mul_pd(yi[p], wi)));
            _mm512_store_pd(s1_im + 8 * (p * R2 + k), _mm512_fmadd_pd(yr[p], wi, _mm512_mul_pd(yi[p], wr)));
        }
    }
    for (size_t p = 0; p < 8; ++p) {
        fft_batch8(s1_re + 8 * p * R2, s1_im + 8 * p * R2, s2_re + 8 * p * R2, s2_im + 8 * p * R2,
                   R2, w_re, w_im, inverse, 8);
    }
}

// LOTR_COLFFT_OLD=1: old column passes (8x8 transpose + 8 separate radix-4 FFTs) - for A/B comparisons
static bool colfft_old_enabled() {
    static const bool v = (std::getenv("LOTR_COLFFT_OLD") != nullptr);
    return v;
}

// Batch-8 kernel for columns of length R <= threshold (no limit by default; R >= 512 computed in two levels, fft_batch8_2l).
// Measured 2026-10-08: 2^16 (R=256) 1 thread FFT .359 -> .285 ms; 2^18 (R=512) 6 threads single-level version .573
// vs old kernel .480 (an 8 x 512 block does not fit in L1) - hence the two-level version.
// LOTR_BATCH8_MAXR=<R> changes the threshold (256 = long columns with the old kernel, 0 = batch-8 nowhere).
static bool use_batch8(size_t R) {
    if (colfft_old_enabled()) return false;
    static const size_t maxR = []() {
        const char* e = std::getenv("LOTR_BATCH8_MAXR");
        return e ? (size_t)std::strtoull(e, nullptr, 10) : ~(size_t)0;
    }();
    return R <= maxR;
}

// Thread buffer: per block of 8 columns, 3 buffers (a, b, c) of 8*R doubles for re and im (48*R doubles) + 'extra'; 64 B aligned.
static double* batch8_scratch(size_t R, size_t blocks, size_t extra) {
    static thread_local double* scr = nullptr;
    static thread_local size_t cap = 0;
    const size_t need = blocks * 48 * R + extra;
    if (cap < need) {
        std::free(scr);
        scr = static_cast<double*>(std::aligned_alloc(64, ((need * sizeof(double) + 63) / 64) * 64));
        cap = need;
    }
    return scr;
}

// Gather 8 columns (cstart..cstart+7) from matrix src (R rows x C columns) into vectors g (8*r), optionally with twiddle
// w^(r*col) (conjugated for inverse). special: lanes 1..7 shifted to 0..6, lane 7 = column spc (fusion block M for b = 0).
static inline void batch8_gather(const double* __restrict src_re, const double* __restrict src_im, size_t R, size_t C,
                                 size_t cstart, bool special, size_t spc, bool twiddle, bool inverse,
                                 const FFTContext& ctx, const Tw2& tw2, double* __restrict g_re, double* __restrict g_im) {
    const size_t N = ctx.fft_len;
    const double* const G_re = ctx.lane_g_re.data();
    const double* const G_im = ctx.lane_g_im.data();
    const __m512i shift1 = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 7);
    const double isg = inverse ? -1.0 : 1.0;
    // with special, the columns of lanes 0..6 are cstart+1 .. cstart+7: twiddle computed from cstart+1
    const size_t c_base = special ? cstart + 1 : cstart;
    for (size_t r = 0; r < R; ++r) {
        // Prefetch into L2 (T1): rows every C*8 B with C a power of two map to one L1 set (12 ways),
        // so a prefetch into L1 would evict lines before use. C = row stride (with or without padding).
        if (r + 16 < R) {
            _mm_prefetch((const char*)(src_re + (r + 16) * C + cstart), _MM_HINT_T1);
            _mm_prefetch((const char*)(src_im + (r + 16) * C + cstart), _MM_HINT_T1);
        }
        __m512d vr = _mm512_loadu_pd(src_re + r * C + cstart);
        __m512d vi = _mm512_loadu_pd(src_im + r * C + cstart);
        if (special) {
            vr = _mm512_mask_mov_pd(_mm512_permutexvar_pd(shift1, vr), 0x80, _mm512_set1_pd(src_re[r * C + spc]));
            vi = _mm512_mask_mov_pd(_mm512_permutexvar_pd(shift1, vi), 0x80, _mm512_set1_pd(src_im[r * C + spc]));
        }
        if (twiddle) {
            double s_re, s_im;
            tw2.get((r * c_base) & (N - 1), s_re, s_im);           // w^(r*c_base)
            const __m512d gr = _mm512_loadu_pd(G_re + 8 * r);     // w^(r*b) (std::vector: no 64 B alignment guarantee)
            const __m512d gi = _mm512_loadu_pd(G_im + 8 * r);
            const __m512d Sr = _mm512_set1_pd(s_re), Si = _mm512_set1_pd(s_im);
            __m512d wr = _mm512_fmsub_pd(gr, Sr, _mm512_mul_pd(gi, Si));
            __m512d wi = _mm512_fmadd_pd(gr, Si, _mm512_mul_pd(gi, Sr));
            if (special) {
                double e_re, e_im;
                tw2.get((r * spc) & (N - 1), e_re, e_im);
                wr = _mm512_mask_mov_pd(wr, 0x80, _mm512_set1_pd(e_re));
                wi = _mm512_mask_mov_pd(wi, 0x80, _mm512_set1_pd(e_im));
            }
            wi = _mm512_mul_pd(wi, _mm512_set1_pd(isg));
            const __m512d xr = _mm512_fmsub_pd(vr, wr, _mm512_mul_pd(vi, wi));
            const __m512d xi = _mm512_fmadd_pd(vr, wi, _mm512_mul_pd(vi, wr));
            vr = xr; vi = xi;
        }
        _mm512_store_pd(g_re + 8 * r, vr);
        _mm512_store_pd(g_im + 8 * r, vi);
    }
}

// Store the 8 result columns as rows: dst[cols[b]*R + k] (8x8 transpose in registers).
// L = true: result in the blocked layout of fft_batch8_2l (vector k+t is at position t*R/8 + k/8).
static inline void batch8_store_rows(const double* __restrict v_re, const double* __restrict v_im, size_t R, const size_t* cols,
                                     double* __restrict dst_re, double* __restrict dst_im, bool L = false, size_t dp = 0) {
    const size_t R2 = R >> 3;
    const size_t pitch = dp ? dp : R;   // destination row stride (padded or R)
    for (size_t k = 0; k < R; k += 8) {
        __m512d xr[8], xi[8];
#pragma GCC unroll 8
        for (int t = 0; t < 8; ++t) {
            const size_t pos = L ? ((size_t)t * R2 + (k >> 3)) : (k + t);
            xr[t] = _mm512_load_pd(v_re + 8 * pos);
            xi[t] = _mm512_load_pd(v_im + 8 * pos);
        }
        st_transpose8x8(xr);
        st_transpose8x8(xi);   // xr[b][t] = column cols[b], index k+t
#pragma GCC unroll 8
        for (int b = 0; b < 8; ++b) {
            _mm512_storeu_pd(dst_re + cols[b] * pitch + k, xr[b]);
            _mm512_storeu_pd(dst_im + cols[b] * pitch + k, xi[b]);
        }
    }
}

// New column pass (batch-8): same result as bailey_colfft_pass, without a transpose on load
// and (in TRANSP mode) on store. w1r/w1i: R-th roots of unity (r_w1* or c_w1*).
static void bailey_colfft_pass8(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t R, size_t C, bool twiddle, bool transposed_out, FFTContext& ctx, bool inverse,
    const std::vector<double>& w1r, const std::vector<double>& w1i, size_t sp = 0, size_t dp = 0) {
    const Tw2 tw2(ctx);
    const bool two = batch8_two_level(R);
    const size_t R2 = R >> 3;
    // row strides: source (R rows of C columns) and destination (TRANSP: R rows of C; ROWS: C rows of R); 0 = no padding
    if (sp == 0) sp = C;
    if (dp == 0) dp = transposed_out ? C : R;
#pragma omp parallel
    {
        double* const scr = batch8_scratch(R, 1, 0);
        double* const a_re = scr;          double* const a_im = scr + 8 * R;
        double* const b_re = scr + 16 * R; double* const b_im = scr + 24 * R;
        double* const c_re = scr + 32 * R; double* const c_im = scr + 40 * R;
#pragma omp for schedule(static)
        for (size_t cb = 0; cb < C / 8; ++cb) {
            const size_t c0 = cb * 8;
            batch8_gather(src_re, src_im, R, sp, c0, false, 0, twiddle, inverse, ctx, tw2, a_re, a_im);
            const double* res_re;
            const double* res_im;
            if (two) {
                fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, w1r.data(), w1i.data(), inverse);
                res_re = b_re; res_im = b_im;
            }
            else {
                const bool in_b = fft_batch8(a_re, a_im, b_re, b_im, R, w1r.data(), w1i.data(), inverse);
                res_re = in_b ? b_re : a_re;
                res_im = in_b ? b_im : a_im;
            }
            if (transposed_out) {
                if (two) {
                    // position p*R2 + k' holds X[p + 8k']: read sequentially, write to row K = p + 8k'
                    for (size_t p = 0; p < 8; ++p) {
                        for (size_t kk = 0; kk < R2; ++kk) {
                            const size_t K = p + 8 * kk;
                            _mm512_storeu_pd(dst_re + K * dp + c0, _mm512_load_pd(res_re + 8 * (p * R2 + kk)));
                            _mm512_storeu_pd(dst_im + K * dp + c0, _mm512_load_pd(res_im + 8 * (p * R2 + kk)));
                        }
                    }
                }
                else {
                    for (size_t k = 0; k < R; ++k) {
                        _mm512_storeu_pd(dst_re + k * dp + c0, _mm512_load_pd(res_re + 8 * k));
                        _mm512_storeu_pd(dst_im + k * dp + c0, _mm512_load_pd(res_im + 8 * k));
                    }
                }
            }
            else {
                size_t cols[8];
                for (size_t b = 0; b < 8; ++b) cols[b] = c0 + b;
                batch8_store_rows(res_re, res_im, R, cols, dst_re, dst_im, two, dp);
            }
        }
    }
}

// Choice of the column pass implementation (LOTR_COLFFT_OLD=1 = old).
static void bailey_colpass(const double* src_re, const double* src_im, double* dst_re, double* dst_im,
    size_t R, size_t C, bool twiddle, bool transposed_out, FFTContext& ctx, bool inverse,
    const std::vector<double>& w1r, const std::vector<double>& w1i,
    const std::vector<double>& w2r, const std::vector<double>& w2i,
    const std::vector<double>& w3r, const std::vector<double>& w3i) {
    if (use_batch8(R)) {
        bailey_colfft_pass8(src_re, src_im, dst_re, dst_im, R, C, twiddle, transposed_out, ctx, inverse, w1r, w1i);
    }
    else {
        bailey_colfft_pass(src_re, src_im, dst_re, dst_im, R, C, twiddle, transposed_out, ctx, inverse, w1r, w1i, w2r, w2i, w3r, w3i);
    }
}

// Bailey 2D dimensions (N = n1 * n2, n1 >= n2) and twiddle tables for the row FFTs (r_*, length n1) and column FFTs (c_*, length n2).
static unsigned bailey_setup(FFTContext& ctx, size_t& n1, size_t& n2) {
    size_t N = ctx.fft_len;
    unsigned p = 0;
    size_t temp_for_p = N;
    while (temp_for_p > 1) { temp_for_p >>= 1; p++; }

    if (p % 2 == 0) {
        n1 = 1ULL << (p / 2);
        n2 = 1ULL << (p / 2);
    }
    else {
        n1 = 1ULL << ((p / 2) + 1); // e.g. 512
        n2 = 1ULL << (p / 2);       // e.g. 256
    }

    if (ctx.r_w1r.size() != n1) {
        ctx.r_w1r.assign(n1, 0); ctx.r_w1i.assign(n1, 0);
        ctx.r_w2r.assign(n1, 0); ctx.r_w2i.assign(n1, 0);
        ctx.r_w3r.assign(n1, 0); ctx.r_w3i.assign(n1, 0);

        size_t tw_stride_col = n2;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; ++i) {
            size_t idx = i * tw_stride_col;
            ctx.r_w1r[i] = ctx.w1_re[idx]; ctx.r_w1i[i] = ctx.w1_im[idx];
            ctx.r_w2r[i] = ctx.w2_re[idx]; ctx.r_w2i[i] = ctx.w2_im[idx];
            ctx.r_w3r[i] = ctx.w3_re[idx]; ctx.r_w3i[i] = ctx.w3_im[idx];
        }
    }

    if (ctx.c_w1r.size() != n2) {
        ctx.c_w1r.assign(n2, 0); ctx.c_w1i.assign(n2, 0);
        ctx.c_w2r.assign(n2, 0); ctx.c_w2i.assign(n2, 0);
        ctx.c_w3r.assign(n2, 0); ctx.c_w3i.assign(n2, 0);

        size_t tw_stride_row = n1;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; ++i) {
            size_t idx = i * tw_stride_row;
            ctx.c_w1r[i] = ctx.w1_re[idx]; ctx.c_w1i[i] = ctx.w1_im[idx];
            ctx.c_w2r[i] = ctx.w2_re[idx]; ctx.c_w2i[i] = ctx.w2_im[idx];
            ctx.c_w3r[i] = ctx.w3_re[idx]; ctx.c_w3i[i] = ctx.w3_im[idx];
        }
    }

    if (ctx.tw2_N != N) {
        const unsigned sh = (p + 1) / 2;
        const size_t L = size_t(1) << sh;
        const size_t H = N >> sh;
        ctx.tw2_lo_re.assign(L, 0); ctx.tw2_lo_im.assign(L, 0);
        ctx.tw2_hi_re.assign(H, 0); ctx.tw2_hi_im.assign(H, 0);
        for (size_t i = 0; i < L; ++i) { ctx.tw2_lo_re[i] = ctx.w1_re[i]; ctx.tw2_lo_im[i] = ctx.w1_im[i]; }
        for (size_t i = 0; i < H; ++i) { ctx.tw2_hi_re[i] = ctx.w1_re[i << sh]; ctx.tw2_hi_im[i] = ctx.w1_im[i << sh]; }
        ctx.tw2_shift = sh;
        ctx.tw2_N = N;
    }

    if (ctx.lane_g_N != N) {
        ctx.lane_g_re.assign(8 * n1, 0.0);
        ctx.lane_g_im.assign(8 * n1, 0.0);
        for (size_t r = 0; r < n1; ++r) {
            for (size_t b = 0; b < 8; ++b) {
                const size_t e = (r * b) & (N - 1);
                ctx.lane_g_re[8 * r + b] = ctx.w1_re[e];
                ctx.lane_g_im[8 * r + b] = ctx.w1_im[e];
            }
        }
        ctx.lane_g_N = N;
    }
    return p;
}

void fft_bailey_2d(FFTContext& ctx, bool inverse) {
    size_t n1, n2;
    const unsigned p = bailey_setup(ctx, n1, n2);

    // Two-pass version by default (transposes built into the FFT passes); LOTR_BAILEY_OLD=1 = old path (5 passes).
    if (bailey_twopass_for(p) && (n1 % 8 == 0) && (n2 % 8 == 0)) {
        if (!inverse) {
            // A: x (n1 rows x n2 columns) -> length-n1 FFT over columns -> Y1 (n2 rows x n1)
            { LOTR_PASS(ctx, 0); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, false, false, ctx, false,
                ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i); }
            // B: Y1 (n2 x n1) -> twiddle w^(b*k1), length-n2 FFT over columns, transposed store -> natural order
            { LOTR_PASS(ctx, 1); bailey_colpass(ctx.re2, ctx.im2, ctx.re, ctx.im, n2, n1, true, true, ctx, false,
                ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i); }
        }
        else {
            // A': spectrum X (n2 rows x n1 columns) -> length-n2 FFT over columns -> Y (n1 rows x n2)
            { LOTR_PASS(ctx, 0); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1, false, false, ctx, true,
                ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i); }
            // B': Y (n1 x n2) -> conjugate twiddle w^(k1*b), length-n1 FFT over columns, transposed store -> natural order
            { LOTR_PASS(ctx, 1); bailey_colpass(ctx.re2, ctx.im2, ctx.re, ctx.im, n1, n2, true, true, ctx, true,
                ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i); }
        }
        return;
    }
    const bool legacy_rowcopy = row_memcpy_legacy();

    // One row FFT pass: `rows` rows of length `len`. When the number of Stockham stages is
    // odd, the result is in the out buffer (re2/im2) for all rows; instead of copying it back
    // row by row, the buffer pointers are swapped (the next transpose reads from ctx.re).
    auto row_fft_pass = [&](size_t rows, size_t len,
                            const std::vector<double>& w1r, const std::vector<double>& w1i,
                            const std::vector<double>& w2r, const std::vector<double>& w2i,
                            const std::vector<double>& w3r, const std::vector<double>& w3i) {
        bool result_in_out = false;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; i++) {
            double* r_in_re = &ctx.re[i * len]; double* r_in_im = &ctx.im[i * len];
            double* r_out_re = &ctx.re2[i * len]; double* r_out_im = &ctx.im2[i * len];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, len,
                w1r, w1i, w2r, w2i, w3r, w3i, inverse);
            const bool moved = (res != r_in_re);
            if (i == 0) result_in_out = moved;
            if (moved && legacy_rowcopy) {
                std::memcpy(r_in_re, r_out_re, len * sizeof(double));
                std::memcpy(r_in_im, r_out_im, len * sizeof(double));
            }
        }
        if (result_in_out && !legacy_rowcopy) {
            std::swap(ctx.re, ctx.re2);
            std::swap(ctx.im, ctx.im2);
        }
    };

    if (!inverse) {
        { LOTR_PASS(ctx, 0); transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2); }
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        { LOTR_PASS(ctx, 1); row_fft_pass(n2, n1, ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i); }

        { LOTR_PASS(ctx, 2); transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1, ctx, false); }
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        { LOTR_PASS(ctx, 3); row_fft_pass(n1, n2, ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i); }

        { LOTR_PASS(ctx, 4); transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2); }
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);
    }
    else {
        { LOTR_PASS(ctx, 0); transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1); }
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        { LOTR_PASS(ctx, 1); row_fft_pass(n1, n2, ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i); }

        { LOTR_PASS(ctx, 2); transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, ctx, true); }
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        { LOTR_PASS(ctx, 3); row_fft_pass(n2, n1, ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i); }

        { LOTR_PASS(ctx, 4); transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1); }
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);
    }
}

// Square of a spectrum pair (k, N-k) for the real FFT (formulas and description at fft_square_real): input Z[k], Z[N-k], w^k; output Z'[k], Z'[N-k].
static inline __attribute__((always_inline)) void sqr_pair_vec(
    __m512d Rk, __m512d Ik, __m512d Rj, __m512d Ij, __m512d wr, __m512d wi, __m512d v_s,
    __m512d& okr, __m512d& oki, __m512d& ojr, __m512d& oji) {
    const __m512d v_two = _mm512_set1_pd(2.0);
    const __m512d Er = _mm512_add_pd(Rk, Rj);
    const __m512d Ei = _mm512_sub_pd(Ik, Ij);
    const __m512d Or = _mm512_add_pd(Ik, Ij);
    const __m512d Oi = _mm512_sub_pd(Rj, Rk);
    const __m512d E2r = _mm512_fmsub_pd(Er, Er, _mm512_mul_pd(Ei, Ei));
    const __m512d E2i = _mm512_mul_pd(v_two, _mm512_mul_pd(Er, Ei));
    const __m512d O2r = _mm512_fmsub_pd(Or, Or, _mm512_mul_pd(Oi, Oi));
    const __m512d O2i = _mm512_mul_pd(v_two, _mm512_mul_pd(Or, Oi));
    const __m512d Epr = _mm512_add_pd(E2r, _mm512_fmsub_pd(wr, O2r, _mm512_mul_pd(wi, O2i)));
    const __m512d Epi = _mm512_add_pd(E2i, _mm512_fmadd_pd(wr, O2i, _mm512_mul_pd(wi, O2r)));
    const __m512d Opr = _mm512_mul_pd(v_two, _mm512_fmsub_pd(Er, Or, _mm512_mul_pd(Ei, Oi)));
    const __m512d Opi = _mm512_mul_pd(v_two, _mm512_fmadd_pd(Er, Oi, _mm512_mul_pd(Ei, Or)));
    okr = _mm512_mul_pd(_mm512_sub_pd(Epr, Opi), v_s);
    oki = _mm512_mul_pd(_mm512_add_pd(Epi, Opr), v_s);
    ojr = _mm512_mul_pd(_mm512_add_pd(Epr, Opi), v_s);
    oji = _mm512_mul_pd(_mm512_sub_pd(Opr, Epi), v_s);
}

static inline void sqr_pair_scalar(double Rk, double Ik, double Rj, double Ij, double wr, double wi, double s,
    double& okr, double& oki, double& ojr, double& oji) {
    const double Er = Rk + Rj, Ei = Ik - Ij;
    const double Or = Ik + Ij, Oi = Rj - Rk;
    const double E2r = Er * Er - Ei * Ei, E2i = 2.0 * Er * Ei;
    const double O2r = Or * Or - Oi * Oi, O2i = 2.0 * Or * Oi;
    const double Epr = E2r + (wr * O2r - wi * O2i);
    const double Epi = E2i + (wr * O2i + wi * O2r);
    const double Opr = 2.0 * (Er * Or - Ei * Oi);
    const double Opi = 2.0 * (Er * Oi + Ei * Or);
    okr = (Epr - Opi) * s;
    oki = (Epi + Opr) * s;
    ojr = (Epr + Opi) * s;
    oji = (Opr - Epi) * s;
}

// =====================================================================
// FUSED MIDDLE OF THE SQUARE (two-pass Bailey + real FFT). In a single pass, on column blocks held in L2:
//   [fwd B: twiddle w^(c*k1) + length-n2 FFT of column k1] + [square pairs (k, N-k)] + [inv A': length-n2 FFT^-1 of column k1]
// Previously 3 passes over memory (store spectrum in natural order, square pass, read the spectrum again).
// Spectrum index k = k1 + n1*k2. The pair N-k is in column n1-k1 (k2 -> n2-1-k2); columns 0 and n1/2 pair with themselves.
// Block b: columns A = [8b, 8b+8) and mirrored M = [n1-8b-7, n1-8b]; lane t of block A <-> lane 7-t of block M.
// For b = 0: M = [n1-7, n1-1] + column n1/2 in lane 7; the self-paired columns (0 and n1/2) are computed separately in scalar code.
// Input src: Y1 after pass A (n2 rows x n1 columns). Output dst: layout as after the old A' (n1 rows x n2).
// LOTR_NO_FUSED_SQ=1 disables it (separate fwd B, square and inv A' passes).
// =====================================================================
static void bailey_fused_square_mid(FFTContext& ctx, const double* __restrict src_re, const double* __restrict src_im,
                                    double* __restrict dst_re, double* __restrict dst_im, size_t n1, size_t n2) {
    const size_t N = ctx.fft_len;
    const size_t R = n2;      // column length (length-n2 FFT)
    const size_t C = n1;      // number of columns (row stride in src)
    const double* const tw_re = ctx.w1_re;
    const double* const tw_im = ctx.w1_im;
    const double s = 0.25 / (double)N;
    const size_t n_blocks = n1 / 16;
    const Tw2 tw2(ctx);

#pragma omp parallel
    {
        // 16 rows (8 block A + 8 block M) x {in, out} x {re, im}, pitch R+16 (as in bailey_colfft_pass), plus copies of the self-paired columns
        const size_t P = R + 16;
        static thread_local double* scr = nullptr;
        static thread_local size_t scr_cap = 0;
        const size_t need = 64 * P + 4 * R;
        if (scr_cap < need) {
            std::free(scr);
            scr = static_cast<double*>(std::aligned_alloc(64, ((need * sizeof(double) + 63) / 64) * 64));
            scr_cap = need;
        }
        double* const in_re = scr;
        double* const in_im = scr + 16 * P;
        double* const out_re = scr + 32 * P;
        double* const out_im = scr + 48 * P;
        double* const selfA_re = scr + 64 * P;
        double* const selfA_im = selfA_re + R;
        double* const selfM_re = selfA_re + 2 * R;
        double* const selfM_im = selfA_re + 3 * R;

        const __m512i rev_idx = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i shift1 = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 7);
        const __m512d v_s = _mm512_set1_pd(s);
        const size_t spc = n1 / 2;

        // Gather 8 columns (cols[bb]) into 8 rows g_* with twiddle w^(r*col). The columns are cstart..cstart+7; with special,
        // lanes 1..7 are shifted to 0..6 and lane 7 = column n1/2 (block M for b = 0).
        auto gather = [&](size_t cstart, bool special, const size_t* cols, double* g_re, double* g_im) {
            __m512d Pr[8], Pi[8];
            for (size_t bb = 0; bb < 8; ++bb) {
                alignas(64) double pr[8];
                alignas(64) double pi[8];
                for (size_t t = 0; t < 8; ++t) {
                    const size_t idx = (t * cols[bb]) & (N - 1);
                    pr[t] = tw_re[idx]; pi[t] = tw_im[idx];
                }
                Pr[bb] = _mm512_load_pd(pr); Pi[bb] = _mm512_load_pd(pi);
            }
            for (size_t r = 0; r < R; r += 8) {
                if (r + 16 < R) {
#pragma GCC unroll 8
                    for (int t = 0; t < 8; ++t) {
                        _mm_prefetch((const char*)(src_re + (r + 16 + t) * C + cstart), _MM_HINT_T1);
                        _mm_prefetch((const char*)(src_im + (r + 16 + t) * C + cstart), _MM_HINT_T1);
                    }
                }
                __m512d vr[8], vi[8];
#pragma GCC unroll 8
                for (int t = 0; t < 8; ++t) {
                    vr[t] = _mm512_loadu_pd(src_re + (r + t) * C + cstart);
                    vi[t] = _mm512_loadu_pd(src_im + (r + t) * C + cstart);
                    if (special) {
                        vr[t] = _mm512_mask_mov_pd(_mm512_permutexvar_pd(shift1, vr[t]), 0x80, _mm512_set1_pd(src_re[(r + t) * C + spc]));
                        vi[t] = _mm512_mask_mov_pd(_mm512_permutexvar_pd(shift1, vi[t]), 0x80, _mm512_set1_pd(src_im[(r + t) * C + spc]));
                    }
                }
                st_transpose8x8(vr);
                st_transpose8x8(vi);   // vr[bb][t] = element (row r+t, column cols[bb])
#pragma GCC unroll 8
                for (int bb = 0; bb < 8; ++bb) {
                    double s_re, s_im;
                    tw2.get((r * cols[bb]) & (N - 1), s_re, s_im);
                    const __m512d Sr = _mm512_set1_pd(s_re);
                    const __m512d Si = _mm512_set1_pd(s_im);
                    const __m512d wr = _mm512_fmsub_pd(Pr[bb], Sr, _mm512_mul_pd(Pi[bb], Si));
                    const __m512d wi = _mm512_fmadd_pd(Pr[bb], Si, _mm512_mul_pd(Pi[bb], Sr));
                    _mm512_store_pd(g_re + bb * P + r, _mm512_fmsub_pd(vr[bb], wr, _mm512_mul_pd(vi[bb], wi)));
                    _mm512_store_pd(g_im + bb * P + r, _mm512_fmadd_pd(vr[bb], wi, _mm512_mul_pd(vi[bb], wr)));
                }
            }
        };

#pragma omp for schedule(static)
        for (size_t b = 0; b < n_blocks; ++b) {
            const size_t c0 = 8 * b;
            const bool sp = (b == 0);
            const size_t mstart = sp ? (n1 - 8) : (n1 - c0 - 7);
            size_t cols[16];
            for (size_t t = 0; t < 8; ++t) {
                cols[t] = c0 + t;
                cols[8 + t] = sp ? ((t < 7) ? (n1 - 7 + t) : spc) : (mstart + t);
            }
            gather(c0, false, cols, in_re, in_im);
            gather(mstart, sp, cols + 8, in_re + 8 * P, in_im + 8 * P);

            // ---- forward length-n2 FFT for 16 rows ----
            double* rre[16];
            double* rim[16];
            bool moved[16];
            for (size_t q = 0; q < 16; ++q) {
                double* a_re = in_re + q * P; double* a_im = in_im + q * P;
                double* o_re = out_re + q * P; double* o_im = out_im + q * P;
                double* res = fft_stockham_radix4_blocked(a_re, a_im, o_re, o_im, R,
                    ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i, false);
                moved[q] = (res != a_re);
                rre[q] = moved[q] ? o_re : a_re;
                rim[q] = moved[q] ? o_im : a_im;
            }
            if (sp) {
                std::memcpy(selfA_re, rre[0], R * sizeof(double));  std::memcpy(selfA_im, rim[0], R * sizeof(double));
                std::memcpy(selfM_re, rre[15], R * sizeof(double)); std::memcpy(selfM_im, rim[15], R * sizeof(double));
            }

            // ---- square pairs: A (column k1, k2) <-> M (column n1-k1, n2-1-k2) ----
            // w^k = w^k1 (lane vector, constant within the block) * w^(n1*k2) (= c_w1[k2], small table)
            const __m512d wk1_r = _mm512_loadu_pd(tw_re + c0);
            const __m512d wk1_i = _mm512_loadu_pd(tw_im + c0);
            for (size_t k2b = 0; k2b < R; k2b += 8) {
                const size_t mb = R - 8 - k2b;
                __m512d ar[8], ai[8], mr[8], mi[8];
#pragma GCC unroll 8
                for (int q = 0; q < 8; ++q) {
                    ar[q] = _mm512_load_pd(rre[q] + k2b);
                    ai[q] = _mm512_load_pd(rim[q] + k2b);
                    mr[q] = _mm512_load_pd(rre[8 + q] + mb);
                    mi[q] = _mm512_load_pd(rim[8 + q] + mb);
                }
                st_transpose8x8(ar); st_transpose8x8(ai);   // ar[t] = columns A (lanes) for k2 = k2b + t
                st_transpose8x8(mr); st_transpose8x8(mi);   // mr[u] = columns M (lanes) for k2 = mb + u
#pragma GCC unroll 8
                for (int t = 0; t < 8; ++t) {
                    const __m512d hr = _mm512_set1_pd(ctx.c_w1r[k2b + t]);
                    const __m512d hi = _mm512_set1_pd(ctx.c_w1i[k2b + t]);
                    const __m512d wr = _mm512_fmsub_pd(wk1_r, hr, _mm512_mul_pd(wk1_i, hi));
                    const __m512d wi = _mm512_fmadd_pd(wk1_r, hi, _mm512_mul_pd(wk1_i, hr));
                    const __m512d Rj = _mm512_permutexvar_pd(rev_idx, mr[7 - t]);
                    const __m512d Ij = _mm512_permutexvar_pd(rev_idx, mi[7 - t]);
                    __m512d okr, oki, ojr, oji;
                    sqr_pair_vec(ar[t], ai[t], Rj, Ij, wr, wi, v_s, okr, oki, ojr, oji);
                    ar[t] = okr; ai[t] = oki;
                    mr[7 - t] = _mm512_permutexvar_pd(rev_idx, ojr);
                    mi[7 - t] = _mm512_permutexvar_pd(rev_idx, oji);
                }
                st_transpose8x8(ar); st_transpose8x8(ai);
                st_transpose8x8(mr); st_transpose8x8(mi);
#pragma GCC unroll 8
                for (int q = 0; q < 8; ++q) {
                    _mm512_store_pd(rre[q] + k2b, ar[q]);
                    _mm512_store_pd(rim[q] + k2b, ai[q]);
                    _mm512_store_pd(rre[8 + q] + mb, mr[q]);
                    _mm512_store_pd(rim[8 + q] + mb, mi[q]);
                }
            }
            if (sp) {
                // column 0: k = n1*k2, pair k2 <-> (n2-k2) mod n2 (k2 = 0 and n2/2 pair with themselves)
                for (size_t k2 = 0; k2 <= R / 2; ++k2) {
                    const size_t j2 = (R - k2) & (R - 1);
                    const size_t widx = n1 * k2;
                    double kr, ki, jr, ji;
                    sqr_pair_scalar(selfA_re[k2], selfA_im[k2], selfA_re[j2], selfA_im[j2], tw_re[widx], tw_im[widx], s, kr, ki, jr, ji);
                    rre[0][j2] = jr; rim[0][j2] = ji;
                    rre[0][k2] = kr; rim[0][k2] = ki;
                }
                // column n1/2 (lane 7 of block M): k = n1/2 + n1*k2, pair k2 <-> n2-1-k2
                for (size_t k2 = 0; k2 < R / 2; ++k2) {
                    const size_t j2 = R - 1 - k2;
                    const size_t widx = spc + n1 * k2;
                    double kr, ki, jr, ji;
                    sqr_pair_scalar(selfM_re[k2], selfM_im[k2], selfM_re[j2], selfM_im[j2], tw_re[widx], tw_im[widx], s, kr, ki, jr, ji);
                    rre[15][j2] = jr; rim[15][j2] = ji;
                    rre[15][k2] = kr; rim[15][k2] = ki;
                }
            }

            // ---- inverse length-n2 FFT and row store dst[col*n2 + c] ----
            for (size_t q = 0; q < 16; ++q) {
                double* x_re = rre[q]; double* x_im = rim[q];
                double* y_re = moved[q] ? (in_re + q * P) : (out_re + q * P);
                double* y_im = moved[q] ? (in_im + q * P) : (out_im + q * P);
                double* res = fft_stockham_radix4_blocked(x_re, x_im, y_re, y_im, R,
                    ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i, true);
                const double* f_re = (res == x_re) ? x_re : y_re;
                const double* f_im = (res == x_re) ? x_im : y_im;
                std::memcpy(dst_re + cols[q] * R, f_re, R * sizeof(double));
                std::memcpy(dst_im + cols[q] * R, f_im, R * sizeof(double));
            }
        }
    }
}

// FUSED MIDDLE in the batch-8 layout (vector = element k2 of eight columns): same steps and the same arithmetic as
// bailey_fused_square_mid, but without transposes. The partner of column A (lane t, k2) is lane 7-t of block M at n2-1-k2, i.e.
// Z[N-k] = reversed vector M[n2-1-k2]. The self-paired columns (0 = lane 0 of block A, n1/2 = lane 7 of block M at b = 0)
// are computed in scalar code from copies.
static void bailey_fused_square_mid8(FFTContext& ctx, const double* __restrict src_re, const double* __restrict src_im,
                                     double* __restrict dst_re, double* __restrict dst_im, size_t n1, size_t n2,
                                     size_t sp = 0, size_t dp = 0) {
    const size_t N = ctx.fft_len;
    const size_t R = n2;
    const size_t C = (sp != 0) ? sp : n1;     // source row stride (Y1: n2 rows of n1, padded or not)
    const size_t DP = (dp != 0) ? dp : n2;    // destination row stride (n1 rows of n2)
    const double* const tw_re = ctx.w1_re;
    const double* const tw_im = ctx.w1_im;
    const double s = 0.25 / (double)N;
    const size_t n_blocks = n1 / 16;
    const size_t spc = n1 / 2;
    const Tw2 tw2(ctx);

#pragma omp parallel
    {
        // block A: buffers a, b, c (scr[0 .. 48R)), block M: scr[48R .. 96R), copies of the self-paired columns: 4R
        double* const scr = batch8_scratch(R, 2, 4 * R);
        double* const A_a_re = scr;          double* const A_a_im = scr + 8 * R;
        double* const A_b_re = scr + 16 * R; double* const A_b_im = scr + 24 * R;
        double* const A_c_re = scr + 32 * R; double* const A_c_im = scr + 40 * R;
        double* const M_a_re = scr + 48 * R; double* const M_a_im = scr + 56 * R;
        double* const M_b_re = scr + 64 * R; double* const M_b_im = scr + 72 * R;
        double* const M_c_re = scr + 80 * R; double* const M_c_im = scr + 88 * R;
        double* const selfA_re = scr + 96 * R;
        double* const selfA_im = selfA_re + R;
        double* const selfM_re = selfA_re + 2 * R;
        double* const selfM_im = selfA_re + 3 * R;

        const __m512i rev_idx = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512d v_s = _mm512_set1_pd(s);
        // R >= 512: two-level FFT, spectrum in the blocked layout -> address of vector k2 via P()
        const bool two = batch8_two_level(R);
        const size_t R2 = R >> 3;
        auto P = [&](size_t K) -> size_t { return two ? batch8_posL(K, R2) : K; };

#pragma omp for schedule(static)
        for (size_t b = 0; b < n_blocks; ++b) {
            const size_t c0 = 8 * b;
            const bool sp = (b == 0);
            const size_t mstart = sp ? (n1 - 8) : (n1 - c0 - 7);
            size_t cols[16];
            for (size_t t = 0; t < 8; ++t) {
                cols[t] = c0 + t;
                cols[8 + t] = sp ? ((t < 7) ? (n1 - 7 + t) : spc) : (mstart + t);
            }

            // ---- fwd B: twiddle + length-n2 FFT of both blocks ----
            batch8_gather(src_re, src_im, R, C, c0, false, 0, true, false, ctx, tw2, A_a_re, A_a_im);
            batch8_gather(src_re, src_im, R, C, mstart, sp, spc, true, false, ctx, tw2, M_a_re, M_a_im);
            bool A_in_b = true, M_in_b = true;
            if (two) {
                fft_batch8_2l(A_a_re, A_a_im, false, A_b_re, A_b_im, A_c_re, A_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
                fft_batch8_2l(M_a_re, M_a_im, false, M_b_re, M_b_im, M_c_re, M_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
            }
            else {
                A_in_b = fft_batch8(A_a_re, A_a_im, A_b_re, A_b_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
                M_in_b = fft_batch8(M_a_re, M_a_im, M_b_re, M_b_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
            }
            double* const XA_re = A_in_b ? A_b_re : A_a_re; double* const XA_im = A_in_b ? A_b_im : A_a_im;
            double* const XM_re = M_in_b ? M_b_re : M_a_re; double* const XM_im = M_in_b ? M_b_im : M_a_im;

            if (sp) {
                for (size_t k2 = 0; k2 < R; ++k2) {
                    const size_t q = P(k2);
                    selfA_re[k2] = XA_re[8 * q];     selfA_im[k2] = XA_im[8 * q];       // column 0
                    selfM_re[k2] = XM_re[8 * q + 7]; selfM_im[k2] = XM_im[8 * q + 7];   // column n1/2
                }
            }

            // ---- square pairs: A[k2] (lanes = columns c0..c0+7) <-> reversed M[n2-1-k2] ----
            // w^k = w^k1 (lane vector) * w^(n1*k2) (= c_w1[k2])
            const __m512d wk1_r = _mm512_loadu_pd(tw_re + c0);
            const __m512d wk1_i = _mm512_loadu_pd(tw_im + c0);
            for (size_t k2 = 0; k2 < R; ++k2) {
                const size_t j2 = R - 1 - k2;
                const __m512d hr = _mm512_set1_pd(ctx.c_w1r[k2]);
                const __m512d hi = _mm512_set1_pd(ctx.c_w1i[k2]);
                const __m512d wr = _mm512_fmsub_pd(wk1_r, hr, _mm512_mul_pd(wk1_i, hi));
                const __m512d wi = _mm512_fmadd_pd(wk1_r, hi, _mm512_mul_pd(wk1_i, hr));
                const size_t pk = P(k2), pj = P(j2);
                const __m512d Rk = _mm512_load_pd(XA_re + 8 * pk);
                const __m512d Ik = _mm512_load_pd(XA_im + 8 * pk);
                const __m512d Rj = _mm512_permutexvar_pd(rev_idx, _mm512_load_pd(XM_re + 8 * pj));
                const __m512d Ij = _mm512_permutexvar_pd(rev_idx, _mm512_load_pd(XM_im + 8 * pj));
                __m512d okr, oki, ojr, oji;
                sqr_pair_vec(Rk, Ik, Rj, Ij, wr, wi, v_s, okr, oki, ojr, oji);
                _mm512_store_pd(XA_re + 8 * pk, okr);
                _mm512_store_pd(XA_im + 8 * pk, oki);
                _mm512_store_pd(XM_re + 8 * pj, _mm512_permutexvar_pd(rev_idx, ojr));
                _mm512_store_pd(XM_im + 8 * pj, _mm512_permutexvar_pd(rev_idx, oji));
            }
            if (sp) {
                // column 0: k = n1*k2, pair k2 <-> (n2-k2) mod n2
                for (size_t k2 = 0; k2 <= R / 2; ++k2) {
                    const size_t j2 = (R - k2) & (R - 1);
                    const size_t widx = n1 * k2;
                    double kr, ki, jr, ji;
                    sqr_pair_scalar(selfA_re[k2], selfA_im[k2], selfA_re[j2], selfA_im[j2], tw_re[widx], tw_im[widx], s, kr, ki, jr, ji);
                    XA_re[8 * P(j2)] = jr; XA_im[8 * P(j2)] = ji;
                    XA_re[8 * P(k2)] = kr; XA_im[8 * P(k2)] = ki;
                }
                // column n1/2: k = n1/2 + n1*k2, pair k2 <-> n2-1-k2
                for (size_t k2 = 0; k2 < R / 2; ++k2) {
                    const size_t j2 = R - 1 - k2;
                    const size_t widx = spc + n1 * k2;
                    double kr, ki, jr, ji;
                    sqr_pair_scalar(selfM_re[k2], selfM_im[k2], selfM_re[j2], selfM_im[j2], tw_re[widx], tw_im[widx], s, kr, ki, jr, ji);
                    XM_re[8 * P(j2) + 7] = jr; XM_im[8 * P(j2) + 7] = ji;
                    XM_re[8 * P(k2) + 7] = kr; XM_im[8 * P(k2) + 7] = ki;
                }
            }

            // ---- inv A': length-n2 FFT^-1 of both blocks and row store dst[col*n2 + c] ----
            if (two) {
                // spectrum in the blocked layout (b) -> result in the blocked layout (a), buffer c as scratch
                fft_batch8_2l(A_b_re, A_b_im, true, A_a_re, A_a_im, A_c_re, A_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                batch8_store_rows(A_a_re, A_a_im, R, cols, dst_re, dst_im, true, DP);
                fft_batch8_2l(M_b_re, M_b_im, true, M_a_re, M_a_im, M_c_re, M_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                batch8_store_rows(M_a_re, M_a_im, R, cols + 8, dst_re, dst_im, true, DP);
            }
            else {
                {
                    double* const y_re = A_in_b ? A_a_re : A_b_re; double* const y_im = A_in_b ? A_a_im : A_b_im;
                    const bool in_y = fft_batch8(XA_re, XA_im, y_re, y_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                    batch8_store_rows(in_y ? y_re : XA_re, in_y ? y_im : XA_im, R, cols, dst_re, dst_im, false, DP);
                }
                {
                    double* const y_re = M_in_b ? M_a_re : M_b_re; double* const y_im = M_in_b ? M_a_im : M_b_im;
                    const bool in_y = fft_batch8(XM_re, XM_im, y_re, y_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                    batch8_store_rows(in_y ? y_re : XM_re, in_y ? y_im : XM_im, R, cols + 8, dst_re, dst_im, false, DP);
                }
            }
        }
    }
}

static bool fused_sq_disabled() {
    static const bool v = (std::getenv("LOTR_NO_FUSED_SQ") != nullptr);
    return v;
}

// Can the square use the fused middle (two-pass Bailey for this FFT length); sets n1, n2 and the twiddle tables.
static bool bailey_fused_ok(FFTContext& ctx, size_t& n1, size_t& n2) {
    if (fused_sq_disabled()) return false;
    for (size_t f : ctx.factor_plan.factors) {
        if (f != 2) return false;
    }
    unsigned p = 0; size_t temp = ctx.fft_len;
    while (temp > 1) { temp >>= 1; p++; }
    if (!use_bailey_engine(p) || !bailey_twopass_for(p)) return false;
    bailey_setup(ctx, n1, n2);
    return (n1 % 16 == 0) && (n2 % 8 == 0) && (n2 >= 16);
}

// ---------------------------------------------------------
// KARATSUBA UNPACK: EXACT from v4.2.2
// ---------------------------------------------------------
void unpack_karatsuba_overlap_add(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val) {
    const size_t n_fft = ctx.fft_len;
    const size_t split_chunks = n_fft / 2;

    size_t max_chunks = n_limbs * 8;
    size_t double_limbs = n_limbs * 2 + 2;
    std::vector<uint64_t> sq(double_limbs, 0);

    int64_t carry = 0;
    __m512d v_max_err = _mm512_setzero_pd();

    for (size_t i = 0; i < max_chunks; i += 8) {
        __m512d v_A2 = (i < n_fft) ? _mm512_load_pd(&ctx.k_A_re[i]) : _mm512_setzero_pd();
        __m512d v_AB = ((i >= split_chunks) && ((i - split_chunks) < n_fft))
            ? _mm512_load_pd(&ctx.k_AB_re[i - split_chunks])
            : _mm512_setzero_pd();
        __m512d v_B2 = ((i >= 2 * split_chunks) && ((i - 2 * split_chunks) < n_fft))
            ? _mm512_load_pd(&ctx.k_B_re[i - 2 * split_chunks])
            : _mm512_setzero_pd();

        __m512d v_sum = _mm512_add_pd(_mm512_add_pd(v_A2, v_AB), v_B2);
        __m512i v_int = _mm512_cvtpd_epi64(v_sum);

        __m512d v_rounded_back = _mm512_cvtepi64_pd(v_int);
        __m512d v_err = _mm512_sub_pd(v_sum, v_rounded_back);
        v_max_err = _mm512_max_pd(v_max_err, _mm512_abs_pd(v_err));

        alignas(64) int64_t temp_i[8];
        _mm512_store_si512((__m512i*)temp_i, v_int);

        size_t limb_base = i >> 2;
        uint64_t acc0 = 0;
        uint64_t acc1 = 0;

        
        for (size_t k = 0; k < 8; k++) {
            int64_t ival = temp_i[k] + carry;
            carry = ival >> 16;
            uint64_t chunk = (uint64_t)(ival & 0xFFFF);

            if (k < 4) acc0 |= (chunk << (k << 4));
            else       acc1 |= (chunk << ((k - 4) << 4));
        }

        if (limb_base < double_limbs) sq[limb_base] |= acc0;
        if (limb_base + 1 < double_limbs) sq[limb_base + 1] |= acc1;
    }

    alignas(64) double err_arr[8];
    _mm512_store_pd(err_arr, v_max_err);
    double block_max_err = 0.0;
    for (int i = 0; i < 8; i++) {
        if (err_arr[i] > block_max_err) block_max_err = err_arr[i];
    }
    ctx.last_max_diff = block_max_err;

    size_t carry_idx = max_chunks >> 2;
    while (carry > 0 && carry_idx < sq.size()) {
        unsigned __int128 sum = (unsigned __int128)sq[carry_idx] + carry;
        sq[carry_idx] = (uint64_t)sum;
        carry = (int64_t)(sum >> 64);
        carry_idx++;
    }

    size_t split_q = n_bits / 64;
    unsigned split_r = n_bits % 64;

    std::vector<uint64_t> H_arr(n_limbs + 2, 0);
    std::vector<uint64_t> Q_arr(n_limbs + 2, 0);

    for (size_t i = 0; i < n_limbs; i++) {
        size_t src_idx = split_q + i;
        uint64_t h_val = 0;
        if (src_idx < sq.size()) {
            h_val = sq[src_idx] >> split_r;
            if (split_r > 0 && src_idx + 1 < sq.size()) {
                h_val |= (sq[src_idx + 1] << (64 - split_r));
            }
        }
        H_arr[i] = h_val;
    }

    uint64_t rem = 0;
    for (ptrdiff_t i = n_limbs - 1; i >= 0; i--) {
        uint64_t q_val, r_val;
        uint64_t low = H_arr[i];
        uint64_t high = rem;

        asm volatile (
            "divq %[k_val]"
            : "=a" (q_val), "=d" (r_val)
            : "d" (high), "a" (low), [k_val] "c" (k_val)
            : "cc"
            );
        Q_arr[i] = q_val;
        rem = r_val;
    }

    std::memset(out, 0, n_limbs * sizeof(uint64_t));
    for (size_t i = 0; i < split_q && i < n_limbs; i++) {
        out[i] = sq[i];
    }
    if (split_r > 0 && split_q < n_limbs) {
        out[split_q] = sq[split_q] & ((1ULL << split_r) - 1);
    }

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

            if (i == split_q) {
                val_to_add += (split_r == 0) ? k_val : (k_val << split_r);
            }
            if (i == split_q + 1 && split_r > 0) {
                val_to_add += (k_val >> (64 - split_r));
            }

            unsigned __int128 sum = (unsigned __int128)out[i] + val_to_add;
            out[i] = (uint64_t)sum;
            local_carry += (uint64_t)(sum >> 64);
        }
    }
}


// Old square path: A = lower half of the digits (re), B = upper half (im); 1 forward FFT + 2 inverse (A^2 + i*B^2 and AB).
static void fft_square_karatsuba3(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    auto t_pack_0 = clock::now();
    
    direct_pack_square(v, n_limbs, ctx);
    auto t_pack_1 = clock::now();

    auto run_fft_on_buffer = [&](double*& target_re, double*& target_im, bool inverse) {
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);

        bool is_pure_power_of_two = true;
        for (size_t f : ctx.factor_plan.factors) {
            if (f != 2) { is_pure_power_of_two = false; break; }
        }

        if (is_pure_power_of_two) {
            unsigned p = 0; size_t temp = ctx.fft_len;
            while (temp > 1) { temp >>= 1; p++; }

            // HYBRID ENGINE CHOICE (see use_bailey_engine):
            if (use_bailey_engine(p)) {
                // Bailey 2D (all even powers, e.g. 2^16 = 65536, and some odd ones)
                fft_bailey_2d(ctx, inverse);
            }
            else {
                // Stockham 1D (odd powers, e.g. 2^19 = 524288)
                double* res_re = fft_stockham_radix8(ctx, inverse);
                if (res_re != ctx.re) {
                    std::swap(ctx.re, ctx.re2);
                    std::swap(ctx.im, ctx.im2);
                }
            }
        }
        else {
            fft_mixed_radix_execute(ctx, ctx.factor_plan, inverse);
        }

        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);
        };

    //  FORWARD FFT
    auto t_fwd_0 = clock::now();
    run_fft_on_buffer(ctx.re, ctx.im, false);
    auto t_fwd_1 = clock::now();

    auto t_sq_0 = clock::now();
    double inv_n = 1.0 / (double)ctx.fft_len;

    // IN-PLACE SQUARE: the spectrum P = A^2 + i*B^2 is written back to re/im, and AB to k_AB_*.
    // Each pair (i, N-i) is read and written in one iteration and the pairs are disjoint, so
    // overwriting the input is safe. For element N-i the values Ak_im and Bk_im have the opposite sign,
    // the rest is the same, so the second element of the pair follows from the first (Hermitian symmetry) and
    // every element is read only once (previously twice: as i and as the mirror).
    const size_t fft_n = ctx.fft_len;
    const size_t half_n = fft_n / 2;
    double* const pre = ctx.re;
    double* const pim = ctx.im;
    double* const ab_re = ctx.k_AB_re;
    double* const ab_im = ctx.k_AB_im;

    // Element [0]
    {
        const double A_re = pre[0], B_re = pim[0];
        pre[0] = (A_re * A_re) * inv_n;
        pim[0] = (B_re * B_re) * inv_n;
        ab_re[0] = (A_re * B_re * 2.0) * inv_n;
        ab_im[0] = 0.0;
    }

    // Pair (i, N-i) computed in scalar code (elements 1..7 and small / unusual FFT lengths)
    auto scalar_pair = [&](size_t i) {
        const size_t j = fft_n - i;
        const double Rk = pre[i], Ik = pim[i];
        const double Rj = pre[j], Ij = pim[j];

        const double Ak_re = (Rk + Rj) * 0.5, Ak_im = (Ik - Ij) * 0.5;
        const double Bk_re = (Ik + Ij) * 0.5, Bk_im = (Rj - Rk) * 0.5;

        const double A2_re = Ak_re * Ak_re - Ak_im * Ak_im;
        const double A2_im = 2.0 * Ak_re * Ak_im;
        const double B2_re = Bk_re * Bk_re - Bk_im * Bk_im;
        const double B2_im = 2.0 * Bk_re * Bk_im;
        const double AB_re = Ak_re * Bk_re - Ak_im * Bk_im;
        const double AB_im = Ak_re * Bk_im + Ak_im * Bk_re;

        pre[i] = (A2_re - B2_im) * inv_n;
        pim[i] = (A2_im + B2_re) * inv_n;
        ab_re[i] = (AB_re * 2.0) * inv_n;
        ab_im[i] = (AB_im * 2.0) * inv_n;

        pre[j] = (A2_re + B2_im) * inv_n;
        pim[j] = (B2_re - A2_im) * inv_n;
        ab_re[j] = ab_re[i];
        ab_im[j] = -ab_im[i];
    };

    if (fft_n >= 32 && (fft_n % 16) == 0) {
        // Elements 1..7 and their mirrors (N-7..N-1) in scalar code, so the vector block starts at 8 (64 B alignment)
        for (size_t i = 1; i < 8; i++) scalar_pair(i);

        const __m512i rev_idx = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512d v_half = _mm512_set1_pd(0.5);
        const __m512d v_two = _mm512_set1_pd(2.0);
        const __m512d v_inv_n = _mm512_set1_pd(inv_n);
        const __m512d v_neg_one = _mm512_set1_pd(-1.0);

#pragma omp parallel for schedule(static)
        for (size_t i = 8; i < half_n; i += 8) {
            const size_t j_start = fft_n - i - 7;   // mirror block [N-i-7, N-i], reversed order

            const __m512d Rk = _mm512_load_pd(&pre[i]);
            const __m512d Ik = _mm512_load_pd(&pim[i]);
            const __m512d Rj = _mm512_permutexvar_pd(rev_idx, _mm512_loadu_pd(&pre[j_start]));
            const __m512d Ij = _mm512_permutexvar_pd(rev_idx, _mm512_loadu_pd(&pim[j_start]));

            const __m512d Ak_re = _mm512_mul_pd(_mm512_add_pd(Rk, Rj), v_half);
            const __m512d Ak_im = _mm512_mul_pd(_mm512_sub_pd(Ik, Ij), v_half);
            const __m512d Bk_re = _mm512_mul_pd(_mm512_add_pd(Ik, Ij), v_half);
            const __m512d Bk_im = _mm512_mul_pd(_mm512_sub_pd(Rj, Rk), v_half);

            const __m512d A2_re = _mm512_fmsub_pd(Ak_re, Ak_re, _mm512_mul_pd(Ak_im, Ak_im));
            const __m512d A2_im = _mm512_mul_pd(v_two, _mm512_mul_pd(Ak_re, Ak_im));
            const __m512d B2_re = _mm512_fmsub_pd(Bk_re, Bk_re, _mm512_mul_pd(Bk_im, Bk_im));
            const __m512d B2_im = _mm512_mul_pd(v_two, _mm512_mul_pd(Bk_re, Bk_im));
            const __m512d AB_re = _mm512_fmsub_pd(Ak_re, Bk_re, _mm512_mul_pd(Ak_im, Bk_im));
            const __m512d AB_im = _mm512_fmadd_pd(Ak_re, Bk_im, _mm512_mul_pd(Ak_im, Bk_re));

            // results for i
            const __m512d P_re_i = _mm512_mul_pd(_mm512_sub_pd(A2_re, B2_im), v_inv_n);
            const __m512d P_im_i = _mm512_mul_pd(_mm512_add_pd(A2_im, B2_re), v_inv_n);
            const __m512d AB_re_i = _mm512_mul_pd(_mm512_mul_pd(AB_re, v_two), v_inv_n);
            const __m512d AB_im_i = _mm512_mul_pd(_mm512_mul_pd(AB_im, v_two), v_inv_n);

            // results for N-i (Hermitian mirror)
            const __m512d P_re_j = _mm512_mul_pd(_mm512_add_pd(A2_re, B2_im), v_inv_n);
            const __m512d P_im_j = _mm512_mul_pd(_mm512_sub_pd(B2_re, A2_im), v_inv_n);

            // store in place (P) and into k_AB (AB)
            _mm512_store_pd(&pre[i], P_re_i);
            _mm512_store_pd(&pim[i], P_im_i);
            _mm512_storeu_pd(&pre[j_start], _mm512_permutexvar_pd(rev_idx, P_re_j));
            _mm512_storeu_pd(&pim[j_start], _mm512_permutexvar_pd(rev_idx, P_im_j));

            _mm512_store_pd(&ab_re[i], AB_re_i);
            _mm512_store_pd(&ab_im[i], AB_im_i);
            _mm512_storeu_pd(&ab_re[j_start], _mm512_permutexvar_pd(rev_idx, AB_re_i));
            _mm512_storeu_pd(&ab_im[j_start], _mm512_permutexvar_pd(rev_idx, _mm512_mul_pd(AB_im_i, v_neg_one)));
        }
    }
    else {
        // small / unusual lengths: all pairs in scalar code
        for (size_t i = 1; i < half_n; i++) scalar_pair(i);
    }

    // Middle element N/2 (paired with itself)
    if (half_n >= 1) {
        const double Rk = pre[half_n], Ik = pim[half_n];
        pre[half_n] = (Rk * Rk) * inv_n;
        pim[half_n] = (Ik * Ik) * inv_n;
        ab_re[half_n] = (Rk * Ik * 2.0) * inv_n;
        ab_im[half_n] = 0.0;
    }

    auto t_sq_1 = clock::now();

    auto t_inv_0 = clock::now();
    run_fft_on_buffer(ctx.re, ctx.im, true);
    run_fft_on_buffer(ctx.k_AB_re, ctx.k_AB_im, true);
    auto t_inv_1 = clock::now();

    
    auto t_unpack_0 = clock::now();
    unpack_karatsuba_overlap_add_vbmi(v, n_limbs, ctx, n_bits, k_val, ctx.re, ctx.k_AB_re, ctx.im);
    auto t_unpack_1 = clock::now();

    ctx.prof_pack += secd(t_pack_1 - t_pack_0).count();
    ctx.prof_fwd += secd(t_fwd_1 - t_fwd_0).count();
    ctx.prof_square += secd(t_sq_1 - t_sq_0).count();
    ctx.prof_inv += secd(t_inv_1 - t_inv_0).count();
    ctx.prof_unpack += secd(t_unpack_1 - t_unpack_0).count();
    ctx.prof_calls++;
}

// FFT (forward or inverse) on ctx.re/ctx.im; the result is always in ctx.re/ctx.im (the pointers may get swapped with re2/im2).
static void fft_run_inplace(FFTContext& ctx, bool inverse) {
    bool is_pure_power_of_two = true;
    for (size_t f : ctx.factor_plan.factors) {
        if (f != 2) { is_pure_power_of_two = false; break; }
    }
    if (!is_pure_power_of_two) {
        fft_mixed_radix_execute(ctx, ctx.factor_plan, inverse);
        return;
    }
    unsigned p = 0; size_t temp = ctx.fft_len;
    while (temp > 1) { temp >>= 1; p++; }
    if (use_bailey_engine(p)) {
        fft_bailey_2d(ctx, inverse);
    }
    else {
        double* res_re = fft_stockham_radix8(ctx, inverse);
        if (res_re != ctx.re) {
            std::swap(ctx.re, ctx.re2);
            std::swap(ctx.im, ctx.im2);
        }
    }
}

// =====================================================================
// SQUARE VIA REAL FFT (2 FFTs instead of 3).
// The number x = a sequence of 2N real digits d[m] (N = fft_len; upper half = 0, since the square has 2x more digits),
// stored as complex z[j] = d[2j] + i*d[2j+1] of length N. After the FFT: Z[k]. For the pair (k, N-k):
//   E = (Z[k] + conj Z[N-k]) / 2      (spectrum of the even digits)
//   O = (Z[k] - conj Z[N-k]) / (2i)   (spectrum of the odd digits)
// The spectrum of the square of the 2N-digit sequence (X[k] = E + W^k O, X[k+N] = E - W^k O, W = e^(-i*pi/N)), folded back to
// length N, gives (no separate multiplication by W^-k, because W^k * conj(W^k) = 1):
//   E' = E^2 + w^k O^2   (w = W^2 = e^(-2*pi*i/N), table w1)
//   O' = 2 E O
//   Z'[k] = E' + i O',   Z'[N-k] = conj(E') + i conj(O')
// and one inverse FFT gives y[2j] + i*y[2j+1] = the convolution coefficients (digits of the square) interleaved re/im.
// Previously: 1 forward FFT + 2 inverse (A^2 + iB^2 and AB). LOTR_SQ_KARATSUBA=1 = old path.
// =====================================================================
static void fft_square_real(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    // Two-pass Bailey: fwd A | fusion (fwd B + square + inv A') | inv B'. In the profile: fwd = A, square = fusion, inv = B'.
    size_t n1 = 0, n2 = 0;
    const bool fused = bailey_fused_ok(ctx, n1, n2);

    // Row padding (8 doubles = 64 B) in the matrices between passes: rows whose length is a power of two (e.g. 4 KB)
    // map to the same L1 set (conflicts when gathering columns and in transposed stores). Only when
    // all passes are batch-8 (the old kernel does not know about padding). LOTR_NO_PAD=1 disables it.
    static const bool no_pad = (std::getenv("LOTR_NO_PAD") != nullptr);
    const size_t pad = (fused && !no_pad && use_batch8(n1) && use_batch8(n2)) ? 8 : 0;
    ctx.lay_pad = pad;
    if (pad) {
        unsigned lg = 0; while ((size_t(1) << lg) < n2) ++lg;
        ctx.lay_lg = lg;   // a row of the x and result layout = n2 elements
    }

    auto t_pack_0 = clock::now();
    direct_pack_square_real(v, n_limbs, ctx);
    auto t_pack_1 = clock::now();

    if (fused) {
        auto t0 = clock::now();
        if (pad) {
            // A: x (n1 rows of n2, stride n2+pad) -> Y1 (n2 rows of n1, stride n1+pad)
            { LOTR_PASS(ctx, 0); bailey_colfft_pass8(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, false, false, ctx, false,
                ctx.r_w1r, ctx.r_w1i, n2 + pad, n1 + pad); }
        }
        else {
            { LOTR_PASS(ctx, 0); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, false, false, ctx, false,
                ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i); }
        }
        auto t1 = clock::now();
        {
            LOTR_PASS(ctx, 1);
            // fusion: Y1 (stride n1+pad) -> n1 rows of n2 (stride n2+pad)
            if (use_batch8(n2)) bailey_fused_square_mid8(ctx, ctx.re2, ctx.im2, ctx.re, ctx.im, n1, n2,
                                                         pad ? n1 + pad : 0, pad ? n2 + pad : 0);
            else bailey_fused_square_mid(ctx, ctx.re2, ctx.im2, ctx.re, ctx.im, n1, n2);
        }
        auto t2 = clock::now();
        if (pad) {
            // B': n1 rows of n2 (stride n2+pad) -> result in the x layout (n1 rows of n2, stride n2+pad)
            { LOTR_PASS(ctx, 2); bailey_colfft_pass8(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, true, true, ctx, true,
                ctx.r_w1r, ctx.r_w1i, n2 + pad, n2 + pad); }
        }
        else {
            { LOTR_PASS(ctx, 2); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, true, true, ctx, true,
                ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i); }
        }
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);
        auto t3 = clock::now();

        unpack_karatsuba_overlap_add_vbmi(v, n_limbs, ctx, n_bits, k_val, ctx.re, nullptr, ctx.im);
        ctx.lay_pad = 0;
        auto t4 = clock::now();

        ctx.prof_pack += secd(t_pack_1 - t_pack_0).count();
        ctx.prof_fwd += secd(t1 - t0).count();
        ctx.prof_square += secd(t2 - t1).count();
        ctx.prof_inv += secd(t3 - t2).count();
        ctx.prof_unpack += secd(t4 - t3).count();
        ctx.prof_calls++;
        return;
    }

    auto t_fwd_0 = clock::now();
    fft_run_inplace(ctx, false);
    auto t_fwd_1 = clock::now();

    auto t_sq_0 = clock::now();
    const size_t fft_n = ctx.fft_len;
    const size_t half_n = fft_n / 2;
    // E, O computed without dividing by 2 (i.e. 2E, 2O), so the result is 4x larger: scale 1/(4N)
    const double s = 0.25 / (double)fft_n;
    double* const pre = ctx.re;
    double* const pim = ctx.im;
    const double* const wr_t = ctx.w1_re;
    const double* const wi_t = ctx.w1_im;

    // Pair (k, N-k) in scalar code; for k = 0 and k = N/2 the element is paired with itself (same formulas).
    auto scalar_pair = [&](size_t k) {
        const size_t j = (k == 0) ? 0 : fft_n - k;
        const double Rk = pre[k], Ik = pim[k];
        const double Rj = pre[j], Ij = pim[j];
        const double Er = Rk + Rj, Ei = Ik - Ij;     // 2E
        const double Or = Ik + Ij, Oi = Rj - Rk;     // 2O
        const double wr = wr_t[k], wi = wi_t[k];
        const double E2r = Er * Er - Ei * Ei, E2i = 2.0 * Er * Ei;   // 4E^2
        const double O2r = Or * Or - Oi * Oi, O2i = 2.0 * Or * Oi;   // 4O^2
        const double Epr = E2r + (wr * O2r - wi * O2i);              // 4E'
        const double Epi = E2i + (wr * O2i + wi * O2r);
        const double Opr = 2.0 * (Er * Or - Ei * Oi);                // 4O' = 2 * (2E)(2O)
        const double Opi = 2.0 * (Er * Oi + Ei * Or);
        pre[j] = (Epr + Opi) * s;
        pim[j] = (Opr - Epi) * s;
        pre[k] = (Epr - Opi) * s;
        pim[k] = (Epi + Opr) * s;
    };

    scalar_pair(0);
    if (fft_n >= 32 && (fft_n % 16) == 0) {
        // Elements 1..7 and their mirrors in scalar code, so the vector block starts at 8 (64 B alignment)
        for (size_t k = 1; k < 8; k++) scalar_pair(k);

        const __m512i rev_idx = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512d v_two = _mm512_set1_pd(2.0);
        const __m512d v_s = _mm512_set1_pd(s);

#pragma omp parallel for schedule(static)
        for (size_t i = 8; i < half_n; i += 8) {
            const size_t j_start = fft_n - i - 7;   // mirror block [N-i-7, N-i], reversed order

            const __m512d Rk = _mm512_load_pd(&pre[i]);
            const __m512d Ik = _mm512_load_pd(&pim[i]);
            const __m512d Rj = _mm512_permutexvar_pd(rev_idx, _mm512_loadu_pd(&pre[j_start]));
            const __m512d Ij = _mm512_permutexvar_pd(rev_idx, _mm512_loadu_pd(&pim[j_start]));
            const __m512d wr = _mm512_load_pd(&wr_t[i]);
            const __m512d wi = _mm512_load_pd(&wi_t[i]);

            const __m512d Er = _mm512_add_pd(Rk, Rj);
            const __m512d Ei = _mm512_sub_pd(Ik, Ij);
            const __m512d Or = _mm512_add_pd(Ik, Ij);
            const __m512d Oi = _mm512_sub_pd(Rj, Rk);

            const __m512d E2r = _mm512_fmsub_pd(Er, Er, _mm512_mul_pd(Ei, Ei));
            const __m512d E2i = _mm512_mul_pd(v_two, _mm512_mul_pd(Er, Ei));
            const __m512d O2r = _mm512_fmsub_pd(Or, Or, _mm512_mul_pd(Oi, Oi));
            const __m512d O2i = _mm512_mul_pd(v_two, _mm512_mul_pd(Or, Oi));

            const __m512d Epr = _mm512_add_pd(E2r, _mm512_fmsub_pd(wr, O2r, _mm512_mul_pd(wi, O2i)));
            const __m512d Epi = _mm512_add_pd(E2i, _mm512_fmadd_pd(wr, O2i, _mm512_mul_pd(wi, O2r)));
            const __m512d Opr = _mm512_mul_pd(v_two, _mm512_fmsub_pd(Er, Or, _mm512_mul_pd(Ei, Oi)));
            const __m512d Opi = _mm512_mul_pd(v_two, _mm512_fmadd_pd(Er, Oi, _mm512_mul_pd(Ei, Or)));

            _mm512_store_pd(&pre[i], _mm512_mul_pd(_mm512_sub_pd(Epr, Opi), v_s));
            _mm512_store_pd(&pim[i], _mm512_mul_pd(_mm512_add_pd(Epi, Opr), v_s));
            _mm512_storeu_pd(&pre[j_start], _mm512_permutexvar_pd(rev_idx, _mm512_mul_pd(_mm512_add_pd(Epr, Opi), v_s)));
            _mm512_storeu_pd(&pim[j_start], _mm512_permutexvar_pd(rev_idx, _mm512_mul_pd(_mm512_sub_pd(Opr, Epi), v_s)));
        }
    }
    else {
        for (size_t k = 1; k < half_n; k++) scalar_pair(k);
    }
    if (half_n >= 1) scalar_pair(half_n);
    auto t_sq_1 = clock::now();

    auto t_inv_0 = clock::now();
    fft_run_inplace(ctx, true);
    auto t_inv_1 = clock::now();

    auto t_unpack_0 = clock::now();
    unpack_karatsuba_overlap_add_vbmi(v, n_limbs, ctx, n_bits, k_val, ctx.re, nullptr, ctx.im);
    auto t_unpack_1 = clock::now();

    ctx.prof_pack += secd(t_pack_1 - t_pack_0).count();
    ctx.prof_fwd += secd(t_fwd_1 - t_fwd_0).count();
    ctx.prof_square += secd(t_sq_1 - t_sq_0).count();
    ctx.prof_inv += secd(t_inv_1 - t_inv_0).count();
    ctx.prof_unpack += secd(t_unpack_1 - t_unpack_0).count();
    ctx.prof_calls++;
}

static bool sq_real_disabled() {
    static const bool v = (std::getenv("LOTR_SQ_KARATSUBA") != nullptr);
    return v;
}

// Self-test (once per FFT context): square a pseudo-random number < 2^n with both paths; the results must be identical.
// On a mismatch the context stays on the old path (3 FFTs) - the test still computes correctly, only slower.
static void sq_real_selftest(FFTContext& ctx, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    std::vector<uint64_t> a(n_limbs, 0);
    uint64_t st = 0x9E3779B97F4A7C15ULL ^ (uint64_t)ctx.fft_len ^ (k_val << 20);
    for (size_t i = 0; i < n_limbs; ++i) {   // splitmix64
        uint64_t z = (st += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        a[i] = z ^ (z >> 31);
    }
    const size_t q = n_bits / 64;
    const unsigned r = n_bits % 64;
    for (size_t i = q; i < n_limbs; ++i) a[i] = (i == q && r > 0) ? (a[i] & ((1ULL << r) - 1)) : 0;
    std::vector<uint64_t> b(a);

    fft_square_karatsuba3(ctx, a.data(), n_limbs, n_bits, k_val);
    const double err3 = ctx.last_max_diff;
    fft_square_real(ctx, b.data(), n_limbs, n_bits, k_val);
    const double err2 = ctx.last_max_diff;

    const bool ok = (std::memcmp(a.data(), b.data(), n_limbs * sizeof(uint64_t)) == 0) && (err2 < 0.4);
    ctx.sq_real_state = ok ? 1 : -1;
    static std::atomic<bool> printed{ false };
    if (!ok || (g_detail && !printed.exchange(true))) {
        std::fprintf(stderr, "[SQ_REAL] autotest %s: FFT %zu, max err real-FFT %.3g / Karatsuba %.3g%s\n",
            ok ? "OK" : "MISMATCH", ctx.fft_len, err2, err3, ok ? "" : " -> using the old path (3 FFTs)");
    }
}

void fft_square_karatsuba(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    if (sq_real_disabled() || n_limbs * 4 > ctx.fft_len || ctx.sq_real_state < 0) {
        fft_square_karatsuba3(ctx, v, n_limbs, n_bits, k_val);
        return;
    }
    if (ctx.sq_real_state == 0) sq_real_selftest(ctx, n_limbs, n_bits, k_val);
    if (ctx.sq_real_state > 0) fft_square_real(ctx, v, n_limbs, n_bits, k_val);
    else fft_square_karatsuba3(ctx, v, n_limbs, n_bits, k_val);
}

void fft_mul_karatsuba(FFTContext& ctx, uint64_t* out, const uint64_t* X, const uint64_t* Y, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    extern bool use_avx2_fallback;
    extern void fft_mul_karatsuba_avx2(FFTContext & ctx, uint64_t * target, const uint64_t * A, const uint64_t * B, size_t n_limbs, unsigned n_bits, uint64_t k_val);
    if (use_avx2_fallback) {
        fft_mul_karatsuba_avx2(ctx, out, X, Y, n_limbs, n_bits, k_val);
        return;
    }

    auto run_fft_on_buffer = [&](double*& target_re, double*& target_im, bool inverse) {
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);

        bool is_pure_power_of_two = true;
        for (size_t f : ctx.factor_plan.factors) {
            if (f != 2) { is_pure_power_of_two = false; break; }
        }

        if (is_pure_power_of_two) {
            unsigned p = 0; size_t temp = ctx.fft_len;
            while (temp > 1) { temp >>= 1; p++; }

            // HYBRID ENGINE CHOICE (see use_bailey_engine):
            if (use_bailey_engine(p)) {
                // Bailey 2D (all even powers, e.g. 2^16 = 65536, and some odd ones)
                fft_bailey_2d(ctx, inverse);
            }
            else {
                // Stockham 1D (odd powers, e.g. 2^19 = 524288)
                double* res_re = fft_stockham_radix8(ctx, inverse);
                if (res_re != ctx.re) {
                    std::swap(ctx.re, ctx.re2);
                    std::swap(ctx.im, ctx.im2);
                }
            }
        }
        else {
            fft_mixed_radix_execute(ctx, ctx.factor_plan, inverse);
        }

        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);
        };

    size_t split_chunks = ctx.fft_len / 2;
    size_t half_bytes = split_chunks * sizeof(double);

    pack_to_fft_buffers(Y, n_limbs, ctx.re, ctx.fft_len);
    std::memcpy(ctx.g_C_re, ctx.re, half_bytes);
    std::memset(ctx.g_C_re + split_chunks, 0, half_bytes);
    std::memcpy(ctx.g_D_re, ctx.re + split_chunks, half_bytes);
    std::memset(ctx.g_D_re + split_chunks, 0, half_bytes);

    pack_to_fft_buffers(X, n_limbs, ctx.re, ctx.fft_len);
    std::memcpy(ctx.k_A_re, ctx.re, half_bytes);
    std::memset(ctx.k_A_re + split_chunks, 0, half_bytes);
    std::memcpy(ctx.k_B_re, ctx.re + split_chunks, half_bytes);
    std::memset(ctx.k_B_re + split_chunks, 0, half_bytes);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.k_A_re[i];
        ctx.im[i] = ctx.k_B_re[i];
    }
    run_fft_on_buffer(ctx.re, ctx.im, false);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.k_A_re[i] = ctx.re[i];
        ctx.k_A_im[i] = ctx.im[i];
    }

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.g_C_re[i];
        ctx.im[i] = ctx.g_D_re[i];
    }
    run_fft_on_buffer(ctx.re, ctx.im, false);

    double inv_n = 1.0 / (double)ctx.fft_len;

    {
        double A0 = ctx.k_A_re[0], B0 = ctx.k_A_im[0];
        double C0 = ctx.re[0], D0 = ctx.im[0];

        ctx.g_C_re[0] = (A0 * C0) * inv_n;
        ctx.g_C_im[0] = (B0 * D0) * inv_n;
        ctx.k_AB_re[0] = (A0 * D0 + B0 * C0) * inv_n;
        ctx.k_AB_im[0] = 0.0;
    }

#pragma omp parallel for simd schedule(static)
    for (size_t i = 1; i < ctx.fft_len; i++) {
        size_t j = ctx.fft_len - i;

        double Rk_x = ctx.k_A_re[i], Ik_x = ctx.k_A_im[i];
        double Rj_x = ctx.k_A_re[j], Ij_x = ctx.k_A_im[j];
        double Ak_re = (Rk_x + Rj_x) * 0.5, Ak_im = (Ik_x - Ij_x) * 0.5;
        double Bk_re = (Ik_x + Ij_x) * 0.5, Bk_im = (Rj_x - Rk_x) * 0.5;

        double Rk_y = ctx.re[i], Ik_y = ctx.im[i];
        double Rj_y = ctx.re[j], Ij_y = ctx.im[j];
        double Ck_re = (Rk_y + Rj_y) * 0.5, Ck_im = (Ik_y - Ij_y) * 0.5;
        double Dk_re = (Ik_y + Ij_y) * 0.5, Dk_im = (Rj_y - Rk_y) * 0.5;

        double AC_re = Ak_re * Ck_re - Ak_im * Ck_im;
        double AC_im = Ak_re * Ck_im + Ak_im * Ck_re;

        double BD_re = Bk_re * Dk_re - Bk_im * Dk_im;
        double BD_im = Bk_re * Dk_im + Bk_im * Dk_re;

        double AD_BC_re = (Ak_re * Dk_re - Ak_im * Dk_im) + (Bk_re * Ck_re - Bk_im * Ck_im);
        double AD_BC_im = (Ak_re * Dk_im + Ak_im * Dk_re) + (Bk_re * Ck_im + Bk_im * Ck_re);

        ctx.g_C_re[i] = (AC_re - BD_im) * inv_n;
        ctx.g_C_im[i] = (AC_im + BD_re) * inv_n;
        ctx.k_AB_re[i] = AD_BC_re * inv_n;
        ctx.k_AB_im[i] = AD_BC_im * inv_n;
    }

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.g_C_re[i];
        ctx.im[i] = ctx.g_C_im[i];
    }
    run_fft_on_buffer(ctx.re, ctx.im, true);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.k_A_re[i] = ctx.re[i];
        ctx.k_B_re[i] = ctx.im[i];
    }

    run_fft_on_buffer(ctx.k_AB_re, ctx.k_AB_im, true);

   
    unpack_karatsuba_overlap_add_vbmi(out, n_limbs, ctx, n_bits, k_val, ctx.k_A_re, ctx.k_AB_re, ctx.k_B_re);
}
