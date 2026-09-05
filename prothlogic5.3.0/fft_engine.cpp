// Copyright (C) Damian Koba.

#include "fft_engine.h"
#include "llr_fft_pack.h"
#include "checkpoint_manager.h"
#include "fft_engine_avx2.h"
#include "fft_pack_vbmi.h"
#include "fft_factorizer.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#pragma GCC target("avx512f,avx512dq,fma")
#include <immintrin.h>
#include <sys/mman.h>
#include <omp.h>
#include <vector>

extern bool check_avx512_support();

bool fft_init(FFTContext& ctx, FFTBackend backend, size_t fft_len) {
    ctx.backend = backend;
    ctx.fft_len = fft_len;
    ctx.factor_plan = generate_fft_plan(fft_len);
    ctx.last_max_diff = 0.0;

    
    ctx.use_avx2_fallback = false;

    if (ctx.use_avx2_fallback) {
        return fft_init_avx2(ctx, backend, fft_len);
    }

    size_t table_size = (fft_len + 64) * sizeof(double);
    size_t total_size = table_size * 22;
    void* big_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (big_mem == MAP_FAILED) {
        big_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (big_mem == MAP_FAILED) return false;
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

        if (p % 2 == 0) return "Bailey 2D (Radix-4) [FFT_Len = 2^" + std::to_string(p) + "]";
        else return "Stockham 1D (Radix-8/4/2) [FFT_Len = 2^" + std::to_string(p) + "]";
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
                // FALLBACK 1 wektor
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
                // FALLBACK Skalar
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
                // FALLBACK 1 wektor
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
                // FALLBACK Skalar
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
// TRANSPOZER AVX-512 (8x8  ZMM, ZERO MEMORY SPILL)
// =================================================================
inline __attribute__((always_inline)) void transpose8x8_pd(__m512d& r0, __m512d& r1, __m512d& r2, __m512d& r3,
    __m512d& r4, __m512d& r5, __m512d& r6, __m512d& r7) {
    //(Unpack)
    __m512d t0 = _mm512_unpacklo_pd(r0, r1);
    __m512d t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    //  (Shuffle)
    __m512d u0 = _mm512_shuffle_f64x2(t0, t2, 0x50);
    __m512d u1 = _mm512_shuffle_f64x2(t1, t3, 0x50);
    __m512d u2 = _mm512_shuffle_f64x2(t0, t2, 0xFA);
    __m512d u3 = _mm512_shuffle_f64x2(t1, t3, 0xFA);
    __m512d u4 = _mm512_shuffle_f64x2(t4, t6, 0x50);
    __m512d u5 = _mm512_shuffle_f64x2(t5, t7, 0x50);
    __m512d u6 = _mm512_shuffle_f64x2(t4, t6, 0xFA);
    __m512d u7 = _mm512_shuffle_f64x2(t5, t7, 0xFA);

    // (Insert/Extract)
    r0 = _mm512_insertf64x4(u0, _mm512_castpd512_pd256(u4), 1);
    r1 = _mm512_insertf64x4(u1, _mm512_castpd512_pd256(u5), 1);
    r2 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_extractf64x4_pd(u0, 1)), _mm512_extractf64x4_pd(u4, 1), 1);
    r3 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_extractf64x4_pd(u1, 1)), _mm512_extractf64x4_pd(u5, 1), 1);
    r4 = _mm512_insertf64x4(u2, _mm512_castpd512_pd256(u6), 1);
    r5 = _mm512_insertf64x4(u3, _mm512_castpd512_pd256(u7), 1);
    r6 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_extractf64x4_pd(u2, 1)), _mm512_extractf64x4_pd(u6, 1), 1);
    r7 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_extractf64x4_pd(u3, 1)), _mm512_extractf64x4_pd(u7, 1), 1);
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

    if (r_len <= 16 && c_len <= 16) {
        
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

    if (r_len <= 16 && c_len <= 16) {
        double sign = inverse ? -1.0 : 1.0;
        for (size_t j = c1; j < c2; ++j) {
            for (size_t i = r1; i < r2; ++i) {
                double r = src_re[i * cols + j];
                double im = src_im[i * cols + j];
                size_t idx = j * i;

                double wr = ctx.w1_re[idx];
                double wi = ctx.w1_im[idx] * sign;

                dst_re[j * rows + i] = r * wr - im * wi;
                dst_im[j * rows + i] = r * wi + im * wr;
            }
        }
    }
    else if (r_len >= c_len) {
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
// STOCKHAM 1D multi-threads
// ---------------------------------------------------------
double* fft_stockham_radix8(FFTContext& ctx, bool inverse) {
    const size_t n = ctx.fft_len;
    double* in_re = ctx.re; double* in_im = ctx.im;
    double* out_re = ctx.re2; double* out_im = ctx.im2;
    const double sign = inverse ? -1.0 : 1.0;

    const __m512d v_sign = _mm512_set1_pd(sign);
    const __m512d v_neg_sign = _mm512_set1_pd(-sign);
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    const __m512d v_neg_C = _mm512_set1_pd(-0.70710678118654752440);

    for (size_t m = 1; m < n; ) {
        size_t remaining = n / m;

        if (remaining >= 8) {
            size_t stride = remaining >> 3;
            if (stride >= 8) {
#pragma omp parallel for schedule(static) if(m >= 4)
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
            else {
                double C = 0.70710678118654752440;
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
// CACHE-FRIENDLY RADIX-4 (Bailey 2D) - ILP UNROLLED
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

    for (size_t m = 1; m < n; m <<= 2) {
        size_t stride = n / (m << 2);

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

                    size_t prefetch_offset = 128;
                    _mm_prefetch((const char*)&cur_in_re[i0_A + prefetch_offset], _MM_HINT_NTA);
                    _mm_prefetch((const char*)&cur_in_re[i1_A + prefetch_offset], _MM_HINT_NTA);
                    _mm_prefetch((const char*)&cur_in_re[i2_A + prefetch_offset], _MM_HINT_NTA);
                    _mm_prefetch((const char*)&cur_in_re[i3_A + prefetch_offset], _MM_HINT_NTA);

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

                    _mm512_store_pd(&cur_out_re[o0_A], _mm512_add_pd(s0r_A, s2r_A)); _mm512_store_pd(&cur_out_im[o0_A], _mm512_add_pd(s0i_A, s2i_A));
                    _mm512_store_pd(&cur_out_re[o0_B], _mm512_add_pd(s0r_B, s2r_B)); _mm512_store_pd(&cur_out_im[o0_B], _mm512_add_pd(s0i_B, s2i_B));
                    _mm512_store_pd(&cur_out_re[o1_A], _mm512_add_pd(s1r_A, d3r_A)); _mm512_store_pd(&cur_out_im[o1_A], _mm512_add_pd(s1i_A, d3i_A));
                    _mm512_store_pd(&cur_out_re[o1_B], _mm512_add_pd(s1r_B, d3r_B)); _mm512_store_pd(&cur_out_im[o1_B], _mm512_add_pd(s1i_B, d3i_B));
                    _mm512_store_pd(&cur_out_re[o2_A], _mm512_sub_pd(s0r_A, s2r_A)); _mm512_store_pd(&cur_out_im[o2_A], _mm512_sub_pd(s0i_A, s2i_A));
                    _mm512_store_pd(&cur_out_re[o2_B], _mm512_sub_pd(s0r_B, s2r_B)); _mm512_store_pd(&cur_out_im[o2_B], _mm512_sub_pd(s0i_B, s2i_B));
                    _mm512_store_pd(&cur_out_re[o3_A], _mm512_sub_pd(s1r_A, d3r_A)); _mm512_store_pd(&cur_out_im[o3_A], _mm512_sub_pd(s1i_A, d3i_A));
                    _mm512_store_pd(&cur_out_re[o3_B], _mm512_sub_pd(s1r_B, d3r_B)); _mm512_store_pd(&cur_out_im[o3_B], _mm512_sub_pd(s1i_B, d3i_B));
                }

                
                for (; k < stride; k += 8) {
                    size_t i0 = k + j * (stride << 2);
                    size_t i1 = i0 + stride; size_t i2 = i1 + stride; size_t i3 = i2 + stride;

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
                    _mm512_store_pd(&cur_out_re[o0], _mm512_add_pd(s0r, s2r));
                    _mm512_store_pd(&cur_out_im[o0], _mm512_add_pd(s0i, s2i));
                    _mm512_store_pd(&cur_out_re[o1], _mm512_add_pd(s1r, d3r));
                    _mm512_store_pd(&cur_out_im[o1], _mm512_add_pd(s1i, d3i));
                    _mm512_store_pd(&cur_out_re[o2], _mm512_sub_pd(s0r, s2r));
                    _mm512_store_pd(&cur_out_im[o2], _mm512_sub_pd(s0i, s2i));
                    _mm512_store_pd(&cur_out_re[o3], _mm512_sub_pd(s1r, d3r));
                    _mm512_store_pd(&cur_out_im[o3], _mm512_sub_pd(s1i, d3i));
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
        std::swap(cur_in_re, cur_out_re);
        std::swap(cur_in_im, cur_out_im);
    }
    return cur_in_re;
}

// ---------------------------------------------------------
// CACHE-FRIENDLY BAILEY 2D
// ---------------------------------------------------------
void fft_bailey_2d(FFTContext& ctx, bool inverse) {
    size_t N = ctx.fft_len;
    size_t n1 = 1, n2 = 1, temp = N;
    while (temp > 1) {
        if (n1 <= n2) n1 *= 4;
        else n2 *= 4;
        temp /= 4;
    }

    if (ctx.r_w1r.empty()) {
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

    if (!inverse) {
        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; i++) {
            double* r_in_re = &ctx.re[i * n1]; double* r_in_im = &ctx.im[i * n1];
            double* r_out_re = &ctx.re2[i * n1]; double* r_out_im = &ctx.im2[i * n1];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n1,
                ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i, false);
            if (res != r_in_re) {
                std::memcpy(r_in_re, r_out_re, n1 * sizeof(double));
                std::memcpy(r_in_im, r_out_im, n1 * sizeof(double));
            }
        }

        transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1, ctx, false);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; i++) {
            double* r_in_re = &ctx.re[i * n2]; double* r_in_im = &ctx.im[i * n2];
            double* r_out_re = &ctx.re2[i * n2]; double* r_out_im = &ctx.im2[i * n2];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n2,
                ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i, false);
            if (res != r_in_re) {
                std::memcpy(r_in_re, r_out_re, n2 * sizeof(double));
                std::memcpy(r_in_im, r_out_im, n2 * sizeof(double));
            }
        }

        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

    }
    else {
        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; i++) {
            double* r_in_re = &ctx.re[i * n2]; double* r_in_im = &ctx.im[i * n2];
            double* r_out_re = &ctx.re2[i * n2]; double* r_out_im = &ctx.im2[i * n2];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n2,
                ctx.c_w1r, ctx.c_w1i, ctx.c_w2r, ctx.c_w2i, ctx.c_w3r, ctx.c_w3i, true);
            if (res != r_in_re) {
                std::memcpy(r_in_re, r_out_re, n2 * sizeof(double));
                std::memcpy(r_in_im, r_out_im, n2 * sizeof(double));
            }
        }

        transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, ctx, true);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; i++) {
            double* r_in_re = &ctx.re[i * n1]; double* r_in_im = &ctx.im[i * n1];
            double* r_out_re = &ctx.re2[i * n1]; double* r_out_im = &ctx.im2[i * n1];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n1,
                ctx.r_w1r, ctx.r_w1i, ctx.r_w2r, ctx.r_w2i, ctx.r_w3r, ctx.r_w3i, true);
            if (res != r_in_re) {
                std::memcpy(r_in_re, r_out_re, n1 * sizeof(double));
                std::memcpy(r_in_im, r_out_im, n1 * sizeof(double));
            }
        }

        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);
    }
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


extern void direct_pack_square(const uint64_t* limbs, size_t n_limbs, FFTContext& ctx);
extern void unpack_karatsuba_overlap_add_vbmi(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val, int target_bits, double* ptr_A2, double* ptr_AB, double* ptr_B2);

void fft_square_karatsuba(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
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
            if (p % 2 == 0) fft_bailey_2d(ctx, inverse);
            else {
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

    {
        double A_re = ctx.re[0], B_re = ctx.im[0];
        ctx.k_A_re[0] = (A_re * A_re) * inv_n;
        ctx.k_A_im[0] = (B_re * B_re) * inv_n;
        ctx.k_AB_re[0] = (A_re * B_re * 2.0) * inv_n;
        ctx.k_AB_im[0] = 0.0;
    }

#pragma omp parallel for simd schedule(static)
    for (size_t i = 1; i < ctx.fft_len; i++) {
        size_t j = ctx.fft_len - i;
        double Rk = ctx.re[i], Ik = ctx.im[i];
        double Rj = ctx.re[j], Ij = ctx.im[j];

        double Ak_re = (Rk + Rj) * 0.5, Ak_im = (Ik - Ij) * 0.5;
        double Bk_re = (Ik + Ij) * 0.5, Bk_im = (Rj - Rk) * 0.5;

        double A2_re = Ak_re * Ak_re - Ak_im * Ak_im;
        double A2_im = 2.0 * Ak_re * Ak_im;
        double B2_re = Bk_re * Bk_re - Bk_im * Bk_im;
        double B2_im = 2.0 * Bk_re * Bk_im;
        double AB_re = Ak_re * Bk_re - Ak_im * Bk_im;
        double AB_im = Ak_re * Bk_im + Ak_im * Bk_re;

        ctx.k_A_re[i] = (A2_re - B2_im) * inv_n;
        ctx.k_A_im[i] = (A2_im + B2_re) * inv_n;
        ctx.k_AB_re[i] = (AB_re * 2.0) * inv_n;
        ctx.k_AB_im[i] = (AB_im * 2.0) * inv_n;
    }

    
    std::swap(ctx.re, ctx.k_A_re);
    std::swap(ctx.im, ctx.k_A_im);

    auto t_sq_1 = clock::now();

    auto t_inv_0 = clock::now();
    run_fft_on_buffer(ctx.re, ctx.im, true);
    run_fft_on_buffer(ctx.k_AB_re, ctx.k_AB_im, true);
    auto t_inv_1 = clock::now();

    
    auto t_unpack_0 = clock::now();
    unpack_karatsuba_overlap_add_vbmi(v, n_limbs, ctx, n_bits, k_val, 16, ctx.re, ctx.k_AB_re, ctx.im);
    auto t_unpack_1 = clock::now();

    ctx.prof_pack += secd(t_pack_1 - t_pack_0).count();
    ctx.prof_fwd += secd(t_fwd_1 - t_fwd_0).count();
    ctx.prof_square += secd(t_sq_1 - t_sq_0).count();
    ctx.prof_inv += secd(t_inv_1 - t_inv_0).count();
    ctx.prof_unpack += secd(t_unpack_1 - t_unpack_0).count();
    ctx.prof_calls++;
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
            if (p % 2 == 0) fft_bailey_2d(ctx, inverse);
            else {
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

    extern void pack_to_fft_buffers(const uint64_t * limbs, size_t n_limbs, double* out, size_t n_fft);
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

   
    unpack_karatsuba_overlap_add_vbmi(out, n_limbs, ctx, n_bits, k_val, 16, ctx.k_A_re, ctx.k_AB_re, ctx.k_B_re);
}