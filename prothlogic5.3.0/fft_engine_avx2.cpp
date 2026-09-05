// Copyright (C) Damian Koba.
//test
#include "fft_engine_avx2.h"
#include "llr_fft_pack.h"
#include "checkpoint_manager.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <sys/mman.h>
#include <omp.h>
#include <vector>

bool fft_init_avx2(FFTContext& ctx, FFTBackend backend, size_t fft_len) {
    ctx.backend = backend;
    ctx.fft_len = fft_len;
    ctx.last_max_diff = 0.0;

    size_t table_size = (fft_len + 64) * sizeof(double);
    size_t total_size = table_size * 22;
    void* big_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (big_mem == MAP_FAILED) return false;

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

    // Initialization of weights using AVX2 (4 doubles per vector)
    size_t i = 0;
    for (; i + 3 < fft_len; i += 4) {
        alignas(32) double temp_w1_re[4], temp_w1_im[4];
        alignas(32) double temp_w2_re[4], temp_w2_im[4];
        alignas(32) double temp_w3_re[4], temp_w3_im[4];
        alignas(32) double temp_proth_re[4], temp_proth_im[4];

        for (int k = 0; k < 4; ++k) {
            double i_d = static_cast<double>(i + k);
            double len_d = static_cast<double>(fft_len);

            double ang = -2.0 * M_PI * i_d / len_d;
            temp_w1_re[k] = std::cos(ang);       temp_w1_im[k] = std::sin(ang);
            temp_w2_re[k] = std::cos(2.0 * ang); temp_w2_im[k] = std::sin(2.0 * ang);
            temp_w3_re[k] = std::cos(3.0 * ang); temp_w3_im[k] = std::sin(3.0 * ang);

            double p_ang = -M_PI * i_d / len_d;
            temp_proth_re[k] = std::cos(p_ang);  temp_proth_im[k] = std::sin(p_ang);
        }

        __m256d v_w1_re = _mm256_load_pd(temp_w1_re);
        __m256d v_w1_im = _mm256_load_pd(temp_w1_im);
        __m256d v_w2_re = _mm256_load_pd(temp_w2_re);
        __m256d v_w2_im = _mm256_load_pd(temp_w2_im);
        __m256d v_w3_re = _mm256_load_pd(temp_w3_re);
        __m256d v_w3_im = _mm256_load_pd(temp_w3_im);
        __m256d v_proth_re = _mm256_load_pd(temp_proth_re);
        __m256d v_proth_im = _mm256_load_pd(temp_proth_im);

        _mm256_storeu_pd(&ctx.w1_re[i], v_w1_re);
        _mm256_storeu_pd(&ctx.w1_im[i], v_w1_im);
        _mm256_storeu_pd(&ctx.w2_re[i], v_w2_re);
        _mm256_storeu_pd(&ctx.w2_im[i], v_w2_im);
        _mm256_storeu_pd(&ctx.w3_re[i], v_w3_re);
        _mm256_storeu_pd(&ctx.w3_im[i], v_w3_im);
        _mm256_storeu_pd(&ctx.proth_w_re[i], v_proth_re);
        _mm256_storeu_pd(&ctx.proth_w_im[i], v_proth_im);
    }

    for (; i < fft_len; ++i) {
        double i_d = static_cast<double>(i);
        double len_d = static_cast<double>(fft_len);
        double ang = -2.0 * M_PI * i_d / len_d;
        ctx.w1_re[i] = std::cos(ang);       ctx.w1_im[i] = std::sin(ang);
        ctx.w2_re[i] = std::cos(2.0 * ang); ctx.w2_im[i] = std::sin(2.0 * ang);
        ctx.w3_re[i] = std::cos(3.0 * ang); ctx.w3_im[i] = std::sin(3.0 * ang);
    }

    std::string db_file = "proth_state_N" + std::to_string(fft_len) + ".bin";
    uint64_t recovered_k = 0;
    load_checkpoint(recovered_k, fft_len, reinterpret_cast<uint64_t*>(ctx.re), db_file);

    return true;
}

double* fft_stockham_radix8_avx2(FFTContext& ctx, bool inverse) {
    const size_t n = ctx.fft_len;
    double* in_re = ctx.re; double* in_im = ctx.im;
    double* out_re = ctx.re2; double* out_im = ctx.im2;
    const double sign = inverse ? -1.0 : 1.0;

    const __m256d v_sign = _mm256_set1_pd(sign);
    const __m256d v_neg_sign = _mm256_set1_pd(-sign);
    const __m256d v_C = _mm256_set1_pd(0.70710678118654752440);
    const __m256d v_neg_C = _mm256_set1_pd(-0.70710678118654752440);

    for (size_t m = 1; m < n; ) {
        size_t remaining = n / m;

        
        if (remaining >= 32) {
            size_t stride = remaining >> 3;

#pragma omp parallel for schedule(static) if(m >= 4)
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;

                __m256d v_w1r = _mm256_set1_pd(ctx.w1_re[tw]);       __m256d v_w1i = _mm256_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);
                __m256d v_w2r = _mm256_set1_pd(ctx.w1_re[2 * tw]);   __m256d v_w2i = _mm256_set1_pd(inverse ? -ctx.w1_im[2 * tw] : ctx.w1_im[2 * tw]);
                __m256d v_w3r = _mm256_set1_pd(ctx.w1_re[3 * tw]);   __m256d v_w3i = _mm256_set1_pd(inverse ? -ctx.w1_im[3 * tw] : ctx.w1_im[3 * tw]);
                __m256d v_w4r = _mm256_set1_pd(ctx.w1_re[4 * tw]);   __m256d v_w4i = _mm256_set1_pd(inverse ? -ctx.w1_im[4 * tw] : ctx.w1_im[4 * tw]);
                __m256d v_w5r = _mm256_set1_pd(ctx.w1_re[5 * tw]);   __m256d v_w5i = _mm256_set1_pd(inverse ? -ctx.w1_im[5 * tw] : ctx.w1_im[5 * tw]);
                __m256d v_w6r = _mm256_set1_pd(ctx.w1_re[6 * tw]);   __m256d v_w6i = _mm256_set1_pd(inverse ? -ctx.w1_im[6 * tw] : ctx.w1_im[6 * tw]);
                __m256d v_w7r = _mm256_set1_pd(ctx.w1_re[7 * tw]);   __m256d v_w7i = _mm256_set1_pd(inverse ? -ctx.w1_im[7 * tw] : ctx.w1_im[7 * tw]);

                for (size_t k = 0; k < stride; k += 4) {
                    size_t i0 = k + j * (stride << 3);
                    size_t i1 = i0 + stride; size_t i2 = i1 + stride; size_t i3 = i2 + stride;
                    size_t i4 = i3 + stride; size_t i5 = i4 + stride; size_t i6 = i5 + stride; size_t i7 = i6 + stride;

                    __m256d r0 = _mm256_loadu_pd(&in_re[i0]), i0v = _mm256_loadu_pd(&in_im[i0]);
                    __m256d r1 = _mm256_loadu_pd(&in_re[i1]), i1v = _mm256_loadu_pd(&in_im[i1]);
                    __m256d r2 = _mm256_loadu_pd(&in_re[i2]), i2v = _mm256_loadu_pd(&in_im[i2]);
                    __m256d r3 = _mm256_loadu_pd(&in_re[i3]), i3v = _mm256_loadu_pd(&in_im[i3]);
                    __m256d r4 = _mm256_loadu_pd(&in_re[i4]), i4v = _mm256_loadu_pd(&in_im[i4]);
                    __m256d r5 = _mm256_loadu_pd(&in_re[i5]), i5v = _mm256_loadu_pd(&in_im[i5]);
                    __m256d r6 = _mm256_loadu_pd(&in_re[i6]), i6v = _mm256_loadu_pd(&in_im[i6]);
                    __m256d r7 = _mm256_loadu_pd(&in_re[i7]), i7v = _mm256_loadu_pd(&in_im[i7]);

                    __m256d t1r = _mm256_fmsub_pd(r1, v_w1r, _mm256_mul_pd(i1v, v_w1i)); __m256d t1i = _mm256_fmadd_pd(r1, v_w1i, _mm256_mul_pd(i1v, v_w1r));
                    __m256d t2r = _mm256_fmsub_pd(r2, v_w2r, _mm256_mul_pd(i2v, v_w2i)); __m256d t2i = _mm256_fmadd_pd(r2, v_w2i, _mm256_mul_pd(i2v, v_w2r));
                    __m256d t3r = _mm256_fmsub_pd(r3, v_w3r, _mm256_mul_pd(i3v, v_w3i)); __m256d t3i = _mm256_fmadd_pd(r3, v_w3i, _mm256_mul_pd(i3v, v_w3r));
                    __m256d t4r = _mm256_fmsub_pd(r4, v_w4r, _mm256_mul_pd(i4v, v_w4i)); __m256d t4i = _mm256_fmadd_pd(r4, v_w4i, _mm256_mul_pd(i4v, v_w4r));
                    __m256d t5r = _mm256_fmsub_pd(r5, v_w5r, _mm256_mul_pd(i5v, v_w5i)); __m256d t5i = _mm256_fmadd_pd(r5, v_w5i, _mm256_mul_pd(i5v, v_w5r));
                    __m256d t6r = _mm256_fmsub_pd(r6, v_w6r, _mm256_mul_pd(i6v, v_w6i)); __m256d t6i = _mm256_fmadd_pd(r6, v_w6i, _mm256_mul_pd(i6v, v_w6r));
                    __m256d t7r = _mm256_fmsub_pd(r7, v_w7r, _mm256_mul_pd(i7v, v_w7i)); __m256d t7i = _mm256_fmadd_pd(r7, v_w7i, _mm256_mul_pd(i7v, v_w7r));

                    __m256d u0r = _mm256_add_pd(r0, t4r), u0i = _mm256_add_pd(i0v, t4i); __m256d u4r = _mm256_sub_pd(r0, t4r), u4i = _mm256_sub_pd(i0v, t4i);
                    __m256d u1r = _mm256_add_pd(t1r, t5r), u1i = _mm256_add_pd(t1i, t5i); __m256d u5r = _mm256_sub_pd(t1r, t5r), u5i = _mm256_sub_pd(t1i, t5i);
                    __m256d u2r = _mm256_add_pd(t2r, t6r), u2i = _mm256_add_pd(t2i, t6i); __m256d u6r = _mm256_sub_pd(t2r, t6r), u6i = _mm256_sub_pd(t2i, t6i);
                    __m256d u3r = _mm256_add_pd(t3r, t7r), u3i = _mm256_add_pd(t3i, t7i); __m256d u7r = _mm256_sub_pd(t3r, t7r), u7i = _mm256_sub_pd(t3i, t7i);

                    __m256d u5i_sign = _mm256_mul_pd(u5i, v_sign);
                    __m256d u5r_sign = _mm256_mul_pd(u5r, v_sign);
                    __m256d v5r = _mm256_mul_pd(v_C, _mm256_add_pd(u5r, u5i_sign));
                    __m256d v5i = _mm256_mul_pd(v_C, _mm256_sub_pd(u5i, u5r_sign));

                    __m256d v6r = _mm256_mul_pd(u6i, v_sign);
                    __m256d v6i = _mm256_mul_pd(u6r, v_neg_sign);

                    __m256d u7i_sign = _mm256_mul_pd(u7i, v_sign);
                    __m256d u7r_sign = _mm256_mul_pd(u7r, v_sign);
                    __m256d v7r = _mm256_mul_pd(v_C, _mm256_sub_pd(u7i_sign, u7r));
                    __m256d v7i = _mm256_mul_pd(v_neg_C, _mm256_add_pd(u7r_sign, u7i));

                    __m256d a0r = _mm256_add_pd(u0r, u2r), a0i = _mm256_add_pd(u0i, u2i); __m256d a1r = _mm256_sub_pd(u0r, u2r), a1i = _mm256_sub_pd(u0i, u2i);
                    __m256d a2r = _mm256_add_pd(u1r, u3r), a2i = _mm256_add_pd(u1i, u3i); __m256d a3r = _mm256_sub_pd(u1r, u3r), a3i = _mm256_sub_pd(u1i, u3i);
                    __m256d d3r = _mm256_mul_pd(a3i, v_sign), d3i = _mm256_mul_pd(a3r, v_neg_sign);

                    size_t o0 = k + j * stride;           size_t o1 = o0 + m * stride;
                    size_t o2 = o1 + m * stride;          size_t o3 = o2 + m * stride;
                    size_t o4 = o3 + m * stride;          size_t o5 = o4 + m * stride;
                    size_t o6 = o5 + m * stride;          size_t o7 = o6 + m * stride;

                    _mm256_storeu_pd(&out_re[o0], _mm256_add_pd(a0r, a2r)); _mm256_storeu_pd(&out_im[o0], _mm256_add_pd(a0i, a2i));
                    _mm256_storeu_pd(&out_re[o4], _mm256_sub_pd(a0r, a2r)); _mm256_storeu_pd(&out_im[o4], _mm256_sub_pd(a0i, a2i));
                    _mm256_storeu_pd(&out_re[o2], _mm256_add_pd(a1r, d3r)); _mm256_storeu_pd(&out_im[o2], _mm256_add_pd(a1i, d3i));
                    _mm256_storeu_pd(&out_re[o6], _mm256_sub_pd(a1r, d3r)); _mm256_storeu_pd(&out_im[o6], _mm256_sub_pd(a1i, d3i));

                    __m256d b0r = _mm256_add_pd(u4r, v6r), b0i = _mm256_add_pd(u4i, v6i); __m256d b1r = _mm256_sub_pd(u4r, v6r), b1i = _mm256_sub_pd(u4i, v6i);
                    __m256d b2r = _mm256_add_pd(v5r, v7r), b2i = _mm256_add_pd(v5i, v7i); __m256d b3r = _mm256_sub_pd(v5r, v7r), b3i = _mm256_sub_pd(v5i, v7i);
                    __m256d e3r = _mm256_mul_pd(b3i, v_sign), e3i = _mm256_mul_pd(b3r, v_neg_sign);

                    _mm256_storeu_pd(&out_re[o1], _mm256_add_pd(b0r, b2r)); _mm256_storeu_pd(&out_im[o1], _mm256_add_pd(b0i, b2i));
                    _mm256_storeu_pd(&out_re[o5], _mm256_sub_pd(b0r, b2r)); _mm256_storeu_pd(&out_im[o5], _mm256_sub_pd(b0i, b2i));
                    _mm256_storeu_pd(&out_re[o3], _mm256_add_pd(b1r, e3r)); _mm256_storeu_pd(&out_im[o3], _mm256_add_pd(b1i, e3i));
                    _mm256_storeu_pd(&out_re[o7], _mm256_sub_pd(b1r, e3r)); _mm256_storeu_pd(&out_im[o7], _mm256_sub_pd(b1i, e3i));
                }
            }
            m <<= 3; 
        }
        else if (remaining >= 16) {
            
            size_t stride = remaining >> 2;
#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;
                __m256d v_w1r = _mm256_set1_pd(ctx.w1_re[tw]);     __m256d v_w1i = _mm256_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);
                __m256d v_w2r = _mm256_set1_pd(ctx.w1_re[2 * tw]); __m256d v_w2i = _mm256_set1_pd(inverse ? -ctx.w1_im[2 * tw] : ctx.w1_im[2 * tw]);
                __m256d v_w3r = _mm256_set1_pd(ctx.w1_re[3 * tw]); __m256d v_w3i = _mm256_set1_pd(inverse ? -ctx.w1_im[3 * tw] : ctx.w1_im[3 * tw]);

                for (size_t k = 0; k < stride; k += 4) {
                    size_t i0 = k + j * (stride << 2);
                    size_t i1 = i0 + stride; size_t i2 = i1 + stride; size_t i3 = i2 + stride;

                    __m256d r0 = _mm256_loadu_pd(&in_re[i0]), i0v = _mm256_loadu_pd(&in_im[i0]);
                    __m256d r1 = _mm256_loadu_pd(&in_re[i1]), i1v = _mm256_loadu_pd(&in_im[i1]);
                    __m256d r2 = _mm256_loadu_pd(&in_re[i2]), i2v = _mm256_loadu_pd(&in_im[i2]);
                    __m256d r3 = _mm256_loadu_pd(&in_re[i3]), i3v = _mm256_loadu_pd(&in_im[i3]);

                    __m256d t1r = _mm256_fmsub_pd(r1, v_w1r, _mm256_mul_pd(i1v, v_w1i));
                    __m256d t1i = _mm256_fmadd_pd(r1, v_w1i, _mm256_mul_pd(i1v, v_w1r));
                    __m256d t2r = _mm256_fmsub_pd(r2, v_w2r, _mm256_mul_pd(i2v, v_w2i));
                    __m256d t2i = _mm256_fmadd_pd(r2, v_w2i, _mm256_mul_pd(i2v, v_w2r));
                    __m256d t3r = _mm256_fmsub_pd(r3, v_w3r, _mm256_mul_pd(i3v, v_w3i));
                    __m256d t3i = _mm256_fmadd_pd(r3, v_w3i, _mm256_mul_pd(i3v, v_w3r));

                    __m256d u0r = _mm256_add_pd(r0, t2r), u0i = _mm256_add_pd(i0v, t2i);
                    __m256d u2r = _mm256_sub_pd(r0, t2r), u2i = _mm256_sub_pd(i0v, t2i);
                    __m256d u1r = _mm256_add_pd(t1r, t3r), u1i = _mm256_add_pd(t1i, t3i);
                    __m256d u3r = _mm256_sub_pd(t1r, t3r), u3i = _mm256_sub_pd(t1i, t3i);

                    __m256d u3r_sign = _mm256_mul_pd(u3r, v_sign);
                    __m256d u3i_sign = _mm256_mul_pd(u3i, v_sign);

                    size_t o0 = k + j * stride;
                    size_t o1 = o0 + m * stride; size_t o2 = o1 + m * stride; size_t o3 = o2 + m * stride;

                    _mm256_storeu_pd(&out_re[o0], _mm256_add_pd(u0r, u1r)); _mm256_storeu_pd(&out_im[o0], _mm256_add_pd(u0i, u1i));
                    _mm256_storeu_pd(&out_re[o2], _mm256_sub_pd(u0r, u1r)); _mm256_storeu_pd(&out_im[o2], _mm256_sub_pd(u0i, u1i));
                    _mm256_storeu_pd(&out_re[o1], _mm256_add_pd(u2r, u3i_sign)); _mm256_storeu_pd(&out_im[o1], _mm256_sub_pd(u2i, u3r_sign));
                    _mm256_storeu_pd(&out_re[o3], _mm256_sub_pd(u2r, u3i_sign)); _mm256_storeu_pd(&out_im[o3], _mm256_add_pd(u2i, u3r_sign));
                }
            }
            m <<= 2;
        }
        else if (remaining >= 8) {
            
            size_t stride = remaining >> 1;
#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;
                __m256d v_wr = _mm256_set1_pd(ctx.w1_re[tw]);
                __m256d v_wi = _mm256_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);
                for (size_t k = 0; k < stride; k += 4) {
                    size_t i0 = k + j * (stride << 1); size_t i1 = i0 + stride;
                    __m256d r0 = _mm256_loadu_pd(&in_re[i0]), i0v = _mm256_loadu_pd(&in_im[i0]);
                    __m256d r1 = _mm256_loadu_pd(&in_re[i1]), i1v = _mm256_loadu_pd(&in_im[i1]);

                    __m256d t1r = _mm256_fmsub_pd(r1, v_wr, _mm256_mul_pd(i1v, v_wi));
                    __m256d t1i = _mm256_fmadd_pd(r1, v_wi, _mm256_mul_pd(i1v, v_wr));

                    size_t o0 = k + j * stride, o1 = o0 + m * stride;
                    _mm256_storeu_pd(&out_re[o0], _mm256_add_pd(r0, t1r)); _mm256_storeu_pd(&out_im[o0], _mm256_add_pd(i0v, t1i));
                    _mm256_storeu_pd(&out_re[o1], _mm256_sub_pd(r0, t1r)); _mm256_storeu_pd(&out_im[o1], _mm256_sub_pd(i0v, t1i));
                }
            }
            m <<= 1;
        }
        else if (remaining >= 2) {
            
            size_t stride = remaining >> 1;
#pragma omp parallel for schedule(static)
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;
                double wr = ctx.w1_re[tw], wi = inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw];
#pragma omp simd
                for (size_t k = 0; k < stride; k++) {
                    size_t i0 = k + j * (stride << 1); size_t i1 = i0 + stride;
                    double r0 = in_re[i0], i0v = in_im[i0], r1 = in_re[i1], i1v = in_im[i1];
                    double t1r = r1 * wr - i1v * wi, t1i = r1 * wi + i1v * wr;
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

void fft_square_karatsuba_avx2(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    pack_karatsuba_limbs(v, n_limbs, ctx);

    auto run_fft_on_buffer = [&](double*& target_re, double*& target_im, bool inverse) {
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);
        double* res_re = fft_stockham_radix8_avx2(ctx, inverse);
        if (res_re != ctx.re) {
            std::swap(ctx.re, ctx.re2);
            std::swap(ctx.im, ctx.im2);
        }
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);
        };

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.k_A_re[i];
        ctx.im[i] = ctx.k_B_re[i];
    }

    run_fft_on_buffer(ctx.re, ctx.im, false);

    double inv_n = 1.0 / (double)ctx.fft_len;
    {
        double A_re = ctx.re[0];
        double B_re = ctx.im[0];
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

        double Ak_re = (Rk + Rj) * 0.5;
        double Ak_im = (Ik - Ij) * 0.5;
        double Bk_re = (Ik + Ij) * 0.5;
        double Bk_im = (Rj - Rk) * 0.5;

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

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.k_A_re[i];
        ctx.im[i] = ctx.k_A_im[i];
    }

    run_fft_on_buffer(ctx.re, ctx.im, true);

#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.k_A_re[i] = ctx.re[i];
        ctx.k_B_re[i] = ctx.im[i];
    }

    run_fft_on_buffer(ctx.k_AB_re, ctx.k_AB_im, true);
    unpack_karatsuba_overlap_add_avx2(v, n_limbs, ctx, n_bits, k_val);
}

void unpack_karatsuba_overlap_add_avx2(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val) {
    const size_t n_fft = ctx.fft_len;
    const size_t split_chunks = n_fft / 2;

    size_t max_chunks = n_limbs * 8;
    size_t double_limbs = n_limbs * 2 + 2;
    std::vector<uint64_t> sq(double_limbs + 4, 0);

    uint64_t bit_buf = 0;
    int bit_count = 0;
    size_t sq_idx = 0;
    int64_t carry = 0;

    int bits_per_word = (n_limbs * 64ULL) / ctx.fft_len + 1;
    uint64_t mask = (1ULL << bits_per_word) - 1;

    
    __m256d v_block_max_err = _mm256_setzero_pd();

    
    for (size_t i = 0; i < max_chunks; i += 4) {
        __m256d v_A2 = _mm256_setzero_pd();
        __m256d v_AB = _mm256_setzero_pd();
        __m256d v_B2 = _mm256_setzero_pd();

        if (i < n_fft) {
            v_A2 = _mm256_loadu_pd(&ctx.k_A_re[i]);
        }
        if (i >= split_chunks && (i - split_chunks) < n_fft) {
            v_AB = _mm256_loadu_pd(&ctx.k_AB_re[i - split_chunks]);
        }
        if (i >= 2 * split_chunks && (i - 2 * split_chunks) < n_fft) {
            v_B2 = _mm256_loadu_pd(&ctx.k_B_re[i - 2 * split_chunks]);
        }

        // sum = A2 + AB + B2
        __m256d v_sum = _mm256_add_pd(_mm256_add_pd(v_A2, v_AB), v_B2);

        
        __m256d v_round = _mm256_round_pd(v_sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        
        __m256d v_err = _mm256_sub_pd(v_sum, v_round);
        __m256d v_abs_err = _mm256_max_pd(v_err, _mm256_sub_pd(_mm256_setzero_pd(), v_err));
        v_block_max_err = _mm256_max_pd(v_block_max_err, v_abs_err);

        
        alignas(32) double temp_d[4];
        _mm256_store_pd(temp_d, v_round);

        for (int k = 0; k < 4; k++) {
            if (i + k >= max_chunks) break; 

            int64_t ival = (int64_t)temp_d[k];
            ival += carry;
            carry = ival >> bits_per_word;
            uint64_t chunk = (uint64_t)(ival & mask);

            if (bit_count + bits_per_word >= 64) {
                int space = 64 - bit_count;
                bit_buf |= (chunk & ((1ULL << space) - 1)) << bit_count;
                if (sq_idx < sq.size()) sq[sq_idx] = bit_buf;
                sq_idx++;
                bit_buf = chunk >> space;
                bit_count = bits_per_word - space;
            }
            else {
                bit_buf |= (chunk << bit_count);
                bit_count += bits_per_word;
            }
        }
    }

    
    alignas(32) double err_arr[4];
    _mm256_store_pd(err_arr, v_block_max_err);
    double block_max_err = std::max({ err_arr[0], err_arr[1], err_arr[2], err_arr[3] });

    
    while (carry > 0 || bit_count > 0) {
        uint64_t chunk = carry & mask;
        carry >>= bits_per_word;

        if (bit_count + bits_per_word >= 64) {
            int space = 64 - bit_count;
            bit_buf |= (chunk & ((1ULL << space) - 1)) << bit_count;
            if (sq_idx < sq.size()) sq[sq_idx] = bit_buf;
            sq_idx++;
            bit_buf = chunk >> space;
            bit_count = bits_per_word - space;
        }
        else {
            bit_buf |= (chunk << bit_count);
            bit_count += bits_per_word;
        }

        if (carry == 0 && bit_count > 0) {
            if (sq_idx < sq.size()) sq[sq_idx] = bit_buf;
            break;
        }
    }

    ctx.last_max_diff = block_max_err;

    size_t carry_idx = max_chunks >> 2;
    while (carry > 0 && carry_idx < sq.size()) {
        unsigned __int128 sum_128 = (unsigned __int128)sq[carry_idx] + carry;
        sq[carry_idx] = (uint64_t)sum_128;
        carry = (int64_t)(sum_128 >> 64);
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

        unsigned __int128 sum_128 = (unsigned __int128)out[split_q] + rem_L;
        out[split_q] = (uint64_t)sum_128;
        uint64_t carry_add = (uint64_t)(sum_128 >> 64);

        if (split_q + 1 < n_limbs) {
            sum_128 = (unsigned __int128)out[split_q + 1] + rem_H + carry_add;
            out[split_q + 1] = (uint64_t)sum_128;
            carry_add = (uint64_t)(sum_128 >> 64);

            size_t idx = split_q + 2;
            while (carry_add > 0 && idx < n_limbs) {
                sum_128 = (unsigned __int128)out[idx] + carry_add;
                out[idx] = (uint64_t)sum_128;
                carry_add = (uint64_t)(sum_128 >> 64);
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

            unsigned __int128 sum_128 = (unsigned __int128)out[i] + val_to_add;
            out[i] = (uint64_t)sum_128;
            local_carry = (uint64_t)(sum_128 >> 64);
        }
    }
}

void fft_mul_karatsuba_avx2(FFTContext& ctx, uint64_t* out, const uint64_t* X, const uint64_t* Y, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    auto run_fft_on_buffer = [&](double*& target_re, double*& target_im, bool inverse) {
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);
        double* res_re = fft_stockham_radix8_avx2(ctx, inverse);
        if (res_re != ctx.re) {
            std::swap(ctx.re, ctx.re2);
            std::swap(ctx.im, ctx.im2);
        }
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);
        };

    
    pack_karatsuba_limbs(X, n_limbs, ctx);
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.k_A_re[i];
        ctx.im[i] = ctx.k_B_re[i];
    }
    run_fft_on_buffer(ctx.re, ctx.im, false);

    
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.g_C_re[i] = ctx.re[i];
        ctx.g_C_im[i] = ctx.im[i];
    }

    
    pack_karatsuba_limbs(Y, n_limbs, ctx);
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.k_A_re[i];
        ctx.im[i] = ctx.k_B_re[i];
    }
    run_fft_on_buffer(ctx.re, ctx.im, false);

    
    double inv_n = 1.0 / (double)ctx.fft_len;
    {
        double A1_re = ctx.g_C_re[0], B1_re = ctx.g_C_im[0];
        double A2_re = ctx.re[0], B2_re = ctx.im[0];

        ctx.k_A_re[0] = (A1_re * A2_re) * inv_n;
        ctx.k_A_im[0] = (B1_re * B2_re) * inv_n;
        ctx.k_AB_re[0] = (A1_re * B2_re + B1_re * A2_re) * inv_n;
        ctx.k_AB_im[0] = 0.0;
    }

#pragma omp parallel for simd schedule(static)
    for (size_t i = 1; i < ctx.fft_len; i++) {
        size_t j = ctx.fft_len - i;

        
        double X_Rk = ctx.g_C_re[i], X_Ik = ctx.g_C_im[i];
        double X_Rj = ctx.g_C_re[j], X_Ij = ctx.g_C_im[j];
        double X_Ak_re = (X_Rk + X_Rj) * 0.5, X_Ak_im = (X_Ik - X_Ij) * 0.5;
        double X_Bk_re = (X_Ik + X_Ij) * 0.5, X_Bk_im = (X_Rj - X_Rk) * 0.5;

        
        double Y_Rk = ctx.re[i], Y_Ik = ctx.im[i];
        double Y_Rj = ctx.re[j], Y_Ij = ctx.im[j];
        double Y_Ak_re = (Y_Rk + Y_Rj) * 0.5, Y_Ak_im = (Y_Ik - Y_Ij) * 0.5;
        double Y_Bk_re = (Y_Ik + Y_Ij) * 0.5, Y_Bk_im = (Y_Rj - Y_Rk) * 0.5;

        
        double A2_re = X_Ak_re * Y_Ak_re - X_Ak_im * Y_Ak_im;
        double A2_im = X_Ak_re * Y_Ak_im + X_Ak_im * Y_Ak_re;

        double B2_re = X_Bk_re * Y_Bk_re - X_Bk_im * Y_Bk_im;
        double B2_im = X_Bk_re * Y_Bk_im + X_Bk_im * Y_Bk_re;

        double AB_re = X_Ak_re * Y_Bk_re - X_Ak_im * Y_Bk_im + X_Bk_re * Y_Ak_re - X_Bk_im * Y_Ak_im;
        double AB_im = X_Ak_re * Y_Bk_im + X_Ak_im * Y_Bk_re + X_Bk_re * Y_Ak_im + X_Bk_im * Y_Ak_re;

        ctx.k_A_re[i] = (A2_re - B2_im) * inv_n;
        ctx.k_A_im[i] = (A2_im + B2_re) * inv_n;
        ctx.k_AB_re[i] = AB_re * inv_n;
        ctx.k_AB_im[i] = AB_im * inv_n;
    }

   
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.k_A_re[i];
        ctx.im[i] = ctx.k_A_im[i];
    }
    run_fft_on_buffer(ctx.re, ctx.im, true);
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.k_A_re[i] = ctx.re[i];
        ctx.k_B_re[i] = ctx.im[i];
    }

    
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.re[i] = ctx.k_AB_re[i];
        ctx.im[i] = ctx.k_AB_im[i];
    }
    run_fft_on_buffer(ctx.re, ctx.im, true);
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i++) {
        ctx.k_AB_re[i] = ctx.re[i];
        ctx.k_AB_im[i] = ctx.im[i];
    }

    
    unpack_karatsuba_overlap_add_avx2(out, n_limbs, ctx, n_bits, k_val);
}