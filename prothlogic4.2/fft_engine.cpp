// Copyright (C) Damian Koba.

#include "fft_engine.h"
#include "llr_fft_pack.h"
#include "checkpoint_manager.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <sys/mman.h>
#include <omp.h>
#include <vector>


bool fft_init(FFTContext& ctx, FFTBackend backend, size_t fft_len) {
    ctx.backend = backend;
    ctx.fft_len = fft_len;
    ctx.last_max_diff = 0.0;

    size_t table_size = (fft_len + 64) * sizeof(double);
    size_t total_size = table_size * 18;   
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

    
    for (size_t i = 0; i < fft_len; ++i) {
        double i_d = static_cast<double>(i);
        double len_d = static_cast<double>(fft_len);

        double ang = -2.0 * M_PI * i_d / len_d;
        ctx.w1_re[i] = std::cos(ang); ctx.w1_im[i] = std::sin(ang);
        ctx.w2_re[i] = std::cos(2.0 * ang); ctx.w2_im[i] = std::sin(2.0 * ang);
        ctx.w3_re[i] = std::cos(3.0 * ang); ctx.w3_im[i] = std::sin(3.0 * ang);

        double p_ang = -M_PI * i_d / len_d;
        ctx.proth_w_re[i] = std::cos(p_ang); ctx.proth_w_im[i] = std::sin(p_ang);
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
void transpose_blocked_complex(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t n1, size_t n2) {    
    const size_t BLOCK_SIZE = 8;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n1; i += BLOCK_SIZE) {
        for (size_t j = 0; j < n2; j += BLOCK_SIZE) {

            size_t max_i = std::min(i + BLOCK_SIZE, n1);
            size_t max_j = std::min(j + BLOCK_SIZE, n2);

            for (size_t ii = i; ii < max_i; ii++) {                
                _mm_prefetch((const char*)&src_re[ii * n2 + j + 32], _MM_HINT_T0);
                _mm_prefetch((const char*)&src_im[ii * n2 + j + 32], _MM_HINT_T0);

                for (size_t jj = j; jj < max_j; jj++) {
                    dst_re[jj * n1 + ii] = src_re[ii * n2 + jj];
                    dst_im[jj * n1 + ii] = src_im[ii * n2 + jj];
                }
            }
        }
    }
}

void transpose_and_twiddle(const double* __restrict src_re, const double* __restrict src_im,
    double* __restrict dst_re, double* __restrict dst_im,
    size_t n2, size_t n1, FFTContext& ctx, bool inverse) {

    const size_t BLOCK_SIZE = 8;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n2; i += BLOCK_SIZE) {
        for (size_t j = 0; j < n1; j += BLOCK_SIZE) {

            size_t max_i = std::min(i + BLOCK_SIZE, n2);
            size_t max_j = std::min(j + BLOCK_SIZE, n1);

            for (size_t ii = i; ii < max_i; ii++) {               
                _mm_prefetch((const char*)&src_re[ii * n1 + j + 32], _MM_HINT_T0);
                _mm_prefetch((const char*)&src_im[ii * n1 + j + 32], _MM_HINT_T0);

                for (size_t jj = j; jj < max_j; jj++) {
                    double r = src_re[ii * n1 + jj];
                    double im = src_im[ii * n1 + jj];

                    size_t idx = jj * ii;
                    double wr = ctx.w1_re[idx];
                    double wi = ctx.w1_im[idx];

                    if (inverse) wi = -wi;                    
                    dst_re[jj * n2 + ii] = r * wr - im * wi;
                    dst_im[jj * n2 + ii] = r * wi + im * wr;
                }
            }
        }
    }
}

// ---------------------------------------------------------
// WERSJA 1: STOCKHAM 1D WIELOWĄTKOWY
// ---------------------------------------------------------
double* fft_stockham_radix4(FFTContext& ctx, bool inverse) {
    const size_t n = ctx.fft_len;
    double* in_re = ctx.re; double* in_im = ctx.im;
    double* out_re = ctx.re2; double* out_im = ctx.im2;
    const double sign = inverse ? -1.0 : 1.0;
    const __m512d v_sign = _mm512_set1_pd(sign);
    const __m512d v_neg_sign = _mm512_set1_pd(-sign);

    for (size_t m = 1; m < n; m <<= 2) {
        size_t stride = n / (m << 2);

        if (stride >= 8) {
#pragma omp parallel for schedule(static) if(m >= 4)
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;

                __m512d v_w1r = _mm512_set1_pd(ctx.w1_re[tw]);
                __m512d v_w1i = _mm512_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);
                __m512d v_w2r = _mm512_set1_pd(ctx.w2_re[tw]);
                __m512d v_w2i = _mm512_set1_pd(inverse ? -ctx.w2_im[tw] : ctx.w2_im[tw]);
                __m512d v_w3r = _mm512_set1_pd(ctx.w3_re[tw]);
                __m512d v_w3i = _mm512_set1_pd(inverse ? -ctx.w3_im[tw] : ctx.w3_im[tw]);

                for (size_t k = 0; k < stride; k += 8) {
                    size_t i0 = k + j * (stride << 2);
                    size_t i1 = i0 + stride;
                    size_t i2 = i1 + stride;
                    size_t i3 = i2 + stride;

                    __m512d r0 = _mm512_load_pd(&in_re[i0]), i0v = _mm512_load_pd(&in_im[i0]);
                    __m512d r1 = _mm512_load_pd(&in_re[i1]), i1v = _mm512_load_pd(&in_im[i1]);
                    __m512d r2 = _mm512_load_pd(&in_re[i2]), i2v = _mm512_load_pd(&in_im[i2]);
                    __m512d r3 = _mm512_load_pd(&in_re[i3]), i3v = _mm512_load_pd(&in_im[i3]);

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

                    size_t o0 = k + j * stride;
                    size_t o1 = o0 + m * stride;
                    size_t o2 = o1 + m * stride;
                    size_t o3 = o2 + m * stride;

                    _mm512_store_pd(&out_re[o0], _mm512_add_pd(s0r, s2r));
                    _mm512_store_pd(&out_im[o0], _mm512_add_pd(s0i, s2i));
                    _mm512_store_pd(&out_re[o1], _mm512_add_pd(s1r, d3r));
                    _mm512_store_pd(&out_im[o1], _mm512_add_pd(s1i, d3i));
                    _mm512_store_pd(&out_re[o2], _mm512_sub_pd(s0r, s2r));
                    _mm512_store_pd(&out_im[o2], _mm512_sub_pd(s0i, s2i));
                    _mm512_store_pd(&out_re[o3], _mm512_sub_pd(s1r, d3r));
                    _mm512_store_pd(&out_im[o3], _mm512_sub_pd(s1i, d3i));
                }
            }
        }
        else {
            for (size_t j = 0; j < m; j++) {
                size_t tw = j * stride;
                double w1r = ctx.w1_re[tw], w1i = inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw];
                double w2r = ctx.w2_re[tw], w2i = inverse ? -ctx.w2_im[tw] : ctx.w2_im[tw];
                double w3r = ctx.w3_re[tw], w3i = inverse ? -ctx.w3_im[tw] : ctx.w3_im[tw];

                for (size_t k = 0; k < stride; ++k) {
                    size_t i0 = k + j * 4 * stride, i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;
                    double r0 = in_re[i0], i0v = in_im[i0], r1 = in_re[i1], i1v = in_im[i1], r2 = in_re[i2], i2v = in_im[i2], r3 = in_re[i3], i3v = in_im[i3];
                    double t1r = r1 * w1r - i1v * w1i, t1i = r1 * w1i + i1v * w1r;
                    double t2r = r2 * w2r - i2v * w2i, t2i = r2 * w2i + i2v * w2r;
                    double t3r = r3 * w3r - i3v * w3i, t3i = r3 * w3i + i3v * w3r;
                    double s0r = r0 + t2r, s0i = i0v + t2i, s1r = r0 - t2r, s1i = i0v - t2i, s2r = t1r + t3r, s2i = t1i + t3i, s3r = t1r - t3r, s3i = t1i - t3i;
                    double d3r = s3i * sign, d3i = -s3r * sign;
                    size_t o0 = k + j * stride, o1 = o0 + m * stride, o2 = o1 + m * stride, o3 = o2 + m * stride;
                    out_re[o0] = s0r + s2r; out_im[o0] = s0i + s2i; out_re[o1] = s1r + d3r; out_im[o1] = s1i + d3i; out_re[o2] = s0r - s2r; out_im[o2] = s0i - s2i; out_re[o3] = s1r - d3r; out_im[o3] = s1i - d3i;
                }
            }
        }
        std::swap(in_re, out_re); std::swap(in_im, out_im);
    }
    return in_re;
}

// ---------------------------------------------------------
// STOCKHAM MIXED-RADIX (8 -> 4 -> 2)
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

        // =========================================================
        // RADIX-8
        // =========================================================
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
                        size_t i0 = k + j * (stride << 3);
                        size_t i1 = i0 + stride; size_t i2 = i1 + stride; size_t i3 = i2 + stride;
                        size_t i4 = i3 + stride; size_t i5 = i4 + stride; size_t i6 = i5 + stride; size_t i7 = i6 + stride;

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
                                                
                        __m512d u5i_sign = _mm512_mul_pd(u5i, v_sign);
                        __m512d u5r_sign = _mm512_mul_pd(u5r, v_sign);
                        __m512d v5r = _mm512_mul_pd(v_C, _mm512_add_pd(u5r, u5i_sign));
                        __m512d v5i = _mm512_mul_pd(v_C, _mm512_sub_pd(u5i, u5r_sign));

                        __m512d v6r = _mm512_mul_pd(u6i, v_sign);
                        __m512d v6i = _mm512_mul_pd(u6r, v_neg_sign);

                        __m512d u7i_sign = _mm512_mul_pd(u7i, v_sign);
                        __m512d u7r_sign = _mm512_mul_pd(u7r, v_sign);
                        __m512d v7r = _mm512_mul_pd(v_C, _mm512_sub_pd(u7i_sign, u7r));
                        __m512d v7i = _mm512_mul_pd(v_neg_C, _mm512_add_pd(u7r_sign, u7i));

                        __m512d a0r = _mm512_add_pd(u0r, u2r), a0i = _mm512_add_pd(u0i, u2i); __m512d a1r = _mm512_sub_pd(u0r, u2r), a1i = _mm512_sub_pd(u0i, u2i);
                        __m512d a2r = _mm512_add_pd(u1r, u3r), a2i = _mm512_add_pd(u1i, u3i); __m512d a3r = _mm512_sub_pd(u1r, u3r), a3i = _mm512_sub_pd(u1i, u3i);
                        __m512d d3r = _mm512_mul_pd(a3i, v_sign), d3i = _mm512_mul_pd(a3r, v_neg_sign);

                        size_t o0 = k + j * stride;           size_t o1 = o0 + m * stride;
                        size_t o2 = o1 + m * stride;          size_t o3 = o2 + m * stride;
                        size_t o4 = o3 + m * stride;          size_t o5 = o4 + m * stride;
                        size_t o6 = o5 + m * stride;          size_t o7 = o6 + m * stride;

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
        // =========================================================
        // RADIX-4
        // =========================================================
        else if (remaining >= 4) {
            size_t stride = remaining >> 2;
            if (stride >= 8) {
#pragma omp parallel for schedule(static) if(m >= 4)
                for (size_t j = 0; j < m; j++) {
                    size_t tw = j * stride;
                    __m512d v_w1r = _mm512_set1_pd(ctx.w1_re[tw]);       __m512d v_w1i = _mm512_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);
                    __m512d v_w2r = _mm512_set1_pd(ctx.w1_re[2 * tw]);   __m512d v_w2i = _mm512_set1_pd(inverse ? -ctx.w1_im[2 * tw] : ctx.w1_im[2 * tw]);
                    __m512d v_w3r = _mm512_set1_pd(ctx.w1_re[3 * tw]);   __m512d v_w3i = _mm512_set1_pd(inverse ? -ctx.w1_im[3 * tw] : ctx.w1_im[3 * tw]);

                    for (size_t k = 0; k < stride; k += 8) {
                        size_t i0 = k + j * (stride << 2), i1 = i0 + stride, i2 = i1 + stride, i3 = i2 + stride;

                        __m512d r0 = _mm512_load_pd(&in_re[i0]), i0v = _mm512_load_pd(&in_im[i0]);
                        __m512d r1 = _mm512_load_pd(&in_re[i1]), i1v = _mm512_load_pd(&in_im[i1]);
                        __m512d r2 = _mm512_load_pd(&in_re[i2]), i2v = _mm512_load_pd(&in_im[i2]);
                        __m512d r3 = _mm512_load_pd(&in_re[i3]), i3v = _mm512_load_pd(&in_im[i3]);

                        __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i)); __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                        __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i)); __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r));
                        __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i)); __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r));

                        __m512d s0r = _mm512_add_pd(r0, t2r), s0i = _mm512_add_pd(i0v, t2i);
                        __m512d s1r = _mm512_sub_pd(r0, t2r), s1i = _mm512_sub_pd(i0v, t2i);
                        __m512d s2r = _mm512_add_pd(t1r, t3r), s2i = _mm512_add_pd(t1i, t3i);
                        __m512d s3r = _mm512_sub_pd(t1r, t3r), s3i = _mm512_sub_pd(t1i, t3i);

                        __m512d d3r = _mm512_mul_pd(s3i, v_sign);
                        __m512d d3i = _mm512_mul_pd(s3r, v_neg_sign);

                        size_t o0 = k + j * stride, o1 = o0 + m * stride, o2 = o1 + m * stride, o3 = o2 + m * stride;

                        _mm512_store_pd(&out_re[o0], _mm512_add_pd(s0r, s2r)); _mm512_store_pd(&out_im[o0], _mm512_add_pd(s0i, s2i));
                        _mm512_store_pd(&out_re[o1], _mm512_add_pd(s1r, d3r)); _mm512_store_pd(&out_im[o1], _mm512_add_pd(s1i, d3i));
                        _mm512_store_pd(&out_re[o2], _mm512_sub_pd(s0r, s2r)); _mm512_store_pd(&out_im[o2], _mm512_sub_pd(s0i, s2i));
                        _mm512_store_pd(&out_re[o3], _mm512_sub_pd(s1r, d3r)); _mm512_store_pd(&out_im[o3], _mm512_sub_pd(s1i, d3i));
                    }
                }
            }
            else {
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
            }
            m <<= 2;
        }
        // =========================================================
        // RADIX-2
        // =========================================================
        else if (remaining >= 2) {
            size_t stride = remaining >> 1;
            if (stride >= 8) {
                for (size_t j = 0; j < m; j++) {
                    size_t tw = j * stride;
                    __m512d v_w1r = _mm512_set1_pd(ctx.w1_re[tw]);
                    __m512d v_w1i = _mm512_set1_pd(inverse ? -ctx.w1_im[tw] : ctx.w1_im[tw]);

                    for (size_t k = 0; k < stride; k += 8) {
                        size_t i0 = k + j * (stride << 1), i1 = i0 + stride;
                        __m512d r0 = _mm512_load_pd(&in_re[i0]), i0v = _mm512_load_pd(&in_im[i0]);
                        __m512d r1 = _mm512_load_pd(&in_re[i1]), i1v = _mm512_load_pd(&in_im[i1]);
                        __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i));
                        __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r));
                        size_t o0 = k + j * stride, o1 = o0 + m * stride;
                        _mm512_store_pd(&out_re[o0], _mm512_add_pd(r0, t1r)); _mm512_store_pd(&out_im[o0], _mm512_add_pd(i0v, t1i));
                        _mm512_store_pd(&out_re[o1], _mm512_sub_pd(r0, t1r)); _mm512_store_pd(&out_im[o1], _mm512_sub_pd(i0v, t1i));
                    }
                }
            }
            else {
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
            }
            m <<= 1;
        }

        std::swap(in_re, out_re);
        std::swap(in_im, out_im);
    }
    return in_re;
}

// ---------------------------------------------------------
// CACHE-FRIENDLY
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

                for (size_t k = 0; k < stride; k += 8) {
                    size_t i0 = k + j * (stride << 2);
                    size_t i1 = i0 + stride; size_t i2 = i1 + stride; size_t i3 = i2 + stride;

                    size_t prefetch_offset = 32;
                    _mm_prefetch((const char*)&cur_in_re[i0 + prefetch_offset], _MM_HINT_T1);
                    _mm_prefetch((const char*)&cur_in_im[i0 + prefetch_offset], _MM_HINT_T1);
                    _mm_prefetch((const char*)&cur_in_re[i1 + prefetch_offset], _MM_HINT_T1);
                    _mm_prefetch((const char*)&cur_in_im[i1 + prefetch_offset], _MM_HINT_T1);
                    _mm_prefetch((const char*)&cur_in_re[i2 + prefetch_offset], _MM_HINT_T1);
                    _mm_prefetch((const char*)&cur_in_im[i2 + prefetch_offset], _MM_HINT_T1);
                    _mm_prefetch((const char*)&cur_in_re[i3 + prefetch_offset], _MM_HINT_T1);
                    _mm_prefetch((const char*)&cur_in_im[i3 + prefetch_offset], _MM_HINT_T1);

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

#define BUTTERFLY_MATH \
                __m512d t1r = _mm512_fmsub_pd(r1, v_w1r, _mm512_mul_pd(i1v, v_w1i)); \
                __m512d t1i = _mm512_fmadd_pd(r1, v_w1i, _mm512_mul_pd(i1v, v_w1r)); \
                __m512d t2r = _mm512_fmsub_pd(r2, v_w2r, _mm512_mul_pd(i2v, v_w2i)); \
                __m512d t2i = _mm512_fmadd_pd(r2, v_w2i, _mm512_mul_pd(i2v, v_w2r)); \
                __m512d t3r = _mm512_fmsub_pd(r3, v_w3r, _mm512_mul_pd(i3v, v_w3i)); \
                __m512d t3i = _mm512_fmadd_pd(r3, v_w3i, _mm512_mul_pd(i3v, v_w3r)); \
                __m512d s0r = _mm512_add_pd(r0, t2r), s0i = _mm512_add_pd(i0v, t2i); \
                __m512d s1r = _mm512_sub_pd(r0, t2r), s1i = _mm512_sub_pd(i0v, t2i); \
                __m512d s2r = _mm512_add_pd(t1r, t3r), s2i = _mm512_add_pd(t1i, t3i); \
                __m512d s3r = _mm512_sub_pd(t1r, t3r), s3i = _mm512_sub_pd(t1i, t3i); \
                __m512d d3r = _mm512_mul_pd(s3i, v_sign), d3i = _mm512_mul_pd(s3r, v_neg_sign);

#define STORE_MATH(out_r, out_i, o0, o1, o2, o3) \
                _mm512_storeu_pd(&(out_r)[o0], _mm512_add_pd(s0r, s2r)); _mm512_storeu_pd(&(out_i)[o0], _mm512_add_pd(s0i, s2i)); \
                _mm512_storeu_pd(&(out_r)[o1], _mm512_add_pd(s1r, d3r)); _mm512_storeu_pd(&(out_i)[o1], _mm512_add_pd(s1i, d3i)); \
                _mm512_storeu_pd(&(out_r)[o2], _mm512_sub_pd(s0r, s2r)); _mm512_storeu_pd(&(out_i)[o2], _mm512_sub_pd(s0i, s2i)); \
                _mm512_storeu_pd(&(out_r)[o3], _mm512_sub_pd(s1r, d3r)); _mm512_storeu_pd(&(out_i)[o3], _mm512_sub_pd(s1i, d3i));

                BUTTERFLY_MATH
                size_t o0 = j * 4; size_t o1 = o0 + m * 4; size_t o2 = o1 + m * 4; size_t o3 = o2 + m * 4;
                STORE_MATH(cur_out_re, cur_out_im, o0, o1, o2, o3)
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

                BUTTERFLY_MATH
                size_t o0 = j * 2; size_t o1 = o0 + m * 2; size_t o2 = o1 + m * 2; size_t o3 = o2 + m * 2;
                STORE_MATH(cur_out_re, cur_out_im, o0, o1, o2, o3)
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

                BUTTERFLY_MATH
                size_t o0 = j; size_t o1 = o0 + m; size_t o2 = o1 + m; size_t o3 = o2 + m;
                STORE_MATH(cur_out_re, cur_out_im, o0, o1, o2, o3)
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

void fft_bailey_2d(FFTContext& ctx, bool inverse) {
    size_t N = ctx.fft_len;
    size_t n1 = 1, n2 = 1, temp = N;
    while (temp > 1) {
        if (n1 <= n2) n1 *= 4;
        else n2 *= 4;
        temp /= 4;
    }

    if (!inverse) {
        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        size_t tw_stride_col = n2;
        std::vector<double> r_w1r(n1), r_w1i(n1), r_w2r(n1), r_w2i(n1), r_w3r(n1), r_w3i(n1);
#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n1; ++i) {
            size_t idx = i * tw_stride_col;
            r_w1r[i] = ctx.w1_re[idx]; r_w1i[i] = ctx.w1_im[idx];
            r_w2r[i] = ctx.w2_re[idx]; r_w2i[i] = ctx.w2_im[idx];
            r_w3r[i] = ctx.w3_re[idx]; r_w3i[i] = ctx.w3_im[idx];
        }
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; i++) {
            double* r_in_re = &ctx.re[i * n1]; double* r_in_im = &ctx.im[i * n1];
            double* r_out_re = &ctx.re2[i * n1]; double* r_out_im = &ctx.im2[i * n1];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n1,
                                                      r_w1r, r_w1i, r_w2r, r_w2i, r_w3r, r_w3i, false);
            if (res != r_in_re) {
                std::memcpy(r_in_re, r_out_re, n1 * sizeof(double));
                std::memcpy(r_in_im, r_out_im, n1 * sizeof(double));
            }
        }

        transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1, ctx, false);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        size_t tw_stride_row = n1;
        std::vector<double> c_w1r(n2), c_w1i(n2), c_w2r(n2), c_w2i(n2), c_w3r(n2), c_w3i(n2);
#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n2; ++i) {
            size_t idx = i * tw_stride_row;
            c_w1r[i] = ctx.w1_re[idx]; c_w1i[i] = ctx.w1_im[idx];
            c_w2r[i] = ctx.w2_re[idx]; c_w2i[i] = ctx.w2_im[idx];
            c_w3r[i] = ctx.w3_re[idx]; c_w3i[i] = ctx.w3_im[idx];
        }
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; i++) {
            double* r_in_re = &ctx.re[i * n2]; double* r_in_im = &ctx.im[i * n2];
            double* r_out_re = &ctx.re2[i * n2]; double* r_out_im = &ctx.im2[i * n2];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n2,
                                                      c_w1r, c_w1i, c_w2r, c_w2i, c_w3r, c_w3i, false);
            if (res != r_in_re) {
                std::memcpy(r_in_re, r_out_re, n2 * sizeof(double));
                std::memcpy(r_in_im, r_out_im, n2 * sizeof(double));
            }
        }

        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

    } else {
        transpose_blocked_complex(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        size_t tw_stride_row = n1;
        std::vector<double> c_w1r(n2), c_w1i(n2), c_w2r(n2), c_w2i(n2), c_w3r(n2), c_w3i(n2);
#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n2; ++i) {
            size_t idx = i * tw_stride_row;
            c_w1r[i] = ctx.w1_re[idx]; c_w1i[i] = ctx.w1_im[idx];
            c_w2r[i] = ctx.w2_re[idx]; c_w2i[i] = ctx.w2_im[idx];
            c_w3r[i] = ctx.w3_re[idx]; c_w3i[i] = ctx.w3_im[idx];
        }
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; i++) {
            double* r_in_re = &ctx.re[i * n2]; double* r_in_im = &ctx.im[i * n2];
            double* r_out_re = &ctx.re2[i * n2]; double* r_out_im = &ctx.im2[i * n2];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n2,
                                                      c_w1r, c_w1i, c_w2r, c_w2i, c_w3r, c_w3i, true);
            if (res != r_in_re) {
                std::memcpy(r_in_re, r_out_re, n2 * sizeof(double));
                std::memcpy(r_in_im, r_out_im, n2 * sizeof(double));
            }
        }

        transpose_and_twiddle(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, ctx, true);
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);

        size_t tw_stride_col = n2;
        std::vector<double> r_w1r(n1), r_w1i(n1), r_w2r(n1), r_w2i(n1), r_w3r(n1), r_w3i(n1);
#pragma omp parallel for schedule(static)
        for(size_t i = 0; i < n1; ++i) {
            size_t idx = i * tw_stride_col;
            r_w1r[i] = ctx.w1_re[idx]; r_w1i[i] = ctx.w1_im[idx];
            r_w2r[i] = ctx.w2_re[idx]; r_w2i[i] = ctx.w2_im[idx];
            r_w3r[i] = ctx.w3_re[idx]; r_w3i[i] = ctx.w3_im[idx];
        }

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; i++) {
            double* r_in_re = &ctx.re[i * n1]; double* r_in_im = &ctx.im[i * n1];
            double* r_out_re = &ctx.re2[i * n1]; double* r_out_im = &ctx.im2[i * n1];
            double* res = fft_stockham_radix4_blocked(r_in_re, r_in_im, r_out_re, r_out_im, n1,
                                                      r_w1r, r_w1i, r_w2r, r_w2i, r_w3r, r_w3i, true);
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
// KARATSUBA UNPACK: Overlap-Add + EXACT Modulo Proth
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

    // =========================================================
    // OPTYMALIZACJA: Sprzętowe dzielenie (Bezpieczny rejestr)
    // =========================================================
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

    // =========================================================
    // OPTYMALIZACJA 2: Sprzętowe odejmowanie (Intrinsics SBB)
    // =========================================================
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

void fft_square_karatsuba(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    auto t_pack_0 = clock::now();
    pack_karatsuba_limbs(v, n_limbs, ctx);
    auto t_pack_1 = clock::now();

    auto run_fft_on_buffer = [&](double*& target_re, double*& target_im, bool inverse) {
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);

        unsigned p = 0;
        size_t temp = ctx.fft_len;
        while (temp > 1) { temp >>= 1; p++; }        
        if (p % 2 == 0) {
            fft_bailey_2d(ctx, inverse);
        }
        else {            
            double* res_re = fft_stockham_radix8(ctx, inverse);
            if (res_re != ctx.re) {
                std::swap(ctx.re, ctx.re2);
                std::swap(ctx.im, ctx.im2);
            }
        }

        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);
        };

    auto t_fwd_0 = clock::now();
    run_fft_on_buffer(ctx.k_A_re, ctx.k_A_im, false);
    run_fft_on_buffer(ctx.k_B_re, ctx.k_B_im, false);
    auto t_fwd_1 = clock::now();

    auto t_sq_0 = clock::now();

    double inv_n = 1.0 / (double)ctx.fft_len;
    __m512d v_inv_n = _mm512_set1_pd(inv_n);
    __m512d v_two = _mm512_set1_pd(2.0);

    const size_t PREFETCH_DIST = 32;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ctx.fft_len; i += 8) {
        _mm_prefetch((const char*)&ctx.k_A_re[i + PREFETCH_DIST], _MM_HINT_T0);
        _mm_prefetch((const char*)&ctx.k_A_im[i + PREFETCH_DIST], _MM_HINT_T0);
        _mm_prefetch((const char*)&ctx.k_B_re[i + PREFETCH_DIST], _MM_HINT_T0);
        _mm_prefetch((const char*)&ctx.k_B_im[i + PREFETCH_DIST], _MM_HINT_T0);

        __m512d a_re = _mm512_load_pd(&ctx.k_A_re[i]);
        __m512d a_im = _mm512_load_pd(&ctx.k_A_im[i]);
        __m512d b_re = _mm512_load_pd(&ctx.k_B_re[i]);
        __m512d b_im = _mm512_load_pd(&ctx.k_B_im[i]);

        __m512d a2_re = _mm512_fmsub_pd(a_re, a_re, _mm512_mul_pd(a_im, a_im));
        __m512d a2_im = _mm512_mul_pd(v_two, _mm512_mul_pd(a_re, a_im));
        _mm512_store_pd(&ctx.k_A_re[i], _mm512_mul_pd(a2_re, v_inv_n));
        _mm512_store_pd(&ctx.k_A_im[i], _mm512_mul_pd(a2_im, v_inv_n));

        __m512d b2_re = _mm512_fmsub_pd(b_re, b_re, _mm512_mul_pd(b_im, b_im));
        __m512d b2_im = _mm512_mul_pd(v_two, _mm512_mul_pd(b_re, b_im));
        _mm512_store_pd(&ctx.k_B_re[i], _mm512_mul_pd(b2_re, v_inv_n));
        _mm512_store_pd(&ctx.k_B_im[i], _mm512_mul_pd(b2_im, v_inv_n));

        __m512d ab_re = _mm512_fmsub_pd(a_re, b_re, _mm512_mul_pd(a_im, b_im));
        __m512d ab_im = _mm512_fmadd_pd(a_re, b_im, _mm512_mul_pd(a_im, b_re));
        _mm512_store_pd(&ctx.k_AB_re[i], _mm512_mul_pd(_mm512_mul_pd(ab_re, v_two), v_inv_n));
        _mm512_store_pd(&ctx.k_AB_im[i], _mm512_mul_pd(_mm512_mul_pd(ab_im, v_two), v_inv_n));
    }
    _mm_sfence();

    auto t_sq_1 = clock::now();

    auto t_inv_0 = clock::now();
    run_fft_on_buffer(ctx.k_A_re, ctx.k_A_im, true);
    run_fft_on_buffer(ctx.k_B_re, ctx.k_B_im, true);
    run_fft_on_buffer(ctx.k_AB_re, ctx.k_AB_im, true);
    auto t_inv_1 = clock::now();

    unpack_karatsuba_overlap_add(v, n_limbs, ctx, n_bits, k_val);

    ctx.prof_pack += secd(t_pack_1 - t_pack_0).count();
    ctx.prof_fwd += secd(t_fwd_1 - t_fwd_0).count();
    ctx.prof_square += secd(t_sq_1 - t_sq_0).count();
    ctx.prof_inv += secd(t_inv_1 - t_inv_0).count();
    ctx.prof_calls++;
}

