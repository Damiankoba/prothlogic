// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "fft_engine.h"
#include "checkpoint_manager.h"
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
#include <mutex>
#include <memory>
#include <functional>
#pragma GCC target("avx512f,avx512dq,fma")
#include <immintrin.h>
#include <x86intrin.h>
#include <sys/mman.h>
#include <omp.h>
#include <vector>
#include <gmp.h>


#ifdef LOTR_PROFILE
// Timing of one pass inside the FFT (RAII): adds the object's lifetime to ctx.prof_pass[slot].
struct PassTimer {
    double& acc;
    std::chrono::high_resolution_clock::time_point t0;
    explicit PassTimer(double& a) : acc(a), t0(std::chrono::high_resolution_clock::now()) {}
    ~PassTimer() { acc += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count(); }
};
#define LOTR_PASS(ctx, slot) PassTimer lotr_pass_timer_((ctx).prof_pass[(slot)])

// Parts of a pass (ctx.prof_sub): time stamp counter, a few ns per read. Seconds per tick measured once.
static double tsc_sec_per_tick() {
    static const double v = []() {
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t c0 = __rdtsc();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(20)) {}
        const uint64_t c1 = __rdtsc();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s / (double)(c1 - c0);
    }();
    return v;
}
#define SUB_T(v) const uint64_t v = __rdtsc()
#define SUB_ADD(acc, a, b) ((acc) += (b) - (a))
#else
#define LOTR_PASS(ctx, slot) do {} while (0)
#define SUB_T(v) do {} while (0)
#define SUB_ADD(acc, a, b) do {} while (0)
#endif

// Accumulators of one thread for the pass parts; merged into ctx.prof_sub[base ..] (avg over threads + busiest thread).
struct SubAcc {
    uint64_t c[5] = { 0, 0, 0, 0, 0 };
};
#ifdef LOTR_PROFILE
static void sub_merge(FFTContext& ctx, int base, int nparts, const SubAcc* per_thread, int nt) {
    if (base < 0) return;
    const double k = tsc_sec_per_tick();
    uint64_t busy_max = 0;
    for (int t = 0; t < nt; ++t) {
        uint64_t busy = 0;
        for (int i = 0; i < nparts; ++i) {
            ctx.prof_sub[base + i] += (double)per_thread[t].c[i] * k / (double)nt;
            busy += per_thread[t].c[i];
        }
        busy_max = std::max(busy_max, busy);
    }
    ctx.prof_sub[base + nparts] += (double)busy_max * k;
}
#endif

// w^e (e < N) from the two-level table ctx.tw2_* (built in bailey_setup); directly from w1 while it is not built.
// The large w1 table (2 x 8N bytes) does not fit in L2 at 2^18+, and the r*col indices jump around it - every read is a miss.
struct Tw2 {
    const double* lr; const double* li; const double* hr; const double* hi;
    const double* dr; const double* di;
    size_t mask; unsigned sh; bool direct;
    explicit Tw2(const FFTContext& ctx)
        : lr(ctx.tw2_lo_re.data()), li(ctx.tw2_lo_im.data()), hr(ctx.tw2_hi_re.data()), hi(ctx.tw2_hi_im.data()),
          dr(ctx.w1_re), di(ctx.w1_im), mask((size_t(1) << ctx.tw2_shift) - 1), sh(ctx.tw2_shift),
          direct(ctx.tw2_N != ctx.fft_len) {}
    inline void get(size_t e, double& wr, double& wi) const {
        if (direct) { wr = dr[e]; wi = di[e]; return; }
        const size_t l = e & mask, h = e >> sh;
        wr = hr[h] * lr[l] - hi[h] * li[l];
        wi = hr[h] * li[l] + hi[h] * lr[l];
    }
};

// ENGINE CHOICE by FFT length (measured, 6 threads of one CCD, ms/iter: two-pass Bailey / old Bailey / Stockham):
//   2^15: 0.11/0.13/0.14   2^17: 0.38/0.42/0.46   2^19: 1.87/1.65/1.14   2^21: 21.6/37.4/46.1
// Even powers: always Bailey. Odd: Bailey for 2^15, 2^17 and >= 2^21, Stockham 1D for 2^19 and <= 2^13.
static bool use_bailey_engine(unsigned p) {
    return p % 2 == 0 || p == 15 || p == 17 || p >= 21;
}
bool fft_init(FFTContext& ctx, FFTBackend backend, size_t fft_len) {
    ctx.backend = backend;
    ctx.fft_len = fft_len;
    ctx.factor_plan = generate_fft_plan(fft_len);
    ctx.last_max_diff = 0.0;

    
    // fft_len/16 of headroom for the padded-row layout (ctx.lay_pad: up to 8 doubles per row, rows >= 128)
    size_t table_size = (fft_len + fft_len / 16 + 64) * sizeof(double);
    size_t total_size = table_size * 16;
    void* big_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (big_mem == MAP_FAILED) {
        big_mem = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (big_mem == MAP_FAILED) return false;
        // No HUGETLB pool: ask for transparent huge pages (THP), fewer TLB misses on large-stride accesses.
        madvise(big_mem, total_size, MADV_HUGEPAGE);
    }
    ctx.raw_allocation = big_mem;
    ctx.base_ptr = big_mem;
    ctx.total_allocation_size = total_size;

    // No memset: anonymous mmap pages are already zero, and a memset would make all the tables resident - also the ones the
    // chosen square never touches (giant 195*2^1073000000+1: ~9 GB resident; N = 2.1e9 killed by OOM).
    char* base = (char*)big_mem;

    ctx.re = (double*)(base + 0 * table_size);
    ctx.im = (double*)(base + 1 * table_size);
    ctx.re2 = (double*)(base + 2 * table_size);
    ctx.im2 = (double*)(base + 3 * table_size);
    ctx.w1_re = (double*)(base + 4 * table_size);
    ctx.w1_im = (double*)(base + 5 * table_size);
    ctx.k_A_re = (double*)(base + 6 * table_size);
    ctx.k_A_im = (double*)(base + 7 * table_size);
    ctx.k_B_re = (double*)(base + 8 * table_size);
    ctx.k_B_im = (double*)(base + 9 * table_size);
    ctx.k_AB_re = (double*)(base + 10 * table_size);
    ctx.k_AB_im = (double*)(base + 11 * table_size);

    ctx.g_C_re = (double*)(base + 12 * table_size);
    ctx.g_C_im = (double*)(base + 13 * table_size);
    ctx.g_D_re = (double*)(base + 14 * table_size);
    ctx.g_D_im = (double*)(base + 15 * table_size);

    size_t i = 0;
    for (; i + 7 < fft_len; i += 8) {
        alignas(64) double temp_w1_re[8], temp_w1_im[8];

        for (int k = 0; k < 8; ++k) {
            double i_d = static_cast<double>(i + k);
            double len_d = static_cast<double>(fft_len);

            double ang = -2.0 * M_PI * i_d / len_d;
            temp_w1_re[k] = std::cos(ang);       temp_w1_im[k] = std::sin(ang);
        }

        _mm512_stream_pd(&ctx.w1_re[i], _mm512_load_pd(temp_w1_re));
        _mm512_stream_pd(&ctx.w1_im[i], _mm512_load_pd(temp_w1_im));
    }

    _mm_sfence();
    
    for (; i < fft_len; ++i) {
        double i_d = static_cast<double>(i);
        double len_d = static_cast<double>(fft_len);
        double ang = -2.0 * M_PI * i_d / len_d;
        ctx.w1_re[i] = std::cos(ang);       ctx.w1_im[i] = std::sin(ang);
    }

    return true;
}



void fft_destroy(FFTContext& ctx) {
    if (ctx.raw_allocation && ctx.raw_allocation != MAP_FAILED) {
        munmap(ctx.raw_allocation, ctx.total_allocation_size);
        ctx.raw_allocation = nullptr;
    }
}


static std::string fft_engine_name_base(const FFTContext& ctx);
static bool is_len_3x2p(size_t len);
static bool is_len_7x2p(size_t len);

// Engine name for the log; wide digits are appended after the FFT length (bench.sh cuts the name at "[FFT_Len").
std::string get_fft_engine_name(const FFTContext& ctx) {
    std::string s = fft_engine_name_base(ctx);
    if (ctx.ibdwt) s += " [IBDWT]";
    else if (ctx.digit_bits != 16) s += " [" + std::to_string(ctx.digit_bits) + "-bit digits]";
    return s;
}

static std::string fft_engine_name_base(const FFTContext& ctx) {
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
    else if (is_len_3x2p(ctx.fft_len)) {
        unsigned p = 0;
        size_t temp = ctx.fft_len / 3;
        while (temp > 1) { temp >>= 1; p++; }
        return "Unified Bailey 2D (Radix-8/4/3/2) [FFT_Len = 3*2^" + std::to_string(p) + "]";
    }
    else if (is_len_7x2p(ctx.fft_len)) {
        unsigned p = 0;
        size_t temp = ctx.fft_len / 7;
        while (temp > 1) { temp >>= 1; p++; }
        return "Unified Bailey 2D (Radix-8/7/4/2) [FFT_Len = 7*2^" + std::to_string(p) + "]";
    }
    return "FFT_Len = " + std::to_string(ctx.fft_len) + " (no engine)";
}

// Only powers of two come to the plain FFT; 3*2^p / 7*2^p go through the fused square.
static void fft_run_inplace(FFTContext& ctx, bool inverse);

// ---------------------------------------------------------
// MULTI-THREADED STOCKHAM 1D (DYNAMIC THREADING)
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

// Radix-8 butterfly core of the batch-8 kernels (same result as st_radix8_vec), direction known at compile time:
// the multiplications by sg = +-1 become additions/subtractions, and the twiddle-free form (stage with j = 0, stage 1
// of the two-level FFT) skips the 7 complex multiplications by 1. Input t (already twiddled), output y.
template <bool INV>
static inline __attribute__((always_inline)) void r8_core(const __m512d* tr, const __m512d* ti, __m512d* yr, __m512d* yi,
                                                         __m512d v_C) {
    const __m512d u0r = _mm512_add_pd(tr[0], tr[4]), u0i = _mm512_add_pd(ti[0], ti[4]);
    const __m512d u4r = _mm512_sub_pd(tr[0], tr[4]), u4i = _mm512_sub_pd(ti[0], ti[4]);
    const __m512d u1r = _mm512_add_pd(tr[1], tr[5]), u1i = _mm512_add_pd(ti[1], ti[5]);
    const __m512d u5r = _mm512_sub_pd(tr[1], tr[5]), u5i = _mm512_sub_pd(ti[1], ti[5]);
    const __m512d u2r = _mm512_add_pd(tr[2], tr[6]), u2i = _mm512_add_pd(ti[2], ti[6]);
    const __m512d u6r = _mm512_sub_pd(tr[2], tr[6]), u6i = _mm512_sub_pd(ti[2], ti[6]);
    const __m512d u3r = _mm512_add_pd(tr[3], tr[7]), u3i = _mm512_add_pd(ti[3], ti[7]);
    const __m512d u7r = _mm512_sub_pd(tr[3], tr[7]), u7i = _mm512_sub_pd(ti[3], ti[7]);

    // even part: a = u0 +- u2, u1 +- u3; y0, y4 = a0 +- a2; y2, y6 = a1 +- d3, d3 = (sg a3i, -sg a3r)
    const __m512d a0r = _mm512_add_pd(u0r, u2r), a0i = _mm512_add_pd(u0i, u2i);
    const __m512d a1r = _mm512_sub_pd(u0r, u2r), a1i = _mm512_sub_pd(u0i, u2i);
    const __m512d a2r = _mm512_add_pd(u1r, u3r), a2i = _mm512_add_pd(u1i, u3i);
    const __m512d a3r = _mm512_sub_pd(u1r, u3r), a3i = _mm512_sub_pd(u1i, u3i);
    yr[0] = _mm512_add_pd(a0r, a2r); yi[0] = _mm512_add_pd(a0i, a2i);
    yr[4] = _mm512_sub_pd(a0r, a2r); yi[4] = _mm512_sub_pd(a0i, a2i);
    if constexpr (!INV) {
        yr[2] = _mm512_add_pd(a1r, a3i); yi[2] = _mm512_sub_pd(a1i, a3r);
        yr[6] = _mm512_sub_pd(a1r, a3i); yi[6] = _mm512_add_pd(a1i, a3r);
    }
    else {
        yr[2] = _mm512_sub_pd(a1r, a3i); yi[2] = _mm512_add_pd(a1i, a3r);
        yr[6] = _mm512_add_pd(a1r, a3i); yi[6] = _mm512_sub_pd(a1i, a3r);
    }

    // odd part: b0, b1 = u4 +- v6, v6 = (sg u6i, -sg u6r); b2, b3 = v5 +- v7 with
    // v5 = C (u5r + sg u5i, u5i - sg u5r), v7 = (C (sg u7i - u7r), -C (sg u7r + u7i))
    __m512d b0r, b0i, b1r, b1i, p5r, p5i, p7r, p7i;
    if constexpr (!INV) {
        b0r = _mm512_add_pd(u4r, u6i); b0i = _mm512_sub_pd(u4i, u6r);
        b1r = _mm512_sub_pd(u4r, u6i); b1i = _mm512_add_pd(u4i, u6r);
        p5r = _mm512_add_pd(u5r, u5i); p5i = _mm512_sub_pd(u5i, u5r);
        p7r = _mm512_sub_pd(u7i, u7r); p7i = _mm512_add_pd(u7r, u7i);   // v7 = C (p7r, -p7i)
    }
    else {
        b0r = _mm512_sub_pd(u4r, u6i); b0i = _mm512_add_pd(u4i, u6r);
        b1r = _mm512_add_pd(u4r, u6i); b1i = _mm512_sub_pd(u4i, u6r);
        p5r = _mm512_sub_pd(u5r, u5i); p5i = _mm512_add_pd(u5i, u5r);
        p7r = _mm512_add_pd(u7i, u7r); p7i = _mm512_sub_pd(u7i, u7r);   // v7 = C (-p7r, -p7i)
    }
    __m512d b2r, b2i, b3r, b3i;
    if constexpr (!INV) {
        b2r = _mm512_mul_pd(v_C, _mm512_add_pd(p5r, p7r)); b2i = _mm512_mul_pd(v_C, _mm512_sub_pd(p5i, p7i));
        b3r = _mm512_mul_pd(v_C, _mm512_sub_pd(p5r, p7r)); b3i = _mm512_mul_pd(v_C, _mm512_add_pd(p5i, p7i));
    }
    else {
        b2r = _mm512_mul_pd(v_C, _mm512_sub_pd(p5r, p7r)); b2i = _mm512_mul_pd(v_C, _mm512_sub_pd(p5i, p7i));
        b3r = _mm512_mul_pd(v_C, _mm512_add_pd(p5r, p7r)); b3i = _mm512_mul_pd(v_C, _mm512_add_pd(p5i, p7i));
    }
    yr[1] = _mm512_add_pd(b0r, b2r); yi[1] = _mm512_add_pd(b0i, b2i);
    yr[5] = _mm512_sub_pd(b0r, b2r); yi[5] = _mm512_sub_pd(b0i, b2i);
    // y3, y7 = b1 +- e3, e3 = (sg b3i, -sg b3r)
    if constexpr (!INV) {
        yr[3] = _mm512_add_pd(b1r, b3i); yi[3] = _mm512_sub_pd(b1i, b3r);
        yr[7] = _mm512_sub_pd(b1r, b3i); yi[7] = _mm512_add_pd(b1i, b3r);
    }
    else {
        yr[3] = _mm512_sub_pd(b1r, b3i); yi[3] = _mm512_add_pd(b1i, b3r);
        yr[7] = _mm512_add_pd(b1r, b3i); yi[7] = _mm512_sub_pd(b1i, b3r);
    }
}

// Twiddled form: t_q = x_q * w_q (q = 1..7; wi already carries the sign of the direction), then the core.
template <bool INV>
static inline __attribute__((always_inline)) void r8_tw(const __m512d* r, const __m512d* i, const __m512d* wr, const __m512d* wi,
                                                       __m512d* yr, __m512d* yi, __m512d v_C) {
    __m512d tr[8], ti[8];
    tr[0] = r[0]; ti[0] = i[0];
#pragma GCC unroll 7
    for (int q = 1; q < 8; ++q) {
        tr[q] = _mm512_fmsub_pd(r[q], wr[q], _mm512_mul_pd(i[q], wi[q]));
        ti[q] = _mm512_fmadd_pd(r[q], wi[q], _mm512_mul_pd(i[q], wr[q]));
    }
    r8_core<INV>(tr, ti, yr, yi, v_C);
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

        if (remaining == 16 && m >= 8) {
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
// TWO-PASS BAILEY: the column passes are batch-8 (bailey_colfft_pass8), transposes built in
// ---------------------------------------------------------
// =====================================================================
// LENGTH-R FFT FOR 8 COLUMNS AT ONCE ("batch-8"): vector r (8 doubles at offset 8*r) = element r of the eight columns.
// Stockham radix-8 (radix-4 or radix-2 tail), the index formulas of a Stockham FFT:
//   input  k + j*r*stride + q*stride,  twiddle W_R^(q*j*stride),  output k + j*stride + p*m*stride.
// Twiddles are shared by the 8 columns (broadcast), so no stage needs in-register shuffles.
// w_re/w_im: table of R-th roots of unity (length R, forward direction). Returns true when the result is in b.
// =====================================================================
// c_re/c_im != nullptr (a third buffer, at least 2 stages): stage outputs go b, c, b, ... and the LAST stage writes back
// into a, so the result is always in a (in-place two-level form with an odd number of sub-FFT stages, b8_ip_subs).
// m0 > 1: the stages below m0 are already done (their output is in a; b8_sub_dit8 runs the first radix-8 stage itself,
// reading a strided input) - the loop starts at m = m0.
template <bool INV>
static bool fft_batch8_t(double* __restrict a_re, double* __restrict a_im, double* __restrict b_re, double* __restrict b_im,
                         size_t R, const double* w_re0, const double* w_im0, size_t wstride,
                         double* c_re = nullptr, double* c_im = nullptr, size_t m0 = 1) {
    // wstride > 1: root table of a larger order (R*wstride), we use every wstride-th element
    struct Strided {
        const double* p; size_t s;
        double operator[](size_t i) const { return p[i * s]; }
    };
    const Strided w_re{ w_re0, wstride }, w_im{ w_im0, wstride };
    constexpr double sg = INV ? -1.0 : 1.0;
    const __m512d v_sign = _mm512_set1_pd(sg);
    const __m512d v_neg_sign = _mm512_set1_pd(-sg);
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);

    double* ir = a_re; double* ii = a_im;
    double* orr = b_re; double* oi = b_im;
    bool in_b = false;
    size_t m = m0;
    // Radix of every stage: 8 while the rest is divisible by 8, then 3 (lengths 3*2^k), then a radix-4 or radix-2 tail.
    // For powers of two this is exactly the old sequence (radix-8 ..., then 4 or 2).
    while (m < R) {
        const size_t rem = R / m;
        if (rem % 8 == 0) {
            const size_t stride = rem >> 3;
            const size_t os = m * stride;
            for (size_t j = 0; j < m; ++j) {
                const size_t tw = j * stride;
                const size_t ib = j * 8 * stride;
                const size_t ob = j * stride;
                if (j == 0) {
                    // all twiddles are 1 (the whole first stage, m = 1): the core alone
                    for (size_t k = 0; k < stride; ++k) {
                        __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
                        for (int q = 0; q < 8; ++q) {
                            r[q] = _mm512_load_pd(ir + 8 * (ib + k + q * stride));
                            i[q] = _mm512_load_pd(ii + 8 * (ib + k + q * stride));
                        }
                        r8_core<INV>(r, i, yr, yi, v_C);
#pragma GCC unroll 8
                        for (int p = 0; p < 8; ++p) {
                            _mm512_store_pd(orr + 8 * (ob + k + p * os), yr[p]);
                            _mm512_store_pd(oi + 8 * (ob + k + p * os), yi[p]);
                        }
                    }
                    continue;
                }
                __m512d wr[8], wi[8];
                wr[0] = _mm512_set1_pd(1.0); wi[0] = _mm512_setzero_pd();
#pragma GCC unroll 7
                for (int q = 1; q < 8; ++q) {
                    wr[q] = _mm512_set1_pd(w_re[q * tw]);
                    wi[q] = _mm512_set1_pd(sg * w_im[q * tw]);
                }
                for (size_t k = 0; k < stride; ++k) {
                    __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
                    for (int q = 0; q < 8; ++q) {
                        r[q] = _mm512_load_pd(ir + 8 * (ib + k + q * stride));
                        i[q] = _mm512_load_pd(ii + 8 * (ib + k + q * stride));
                    }
                    r8_tw<INV>(r, i, wr, wi, yr, yi, v_C);
#pragma GCC unroll 8
                    for (int p = 0; p < 8; ++p) {
                        _mm512_store_pd(orr + 8 * (ob + k + p * os), yr[p]);
                        _mm512_store_pd(oi + 8 * (ob + k + p * os), yi[p]);
                    }
                }
            }
            m <<= 3;
        }
        else if (rem % 3 == 0) {
            // radix-3 (FFT lengths 3*2^k), any stride; same Stockham indexing and twiddles as the radix-8 stage:
            //   input ib + k + q*stride, twiddle W_R^(q*j*stride), output ob + k + p*m*stride
            // y0 = t0 + t1 + t2, y1,2 = t0 - (t1+t2)/2 -/+ i*sg*sin(2pi/3)*(t1-t2)   (t_q = x_q * twiddle_q)
            const size_t stride = rem / 3;
            const size_t os = m * stride;
            const __m512d v_half = _mm512_set1_pd(0.5);
            const __m512d v_s3 = _mm512_set1_pd(0.86602540378443864676 * sg);
            for (size_t j = 0; j < m; ++j) {
                const size_t tw = j * stride;
                const __m512d w1r = _mm512_set1_pd(w_re[tw]),     w1i = _mm512_set1_pd(sg * w_im[tw]);
                const __m512d w2r = _mm512_set1_pd(w_re[2 * tw]), w2i = _mm512_set1_pd(sg * w_im[2 * tw]);
                const size_t ib = j * 3 * stride;
                const size_t ob = j * stride;
                for (size_t k = 0; k < stride; ++k) {
                    const __m512d r0 = _mm512_load_pd(ir + 8 * (ib + k)),              i0 = _mm512_load_pd(ii + 8 * (ib + k));
                    const __m512d x1r = _mm512_load_pd(ir + 8 * (ib + k + stride)),     x1i = _mm512_load_pd(ii + 8 * (ib + k + stride));
                    const __m512d x2r = _mm512_load_pd(ir + 8 * (ib + k + 2 * stride)), x2i = _mm512_load_pd(ii + 8 * (ib + k + 2 * stride));
                    const __m512d t1r = _mm512_fmsub_pd(x1r, w1r, _mm512_mul_pd(x1i, w1i)), t1i = _mm512_fmadd_pd(x1r, w1i, _mm512_mul_pd(x1i, w1r));
                    const __m512d t2r = _mm512_fmsub_pd(x2r, w2r, _mm512_mul_pd(x2i, w2i)), t2i = _mm512_fmadd_pd(x2r, w2i, _mm512_mul_pd(x2i, w2r));
                    const __m512d sr = _mm512_add_pd(t1r, t2r), si = _mm512_add_pd(t1i, t2i);
                    const __m512d dr = _mm512_sub_pd(t1r, t2r), di = _mm512_sub_pd(t1i, t2i);
                    const __m512d mr = _mm512_fnmadd_pd(v_half, sr, r0), mi = _mm512_fnmadd_pd(v_half, si, i0);
                    _mm512_store_pd(orr + 8 * (ob + k),          _mm512_add_pd(r0, sr));
                    _mm512_store_pd(oi + 8 * (ob + k),           _mm512_add_pd(i0, si));
                    _mm512_store_pd(orr + 8 * (ob + k + os),     _mm512_fmadd_pd(v_s3, di, mr));
                    _mm512_store_pd(oi + 8 * (ob + k + os),      _mm512_fnmadd_pd(v_s3, dr, mi));
                    _mm512_store_pd(orr + 8 * (ob + k + 2 * os), _mm512_fnmadd_pd(v_s3, di, mr));
                    _mm512_store_pd(oi + 8 * (ob + k + 2 * os),  _mm512_fmadd_pd(v_s3, dr, mi));
                }
            }
            m *= 3;
        }
        else if (rem % 7 == 0) {
            // radix-7 (FFT lengths 7*2^k), same Stockham indexing and twiddles as radix-3. With t_q = x_q * twiddle_q,
            // pairs s_q = t_q + t_(7-q), d_q = t_q - t_(7-q) (q = 1..3) and c_j = cos(2 pi j/7), n_j = sg*sin(2 pi j/7):
            //   y0 = t0 + s1 + s2 + s3
            //   A1 = t0 + c1 s1 + c2 s2 + c3 s3   B1 = n1 d1 + n2 d2 + n3 d3
            //   A2 = t0 + c2 s1 + c3 s2 + c1 s3   B2 = n2 d1 - n3 d2 - n1 d3
            //   A3 = t0 + c3 s1 + c1 s2 + c2 s3   B3 = n3 d1 - n1 d2 + n2 d3
            //   y_p = A_p - i B_p, y_(7-p) = A_p + i B_p
            const size_t stride = rem / 7;
            const size_t os = m * stride;
            const __m512d c1 = _mm512_set1_pd(0.62348980185873353053), c2 = _mm512_set1_pd(-0.22252093395631440429),
                          c3 = _mm512_set1_pd(-0.90096886790241912624);
            const __m512d n1 = _mm512_set1_pd(0.78183148246802980871 * sg), n2 = _mm512_set1_pd(0.97492791218182360702 * sg),
                          n3 = _mm512_set1_pd(0.43388373911755812048 * sg);
            for (size_t j = 0; j < m; ++j) {
                const size_t tw = j * stride;
                __m512d wr[7], wi[7];
#pragma GCC unroll 6
                for (int q = 1; q < 7; ++q) {
                    wr[q] = _mm512_set1_pd(w_re[q * tw]);
                    wi[q] = _mm512_set1_pd(sg * w_im[q * tw]);
                }
                const size_t ib = j * 7 * stride;
                const size_t ob = j * stride;
                for (size_t k = 0; k < stride; ++k) {
                    __m512d tr[7], ti[7];
                    tr[0] = _mm512_load_pd(ir + 8 * (ib + k));
                    ti[0] = _mm512_load_pd(ii + 8 * (ib + k));
#pragma GCC unroll 6
                    for (int q = 1; q < 7; ++q) {
                        const __m512d xr = _mm512_load_pd(ir + 8 * (ib + k + q * stride));
                        const __m512d xi = _mm512_load_pd(ii + 8 * (ib + k + q * stride));
                        if (j == 0) { tr[q] = xr; ti[q] = xi; }
                        else {
                            tr[q] = _mm512_fmsub_pd(xr, wr[q], _mm512_mul_pd(xi, wi[q]));
                            ti[q] = _mm512_fmadd_pd(xr, wi[q], _mm512_mul_pd(xi, wr[q]));
                        }
                    }
                    const __m512d s1r = _mm512_add_pd(tr[1], tr[6]), s1i = _mm512_add_pd(ti[1], ti[6]);
                    const __m512d s2r = _mm512_add_pd(tr[2], tr[5]), s2i = _mm512_add_pd(ti[2], ti[5]);
                    const __m512d s3r = _mm512_add_pd(tr[3], tr[4]), s3i = _mm512_add_pd(ti[3], ti[4]);
                    const __m512d d1r = _mm512_sub_pd(tr[1], tr[6]), d1i = _mm512_sub_pd(ti[1], ti[6]);
                    const __m512d d2r = _mm512_sub_pd(tr[2], tr[5]), d2i = _mm512_sub_pd(ti[2], ti[5]);
                    const __m512d d3r = _mm512_sub_pd(tr[3], tr[4]), d3i = _mm512_sub_pd(ti[3], ti[4]);
                    const __m512d y0r = _mm512_add_pd(tr[0], _mm512_add_pd(s1r, _mm512_add_pd(s2r, s3r)));
                    const __m512d y0i = _mm512_add_pd(ti[0], _mm512_add_pd(s1i, _mm512_add_pd(s2i, s3i)));
                    const __m512d a1r = _mm512_fmadd_pd(c3, s3r, _mm512_fmadd_pd(c2, s2r, _mm512_fmadd_pd(c1, s1r, tr[0])));
                    const __m512d a1i = _mm512_fmadd_pd(c3, s3i, _mm512_fmadd_pd(c2, s2i, _mm512_fmadd_pd(c1, s1i, ti[0])));
                    const __m512d a2r = _mm512_fmadd_pd(c1, s3r, _mm512_fmadd_pd(c3, s2r, _mm512_fmadd_pd(c2, s1r, tr[0])));
                    const __m512d a2i = _mm512_fmadd_pd(c1, s3i, _mm512_fmadd_pd(c3, s2i, _mm512_fmadd_pd(c2, s1i, ti[0])));
                    const __m512d a3r = _mm512_fmadd_pd(c2, s3r, _mm512_fmadd_pd(c1, s2r, _mm512_fmadd_pd(c3, s1r, tr[0])));
                    const __m512d a3i = _mm512_fmadd_pd(c2, s3i, _mm512_fmadd_pd(c1, s2i, _mm512_fmadd_pd(c3, s1i, ti[0])));
                    const __m512d b1r = _mm512_fmadd_pd(n3, d3r, _mm512_fmadd_pd(n2, d2r, _mm512_mul_pd(n1, d1r)));
                    const __m512d b1i = _mm512_fmadd_pd(n3, d3i, _mm512_fmadd_pd(n2, d2i, _mm512_mul_pd(n1, d1i)));
                    const __m512d b2r = _mm512_fnmadd_pd(n1, d3r, _mm512_fnmadd_pd(n3, d2r, _mm512_mul_pd(n2, d1r)));
                    const __m512d b2i = _mm512_fnmadd_pd(n1, d3i, _mm512_fnmadd_pd(n3, d2i, _mm512_mul_pd(n2, d1i)));
                    const __m512d b3r = _mm512_fmadd_pd(n2, d3r, _mm512_fnmadd_pd(n1, d2r, _mm512_mul_pd(n3, d1r)));
                    const __m512d b3i = _mm512_fmadd_pd(n2, d3i, _mm512_fnmadd_pd(n1, d2i, _mm512_mul_pd(n3, d1i)));
                    // y_p = A - i B: (Ar + Bi, Ai - Br); y_(7-p) = A + i B: (Ar - Bi, Ai + Br)
                    _mm512_store_pd(orr + 8 * (ob + k),          y0r);
                    _mm512_store_pd(oi + 8 * (ob + k),           y0i);
                    _mm512_store_pd(orr + 8 * (ob + k + os),     _mm512_add_pd(a1r, b1i));
                    _mm512_store_pd(oi + 8 * (ob + k + os),      _mm512_sub_pd(a1i, b1r));
                    _mm512_store_pd(orr + 8 * (ob + k + 6 * os), _mm512_sub_pd(a1r, b1i));
                    _mm512_store_pd(oi + 8 * (ob + k + 6 * os),  _mm512_add_pd(a1i, b1r));
                    _mm512_store_pd(orr + 8 * (ob + k + 2 * os), _mm512_add_pd(a2r, b2i));
                    _mm512_store_pd(oi + 8 * (ob + k + 2 * os),  _mm512_sub_pd(a2i, b2r));
                    _mm512_store_pd(orr + 8 * (ob + k + 5 * os), _mm512_sub_pd(a2r, b2i));
                    _mm512_store_pd(oi + 8 * (ob + k + 5 * os),  _mm512_add_pd(a2i, b2r));
                    _mm512_store_pd(orr + 8 * (ob + k + 3 * os), _mm512_add_pd(a3r, b3i));
                    _mm512_store_pd(oi + 8 * (ob + k + 3 * os),  _mm512_sub_pd(a3i, b3r));
                    _mm512_store_pd(orr + 8 * (ob + k + 4 * os), _mm512_sub_pd(a3r, b3i));
                    _mm512_store_pd(oi + 8 * (ob + k + 4 * os),  _mm512_add_pd(a3i, b3r));
                }
            }
            m *= 7;
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
        if (c_re == nullptr) {
            std::swap(ir, orr);
            std::swap(ii, oi);
            in_b = !in_b;
        }
        else {
            // three buffers: the next stage reads this output and writes the other of b/c, the last one into a
            ir = orr; ii = oi;
            const size_t rem2 = (m < R) ? R / m : 0;
            const bool last = (rem2 == 8 || rem2 == 7 || rem2 == 4 || rem2 == 3 || rem2 == 2);
            if (last)            { orr = a_re; oi = a_im; }
            else if (ir == b_re) { orr = c_re; oi = c_im; }
            else                 { orr = b_re; oi = b_im; }
            in_b = (ir != a_re);
        }
    }
    return in_b;
}

static inline bool fft_batch8(double* __restrict a_re, double* __restrict a_im, double* __restrict b_re, double* __restrict b_im,
                              size_t R, const double* w_re0, const double* w_im0, bool inverse, size_t wstride = 1,
                              double* c_re = nullptr, double* c_im = nullptr) {
    return inverse ? fft_batch8_t<true>(a_re, a_im, b_re, b_im, R, w_re0, w_im0, wstride, c_re, c_im)
                   : fft_batch8_t<false>(a_re, a_im, b_re, b_im, R, w_re0, w_im0, wstride, c_re, c_im);
}

// Number of fft_batch8 stages for length R (same radix sequence as fft_batch8: 8, 3, 7, then 4 or 2).
static int batch8_stages(size_t R) {
    int s = 0;
    for (size_t m = 1; m < R; ++s) {
        const size_t rem = R / m;
        m *= (rem % 8 == 0) ? 8 : ((rem % 3 == 0) ? 3 : ((rem % 7 == 0) ? 7 : ((rem == 4) ? 4 : 2)));
    }
    return s;
}

// From this column length on, batch-8 computes the FFT in two levels (a block of 8 columns x R no longer fits in L1).
// LOTR_B8_2L_MIN=<R> changes the threshold (experiments; R must be a multiple of 8 when two-level is used).
static inline bool batch8_two_level(size_t R) {
    static const size_t min_r = []() {
        // 256: with the in-place form (b8_inplace) two levels beat one level at R = 256 too
        const char* e = std::getenv("LOTR_B8_2L_MIN");
        return e ? (size_t)std::strtoull(e, nullptr, 10) : (size_t)256;
    }();
    return R >= min_r && (R % 8) == 0;
}

// Position of vector X[K] in the "blocked" output layout of fft_batch8_2l: block p = K & 7 (R/8 vectors, starting at
// p*S, see b8_S), position K >> 3.
// S = vector stride of the 8 blocks of the two-level layout (block p = K & 7 starts at vector p*S).
static inline size_t batch8_posL(size_t K, size_t S) { return (K & 7) * S + (K >> 3); }

// S = R/8 + 1: one spare vector (64 B) per block breaks the power-of-two stride (R/8 vectors = 4 KB for R = 512) that
// would put the 8 blocks into the same L1 sets (stage 1 and the square ~2x slower). LOTR_B8_NOGAP=1 = old layout (S = R/8).
static inline size_t b8_S(size_t R) {
    static const bool nogap = (std::getenv("LOTR_B8_NOGAP") != nullptr);
    return (R >> 3) + (nogap ? 0 : 1);
}
// Doubles of one re or im array holding a block of 8 columns (natural layout 8R, two-level layout 64*S >= 8R).
static inline size_t b8_len(size_t R) { return 64 * b8_S(R); }

// IN-PLACE two-level form (R = 512: -18%, stage 1 -37%): the block of 8 columns
// does not fit L1 (64 KB per re/im for R = 512), so a separate stage-1 output buffer doubled the L2 traffic.
// The natural input is kept in the GAPPED layout, row r at gpos(r) = (r / R2)*S + r % R2: stage 1 (DIF) then reads
// q*S + k and writes p*S + k, the same positions; the sub-FFTs run through a 16 KB scratch and, with an even number
// of stages, end back in their block. The inverse runs DIT: sub-FFTs first, then the combining stage in place, back
// to the gapped natural layout. Used when R/8 has an even number of fft_batch8 stages (R = 512); LOTR_B8_OUTPLACE=1 = off.
// An odd number of sub-FFT stages (R = 768, 1024) uses a second scratch (fft_batch8 with three buffers, the last stage
// writes back into the block); LOTR_B8_IP_EVEN=1 = in place only for an even number (R = 512).
static inline bool b8_inplace(size_t R) {
    static const bool off = (std::getenv("LOTR_B8_OUTPLACE") != nullptr);
    static const bool even_only = (std::getenv("LOTR_B8_IP_EVEN") != nullptr);
    if (off || !batch8_two_level(R)) return false;
    const int st = batch8_stages(R >> 3);
    return (st % 2) == 0 || (!even_only && st >= 2);
}
static inline size_t b8_gpos(size_t r, size_t R2, size_t S) { return (r / R2) * S + (r % R2); }
// The fused middles also run the inverse in place (DIT, b8_ip_ditf). Checked by KBENCH for R = 512/768/1024.
// LOTR_B8_IP_MID_EVEN=1: in the middles only with an even number of sub-FFT stages (R = 512).
static inline bool b8_inplace_mid(size_t R) {
    static const bool even_only = (std::getenv("LOTR_B8_IP_MID_EVEN") != nullptr);
    return b8_inplace(R) && (!even_only || (batch8_stages(R >> 3) % 2) == 0);
}

// DIF stage 1 in place for k in [k_begin, k_end): x_q = row k + R2*q at q*S + k -> y_p * W_R^(p*k) at p*S + k.
// HALF: rows >= R/2 are zero (q >= 4 not read).
template <bool HALF, bool INV>
static void b8_ip_stage1(double* __restrict x_re, double* __restrict x_im, size_t R, const double* w_re, const double* w_im,
                         size_t k_begin, size_t k_end) {
    const size_t S = b8_S(R);
    constexpr double sg = INV ? -1.0 : 1.0;
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    for (size_t k = k_begin; k < k_end; ++k) {
        __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int q = 0; q < 8; ++q) {
            if (HALF && q >= 4) { r[q] = _mm512_setzero_pd(); i[q] = _mm512_setzero_pd(); continue; }
            r[q] = _mm512_load_pd(x_re + 8 * (q * S + k));
            i[q] = _mm512_load_pd(x_im + 8 * (q * S + k));
        }
        r8_core<INV>(r, i, yr, yi, v_C);
        _mm512_store_pd(x_re + 8 * k, yr[0]);
        _mm512_store_pd(x_im + 8 * k, yi[0]);
#pragma GCC unroll 7
        for (int p = 1; p < 8; ++p) {
            const __m512d wr = _mm512_set1_pd(w_re[(size_t)p * k]);
            const __m512d wi = _mm512_set1_pd(sg * w_im[(size_t)p * k]);
            _mm512_store_pd(x_re + 8 * (p * S + k), _mm512_fmsub_pd(yr[p], wr, _mm512_mul_pd(yi[p], wi)));
            _mm512_store_pd(x_im + 8 * (p * S + k), _mm512_fmadd_pd(yr[p], wi, _mm512_mul_pd(yi[p], wr)));
        }
    }
}

// DIT combining stage in place for k in [k_begin, k_end) (the last step of the inverse): y_p at p*S + k (sub-FFT outputs),
// t_p = y_p * W_R^(p*k) (direction sign), x = radix-8 over p -> x_q (row k + R2*q) at q*S + k.
template <bool INV>
static void b8_ip_dit(double* __restrict x_re, double* __restrict x_im, size_t R, const double* w_re, const double* w_im,
                      size_t k_begin, size_t k_end) {
    const size_t S = b8_S(R);
    constexpr double sg = INV ? -1.0 : 1.0;
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    for (size_t k = k_begin; k < k_end; ++k) {
        __m512d r[8], i[8], yr[8], yi[8];
        r[0] = _mm512_load_pd(x_re + 8 * k);
        i[0] = _mm512_load_pd(x_im + 8 * k);
#pragma GCC unroll 7
        for (int p = 1; p < 8; ++p) {
            const __m512d ar = _mm512_load_pd(x_re + 8 * (p * S + k)), ai = _mm512_load_pd(x_im + 8 * (p * S + k));
            const __m512d wr = _mm512_set1_pd(w_re[(size_t)p * k]);
            const __m512d wi = _mm512_set1_pd(sg * w_im[(size_t)p * k]);
            r[p] = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));
            i[p] = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));
        }
        r8_core<INV>(r, i, yr, yi, v_C);
#pragma GCC unroll 8
        for (int q = 0; q < 8; ++q) {
            _mm512_store_pd(x_re + 8 * (q * S + k), yr[q]);
            _mm512_store_pd(x_im + 8 * (q * S + k), yi[q]);
        }
    }
}

// Length-R2 FFT of the R2 consecutive vectors x (natural order in and out) as DIT R2 = 8 * m: eight
// m-point FFTs of the decimated inputs x[8 n1 + n2] (each m vectors, in L1), then one radix-8 combining stage
//   X[k1 + m k2] = sum_n2 W_8^(n2 k2) * (W_R2^(n2 k1) * F_n2[k1]),   F_n2 = DFT_m(x[8 n1 + n2])
// written straight to the natural positions. The block (R2 vectors, e.g. 384 x 64 B x 2 = 49 KB for R = 3072: more than
// the 48 KB L1) goes through L2 about twice instead of once per Stockham stage (4 stages for 384).
// w_re/w_im: roots of order R2*wstride... used with wstride (R2-th roots = every wstride-th). Scratch sc: R2 vectors for
// the F_n2 (8*R2 doubles per array) + 2 * m vectors for the m-point FFT (16*m doubles): 10*R2 doubles per array.
// No copies: the first radix-8 stage of each m-point FFT reads the decimated input straight from x
// (vector 8*n1 + n2) and the remaining stages (fft_batch8_t from m0 = 8) end directly in T (the buffers are chosen by
// the parity of the remaining stage count). Needs m % 8 == 0 (R2 % 64 == 0: 256, 384, ...).
static int batch8_stages_from(size_t R, size_t m0) {
    int s = 0;
    for (size_t m = m0; m < R; ++s) {
        const size_t rem = R / m;
        m *= (rem % 8 == 0) ? 8 : ((rem % 3 == 0) ? 3 : ((rem % 7 == 0) ? 7 : ((rem == 4) ? 4 : 2)));
    }
    return s;
}
template <bool INV>
static void b8_sub_dit8(double* __restrict x_re, double* __restrict x_im, double* __restrict sc_re, double* __restrict sc_im,
                        size_t R2, const double* w_re, const double* w_im, size_t wstride) {
    const size_t m = R2 >> 3;
    const size_t s = m >> 3;   // stride of the first radix-8 stage of the m-point FFT
    constexpr double sg = INV ? -1.0 : 1.0;
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    double* const T_re = sc_re;               double* const T_im = sc_im;               // F_n2 at n2*m .. n2*m + m-1
    double* const P_re = sc_re + 8 * R2;      double* const P_im = sc_im + 8 * R2;      // m vectors of scratch
    const int rest = batch8_stages_from(m, 8);   // stages after the first radix-8 one
    for (size_t n2 = 0; n2 < 8; ++n2) {
        double* const Tn_re = T_re + 8 * n2 * m;
        double* const Tn_im = T_im + 8 * n2 * m;
        // first stage straight from the decimated input: element n1 = k + s*q is vector 8*n1 + n2 of x; output k + p*s
        double* const D_re = (rest % 2 == 0) ? Tn_re : P_re;
        double* const D_im = (rest % 2 == 0) ? Tn_im : P_im;
        for (size_t k = 0; k < s; ++k) {
            __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
            for (int q = 0; q < 8; ++q) {
                const size_t v = 8 * (k + s * (size_t)q) + n2;
                r[q] = _mm512_load_pd(x_re + 8 * v);
                i[q] = _mm512_load_pd(x_im + 8 * v);
            }
            r8_core<INV>(r, i, yr, yi, v_C);
#pragma GCC unroll 8
            for (int p = 0; p < 8; ++p) {
                _mm512_store_pd(D_re + 8 * (k + (size_t)p * s), yr[p]);
                _mm512_store_pd(D_im + 8 * (k + (size_t)p * s), yi[p]);
            }
        }
        // the remaining stages (roots of order m = every (8 * wstride)-th of the table): an even count ends where it
        // starts (T), an odd count in the second buffer (T)
        if (rest > 0) {
            if (rest % 2 == 0) fft_batch8_t<INV>(Tn_re, Tn_im, P_re, P_im, m, w_re, w_im, 8 * wstride, nullptr, nullptr, 8);
            else               fft_batch8_t<INV>(P_re, P_im, Tn_re, Tn_im, m, w_re, w_im, 8 * wstride, nullptr, nullptr, 8);
        }
    }
    for (size_t k1 = 0; k1 < m; ++k1) {
        __m512d r[8], i[8], yr[8], yi[8];
        r[0] = _mm512_load_pd(T_re + 8 * k1);
        i[0] = _mm512_load_pd(T_im + 8 * k1);
#pragma GCC unroll 7
        for (int n2 = 1; n2 < 8; ++n2) {
            const __m512d ar = _mm512_load_pd(T_re + 8 * ((size_t)n2 * m + k1));
            const __m512d ai = _mm512_load_pd(T_im + 8 * ((size_t)n2 * m + k1));
            const size_t e = (size_t)n2 * k1 * wstride;   // W_R2^(n2 k1), n2*k1 < R2
            const __m512d wr = _mm512_set1_pd(w_re[e]);
            const __m512d wi = _mm512_set1_pd(sg * w_im[e]);
            r[n2] = _mm512_fmsub_pd(ar, wr, _mm512_mul_pd(ai, wi));
            i[n2] = _mm512_fmadd_pd(ar, wi, _mm512_mul_pd(ai, wr));
        }
        r8_core<INV>(r, i, yr, yi, v_C);
#pragma GCC unroll 8
        for (int k2 = 0; k2 < 8; ++k2) {
            _mm512_store_pd(x_re + 8 * (k1 + m * (size_t)k2), yr[k2]);
            _mm512_store_pd(x_im + 8 * (k1 + m * (size_t)k2), yi[k2]);
        }
    }
}

// Sub-FFTs of length R2 from this length on through b8_sub_dit8 (LOTR_B8_DIT8_MIN=<R2>, default 256: R = 2048, 3072;
// 0 = never; R2 % 64 == 0). Faster than the Stockham sub-FFTs (R = 2048: -17%) and fewer rounding stages (smaller
// IBDWT calibration error).
static inline bool b8_use_dit8(size_t R2) {
    static const size_t mn = []() {
        const char* e = std::getenv("LOTR_B8_DIT8_MIN");
        return e ? (size_t)std::strtoull(e, nullptr, 10) : (size_t)256;
    }();
    return mn != 0 && R2 >= mn && (R2 % 64) == 0;
}

// Sub-FFTs of blocks [p_begin, p_end) in place through the scratch sc (2 * R/8 vectors per re/im array: an even number
// of stages ping-pongs with the first half and ends back in the block; an odd number goes block -> half 1 -> half 2 ->
// ... -> block, fft_batch8 with three buffers). Long sub-FFTs (b8_use_dit8): b8_sub_dit8, scratch 10 * R/8 * 8 doubles.
static void b8_ip_subs(double* __restrict x_re, double* __restrict x_im, double* __restrict sc_re, double* __restrict sc_im,
                       size_t R, const double* w_re, const double* w_im, bool inverse, size_t p_begin, size_t p_end) {
    const size_t R2 = R >> 3, S = b8_S(R);
    if (b8_use_dit8(R2)) {
        for (size_t p = p_begin; p < p_end; ++p) {
            if (inverse) b8_sub_dit8<true>(x_re + 8 * p * S, x_im + 8 * p * S, sc_re, sc_im, R2, w_re, w_im, 8);
            else         b8_sub_dit8<false>(x_re + 8 * p * S, x_im + 8 * p * S, sc_re, sc_im, R2, w_re, w_im, 8);
        }
        return;
    }
    const bool odd = (batch8_stages(R2) % 2) != 0;
    double* const s2_re = odd ? sc_re + 8 * R2 : nullptr;
    double* const s2_im = odd ? sc_im + 8 * R2 : nullptr;
    for (size_t p = p_begin; p < p_end; ++p)
        fft_batch8(x_re + 8 * p * S, x_im + 8 * p * S, sc_re, sc_im, R2, w_re, w_im, inverse, 8, s2_re, s2_im);
}

// Whole in-place transforms of one block of 8 columns (sc: scratch of R/8 vectors per array).
// Forward/any direction, natural gapped input -> blocked spectrum (DIF):
static void b8_ip_dif(double* x_re, double* x_im, double* sc_re, double* sc_im, size_t R, const double* w_re,
                      const double* w_im, bool inverse, bool half) {
    const size_t R2 = R >> 3;
    if (inverse) {
        if (half) b8_ip_stage1<true, true>(x_re, x_im, R, w_re, w_im, 0, R2);
        else      b8_ip_stage1<false, true>(x_re, x_im, R, w_re, w_im, 0, R2);
    }
    else {
        if (half) b8_ip_stage1<true, false>(x_re, x_im, R, w_re, w_im, 0, R2);
        else      b8_ip_stage1<false, false>(x_re, x_im, R, w_re, w_im, 0, R2);
    }
    b8_ip_subs(x_re, x_im, sc_re, sc_im, R, w_re, w_im, inverse, 0, 8);
}
// Blocked spectrum -> natural gapped output (DIT):
static void b8_ip_ditf(double* x_re, double* x_im, double* sc_re, double* sc_im, size_t R, const double* w_re,
                       const double* w_im, bool inverse) {
    const size_t R2 = R >> 3;
    b8_ip_subs(x_re, x_im, sc_re, sc_im, R, w_re, w_im, inverse, 0, 8);
    if (inverse) b8_ip_dit<true>(x_re, x_im, R, w_re, w_im, 0, R2);
    else         b8_ip_dit<false>(x_re, x_im, R, w_re, w_im, 0, R2);
}

// =====================================================================
// TWO-LEVEL batch-8 FFT (DIF), R = 8 * R2:
//   1) for k < R2: y_p = DFT8 over q of x[k + R2*q], then y_p *= W_R^(p*k)  -> block p, position k
//   2) in each of the 8 blocks a length-R2 FFT (block ~8-16 KB, computed in L1) -> X[p + 8*k'] at position k' of block p.
// Output always in out_*, in the blocked layout (batch8_posL). in_L = true: the input is in the blocked layout too (e.g. the
// spectrum after the square in the fusion). in_* must not overlap out_* or tmp_*. w_re/w_im: R-th roots of unity (length R).
// HALF = true: input elements R/2 .. R-1 are zero and are neither read nor need to exist (forward pass A: the number fills
// only the lower half of the FFT buffer). Requires in_L = false.
// =====================================================================
// Stage 1 of the two-level FFT for k in [k_begin, k_end): radix-8 over x[k + R2*q], twiddle W_R^(p*k) -> s1 block p,
// position k (the k are independent, so the range can be split between threads).
template <bool HALF, bool INV>
static void batch8_2l_stage1_t(const double* __restrict in_re, const double* __restrict in_im, bool in_L,
                               double* __restrict s1_re, double* __restrict s1_im,
                               size_t R, const double* w_re, const double* w_im, size_t k_begin, size_t k_end) {
    const size_t R2 = R >> 3;
    const size_t S = b8_S(R);   // block stride of the two-level layout (input with in_L, and s1)
    constexpr double sg = INV ? -1.0 : 1.0;
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);

    for (size_t k = k_begin; k < k_end; ++k) {
        __m512d r[8], i[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int q = 0; q < 8; ++q) {
            if (HALF && q >= 4) { r[q] = _mm512_setzero_pd(); i[q] = _mm512_setzero_pd(); continue; }
            const size_t nidx = k + R2 * (size_t)q;
            const size_t pos = in_L ? batch8_posL(nidx, S) : nidx;
            r[q] = _mm512_load_pd(in_re + 8 * pos);
            i[q] = _mm512_load_pd(in_im + 8 * pos);
        }
        r8_core<INV>(r, i, yr, yi, v_C);   // no input twiddles in stage 1
        _mm512_store_pd(s1_re + 8 * k, yr[0]);
        _mm512_store_pd(s1_im + 8 * k, yi[0]);
#pragma GCC unroll 7
        for (int p = 1; p < 8; ++p) {
            const __m512d wr = _mm512_set1_pd(w_re[(size_t)p * k]);
            const __m512d wi = _mm512_set1_pd(sg * w_im[(size_t)p * k]);
            _mm512_store_pd(s1_re + 8 * (p * S + k), _mm512_fmsub_pd(yr[p], wr, _mm512_mul_pd(yi[p], wi)));
            _mm512_store_pd(s1_im + 8 * (p * S + k), _mm512_fmadd_pd(yr[p], wi, _mm512_mul_pd(yi[p], wr)));
        }
    }
}

template <bool HALF>
static inline void batch8_2l_stage1(const double* __restrict in_re, const double* __restrict in_im, bool in_L,
                                    double* __restrict s1_re, double* __restrict s1_im,
                                    size_t R, const double* w_re, const double* w_im, bool inverse, size_t k_begin, size_t k_end) {
    if (inverse) batch8_2l_stage1_t<HALF, true>(in_re, in_im, in_L, s1_re, s1_im, R, w_re, w_im, k_begin, k_end);
    else         batch8_2l_stage1_t<HALF, false>(in_re, in_im, in_L, s1_re, s1_im, R, w_re, w_im, k_begin, k_end);
}

// Stage 1 of the two-level FFT writes the buffer from which the sub-FFTs (ping-pong) end up in out: tmp when the
// sub-FFT of length R/8 has an odd number of stages.
static inline bool batch8_2l_s1_in_tmp(size_t R) { return (batch8_stages(R >> 3) & 1) != 0; }

template <bool HALF>
static void fft_batch8_2l_t(const double* __restrict in_re, const double* __restrict in_im, bool in_L,
                            double* __restrict out_re, double* __restrict out_im, double* __restrict tmp_re, double* __restrict tmp_im,
                            size_t R, const double* w_re, const double* w_im, bool inverse, uint64_t* st1_ticks = nullptr) {
    (void)st1_ticks;
    SUB_T(ts0);
    const size_t R2 = R >> 3;
    const size_t S = b8_S(R);

    // stage 1 writes to the buffer from which the sub-FFTs (ping-pong) end up in out_*
    const bool odd = batch8_2l_s1_in_tmp(R);
    double* const s1_re = odd ? tmp_re : out_re;
    double* const s1_im = odd ? tmp_im : out_im;
    double* const s2_re = odd ? out_re : tmp_re;
    double* const s2_im = odd ? out_im : tmp_im;

    batch8_2l_stage1<HALF>(in_re, in_im, in_L, s1_re, s1_im, R, w_re, w_im, inverse, 0, R2);
#ifdef LOTR_PROFILE
    if (st1_ticks) *st1_ticks += __rdtsc() - ts0;
#endif
    for (size_t p = 0; p < 8; ++p) {
        fft_batch8(s1_re + 8 * p * S, s1_im + 8 * p * S, s2_re + 8 * p * S, s2_im + 8 * p * S,
                   R2, w_re, w_im, inverse, 8);
    }
}

static inline void fft_batch8_2l(const double* __restrict in_re, const double* __restrict in_im, bool in_L,
                                 double* __restrict out_re, double* __restrict out_im, double* __restrict tmp_re, double* __restrict tmp_im,
                                 size_t R, const double* w_re, const double* w_im, bool inverse, uint64_t* st1_ticks = nullptr) {
    fft_batch8_2l_t<false>(in_re, in_im, in_L, out_re, out_im, tmp_re, tmp_im, R, w_re, w_im, inverse, st1_ticks);
}

// Thread buffer: per block of 8 columns, 3 buffers (a, b, c) of b8_len(R) doubles for re and im (6 arrays) + 'extra';
// 64 B aligned. Array i of block 0 starts at i*b8_len(R).
static double* batch8_scratch(size_t R, size_t blocks, size_t extra) {
    static thread_local double* scr = nullptr;
    static thread_local size_t cap = 0;
    const size_t need = blocks * 6 * b8_len(R) + extra;
    if (cap < need) {
        std::free(scr);
        scr = static_cast<double*>(std::aligned_alloc(64, ((need * sizeof(double) + 63) / 64) * 64));
        cap = need;
    }
    return scr;
}

// Buffer shared by the threads of one pass (owned by the calling thread, 64 B aligned).
static double* coop_scratch(size_t need) {
    static thread_local double* buf = nullptr;
    static thread_local size_t cap = 0;
    if (cap < need) {
        std::free(buf);
        buf = static_cast<double*>(std::aligned_alloc(64, ((need * sizeof(double) + 63) / 64) * 64));
        cap = need;
    }
    return buf;
}

// Gather 8 columns (cstart..cstart+7) from matrix src (R rows x C columns) into vectors g (8*r), optionally with twiddle
// w^(r*col) (conjugated for inverse). special: lanes 1..7 shifted to 0..6, lane 7 = column spc (fusion block M for b = 0).
// R_load > 0: only rows 0 .. R_load-1 are read (the rest of the source is zero); zero_rest = true writes zeros
// into g for rows R_load .. R-1, false leaves them untouched (the FFT does not read them, fft_batch8_2l_t<true>).
// gapped = true: row r goes to vector b8_gpos(r) (in-place two-level form, b8_inplace) instead of r.
// tw_out_re/im (with twiddle): the twiddle vector of row r is also stored at 8*r (natural order), for a later
// multiplication by its conjugate (batch8_store_rows with tw).
static inline void batch8_gather(const double* __restrict src_re, const double* __restrict src_im, size_t R, size_t C,
                                 size_t cstart, bool special, size_t spc, bool twiddle, bool inverse,
                                 const FFTContext& ctx, const Tw2& tw2, double* __restrict g_re, double* __restrict g_im,
                                 size_t R_load = 0, bool zero_rest = true, size_t r_begin = 0, size_t r_end = ~(size_t)0,
                                 bool gapped = false, double* __restrict tw_out_re = nullptr, double* __restrict tw_out_im = nullptr) {
    if (R_load == 0 || R_load > R) R_load = R;
    if (r_end > R) r_end = R;   // rows [r_begin, r_end) only (a block shared by several threads)
    const size_t gR2 = std::max<size_t>(R >> 3, 1), gS = b8_S(R);
    auto gp = [&](size_t r) -> size_t { return gapped ? b8_gpos(r, gR2, gS) : r; };
    if (zero_rest) {
        for (size_t r = std::max(R_load, r_begin); r < r_end; ++r) {
            _mm512_store_pd(g_re + 8 * gp(r), _mm512_setzero_pd());
            _mm512_store_pd(g_im + 8 * gp(r), _mm512_setzero_pd());
        }
    }
    const double* const G_re = ctx.lane_g_re.data();
    const double* const G_im = ctx.lane_g_im.data();
    const __m512i shift1 = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 7);
    const double isg = inverse ? -1.0 : 1.0;
    // with special, the columns of lanes 0..6 are cstart+1 .. cstart+7: twiddle computed from cstart+1
    const size_t c_base = special ? cstart + 1 : cstart;
    const size_t r_stop = std::min(R_load, r_end);
    // twiddle vector of row r (lanes = the block's columns), from the tables: w^(r*c_base) * w^(r*b) (+ special lane)
    auto exact_tw = [&](size_t r, __m512d& wr, __m512d& wi) {
        double s_re, s_im;
        tw2.get(r * c_base, s_re, s_im);                       // w^(r*c_base); r*c < R*C = N, no reduction needed
        const __m512d gr = _mm512_loadu_pd(G_re + 8 * r);     // w^(r*b) (std::vector: no 64 B alignment guarantee)
        const __m512d gi = _mm512_loadu_pd(G_im + 8 * r);
        const __m512d Sr = _mm512_set1_pd(s_re), Si = _mm512_set1_pd(s_im);
        wr = _mm512_fmsub_pd(gr, Sr, _mm512_mul_pd(gi, Si));
        wi = _mm512_fmadd_pd(gr, Si, _mm512_mul_pd(gi, Sr));
        if (special) {
            double e_re, e_im;
            tw2.get(r * spc, e_re, e_im);
            wr = _mm512_mask_mov_pd(wr, 0x80, _mm512_set1_pd(e_re));
            wi = _mm512_mask_mov_pd(wi, 0x80, _mm512_set1_pd(e_im));
        }
    };
    // ctx.tw_fast (classic real-FFT square): exact twiddles only every 8 rows (row rb), between them
    // w(rb + t) = w(rb) * P_t with P_t = w(t) (8 vectors per block): one complex multiply per row instead of the table
    // lookups. Two table products instead of one: slightly larger twiddle error, so not for the IBDWT (close to its error
    // limit).
    const bool fast = twiddle && ctx.tw_fast;
    __m512d P_re[8], P_im[8], Tb_re = _mm512_set1_pd(1.0), Tb_im = _mm512_setzero_pd();
    if (fast) {
        P_re[0] = _mm512_set1_pd(1.0); P_im[0] = _mm512_setzero_pd();
        for (size_t t = 1; t < 8; ++t) exact_tw(t, P_re[t], P_im[t]);
    }
    size_t rb = r_begin;   // row of the last exact twiddle (fast)
    const size_t pfd = 16;   // prefetch distance in rows
    size_t gq = r_begin / gR2, gk = r_begin % gR2;   // gapped position of row r, kept incrementally (R/8 need not be 2^k)
    for (size_t r = r_begin; r < r_stop; ++r) {
        const size_t gpos = gapped ? gq * gS + gk : r;
        if (++gk == gR2) { gk = 0; ++gq; }
        // Prefetch into L2 (T1): rows every C*8 B with C a power of two map to one L1 set (12 ways),
        // so a prefetch into L1 would evict lines before use. C = row stride (with or without padding).
        if (r + pfd < r_stop) {
            _mm_prefetch((const char*)(src_re + (r + pfd) * C + cstart), _MM_HINT_T1);
            _mm_prefetch((const char*)(src_im + (r + pfd) * C + cstart), _MM_HINT_T1);
        }
        __m512d vr = _mm512_loadu_pd(src_re + r * C + cstart);
        __m512d vi = _mm512_loadu_pd(src_im + r * C + cstart);
        if (special) {
            vr = _mm512_mask_mov_pd(_mm512_permutexvar_pd(shift1, vr), 0x80, _mm512_set1_pd(src_re[r * C + spc]));
            vi = _mm512_mask_mov_pd(_mm512_permutexvar_pd(shift1, vi), 0x80, _mm512_set1_pd(src_im[r * C + spc]));
        }
        if (twiddle) {
            __m512d wr, wi;
            if (!fast) {
                exact_tw(r, wr, wi);
            }
            else if (r == r_begin || r - rb == 8) {
                exact_tw(r, Tb_re, Tb_im);
                rb = r;
                wr = Tb_re; wi = Tb_im;
            }
            else {
                const size_t t = r - rb;
                wr = _mm512_fmsub_pd(Tb_re, P_re[t], _mm512_mul_pd(Tb_im, P_im[t]));
                wi = _mm512_fmadd_pd(Tb_re, P_im[t], _mm512_mul_pd(Tb_im, P_re[t]));
            }
            wi = _mm512_mul_pd(wi, _mm512_set1_pd(isg));
            if (tw_out_re) {
                _mm512_storeu_pd(tw_out_re + 8 * r, wr);
                _mm512_storeu_pd(tw_out_im + 8 * r, wi);
            }
            const __m512d xr = _mm512_fmsub_pd(vr, wr, _mm512_mul_pd(vi, wi));
            const __m512d xi = _mm512_fmadd_pd(vr, wi, _mm512_mul_pd(vi, wr));
            vr = xr; vi = xi;
        }
        _mm512_store_pd(g_re + 8 * gpos, vr);
        _mm512_store_pd(g_im + 8 * gpos, vi);
    }
}

// In-place layout (IBDWT): the fused middle reads ROWS k0 .. k0+7 of the natural matrix x (row stride rs,
// R = row length = the middle's FFT length) instead of columns of a transposed copy: 8 rows x 8 elements per step,
// an 8x8 transpose, then vector r = element r of the 8 rows (lane = row), times the twiddle w^(r*(k0+lane)) as in
// batch8_gather (r*k0 from Tw2, w^(r*lane) from lane_g). gapped as in batch8_gather.
static inline void batch8_gather_nat(const double* src_re, const double* src_im, size_t R, size_t rs, size_t k0,
                                     const FFTContext& ctx, const Tw2& tw2, double* __restrict g_re, double* __restrict g_im,
                                     bool gapped = false) {
    const size_t gR2 = std::max<size_t>(R >> 3, 1), gS = b8_S(R);
    const double* const G_re = ctx.lane_g_re.data();
    const double* const G_im = ctx.lane_g_im.data();
    // Software prefetch: 8 rows x re/im = 16 interleaved streams are more than the hardware prefetcher follows (with the
    // data in RAM this gather took most of the middle's time). Each step fetches the line pfd elements ahead in every row.
    // LOTR_NAT_PFD=<elements> (default 64 = 8 lines; 0 = off).
    static const size_t pfd = []() {
        const char* e = std::getenv("LOTR_NAT_PFD");
        return e ? (size_t)std::strtoull(e, nullptr, 10) : (size_t)64;
    }();
    size_t gq = 0, gk = 0;
    for (size_t r0 = 0; r0 < R; r0 += 8) {
        __m512d xr[8], xi[8];
        if (pfd != 0 && r0 + pfd < R) {
            for (int i = 0; i < 8; ++i) {
                _mm_prefetch((const char*)(src_re + (k0 + (size_t)i) * rs + r0 + pfd), _MM_HINT_T0);
                _mm_prefetch((const char*)(src_im + (k0 + (size_t)i) * rs + r0 + pfd), _MM_HINT_T0);
            }
        }
        for (int i = 0; i < 8; ++i) {
            xr[i] = _mm512_loadu_pd(src_re + (k0 + (size_t)i) * rs + r0);
            xi[i] = _mm512_loadu_pd(src_im + (k0 + (size_t)i) * rs + r0);
        }
        st_transpose8x8(xr);   // xr[t] lane i = element r0+t of row k0+i
        st_transpose8x8(xi);
        for (int t = 0; t < 8; ++t) {
            const size_t r = r0 + (size_t)t;
            const size_t gpos = gapped ? gq * gS + gk : r;
            if (++gk == gR2) { gk = 0; ++gq; }
            double s_re, s_im;
            tw2.get(r * k0, s_re, s_im);                          // w^(r*k0)
            const __m512d gr = _mm512_loadu_pd(G_re + 8 * r);    // w^(r*lane)
            const __m512d gi = _mm512_loadu_pd(G_im + 8 * r);
            const __m512d Sr = _mm512_set1_pd(s_re), Si = _mm512_set1_pd(s_im);
            const __m512d wr = _mm512_fmsub_pd(gr, Sr, _mm512_mul_pd(gi, Si));
            const __m512d wi = _mm512_fmadd_pd(gr, Si, _mm512_mul_pd(gi, Sr));
            _mm512_store_pd(g_re + 8 * gpos, _mm512_fmsub_pd(xr[t], wr, _mm512_mul_pd(xi[t], wi)));
            _mm512_store_pd(g_im + 8 * gpos, _mm512_fmadd_pd(xr[t], wi, _mm512_mul_pd(xi[t], wr)));
        }
    }
}

// Pass A with the pack fused in: columns cstart..cstart+7 of the digit matrix x (R rows of C complex positions,
// element (r, c) = position r*C + c) taken straight from y (digits8), no twiddle. R_load / zero_rest as in batch8_gather.
static inline void batch8_gather_digits(const DigitSrc& ds, size_t R, size_t C, size_t cstart,
                                        double* __restrict g_re, double* __restrict g_im, size_t R_load = 0, bool zero_rest = true,
                                        size_t r_begin = 0, size_t r_end = ~(size_t)0, bool gapped = false) {
    if (R_load == 0 || R_load > R) R_load = R;
    if (r_end > R) r_end = R;   // rows [r_begin, r_end) only (a block shared by several threads)
    const size_t gR2 = std::max<size_t>(R >> 3, 1), gS = b8_S(R);
    auto gp = [&](size_t r) -> size_t { return gapped ? b8_gpos(r, gR2, gS) : r; };
    if (zero_rest) {
        for (size_t r = std::max(R_load, r_begin); r < r_end; ++r) {
            _mm512_store_pd(g_re + 8 * gp(r), _mm512_setzero_pd());
            _mm512_store_pd(g_im + 8 * gp(r), _mm512_setzero_pd());
        }
    }
    const size_t row_bytes = (ds.bits == 16) ? 4 * C : (size_t)ds.bits * C / 4;   // bytes of y per row of x
    const size_t col_byte = (ds.bits == 16) ? 4 * cstart : (size_t)ds.bits * cstart / 4;
    const size_t r_stop = std::min(R_load, r_end);
    size_t gq = r_begin / gR2, gk = r_begin % gR2;   // gapped position of row r, kept incrementally
    for (size_t r = r_begin; r < r_stop; ++r) {
        const size_t gpos = gapped ? gq * gS + gk : r;
        if (++gk == gR2) { gk = 0; ++gq; }
        if (r + 8 < r_stop) _mm_prefetch((const char*)(ds.yb + (r + 8) * row_bytes + col_byte), _MM_HINT_T0);
        __m512d vr, vi;
        digits8(ds, r * C + cstart, vr, vi);
        _mm512_store_pd(g_re + 8 * gpos, vr);
        _mm512_store_pd(g_im + 8 * gpos, vi);
    }
}

// IBDWT digit state (ctx.ib_w: digits in [0, 2^b_j), ctx.ib_b: b_j | c_j << 7) -> run j0 .. j0+7 (j0 a multiple of 8)
// as local balanced digits times the weights a_j. h_prev = top bit of digit j0-1; *h_last (if given) = top bit of
// digit j0+7; top_last: digit j0+7 is the top digit (h = 0).
// ctx.ib_rc_live (normalization fused into B'): the run first gets its carry in (ib_rc of the run before, ib_cin0 for
// run 0, ib_madd at digit ib_m) and is normalized inside itself; what leaves lane 6 stays in lane 7 (unnormalized,
// rare: a carry of ~2^33 is absorbed in ~4 digits). The top bit h of lane 7 is always taken from the stored digit (as
// the next run reads it for its h_prev), so the balanced digits telescope to the same value.
// The balanced, weighted digits of run j0 from v (digits after the carry in: lanes 0..6 in [0, 2^b), lane 7 any),
// f (the stored digits), b (sizes) and c (weight carries a factor 2).
static inline __attribute__((always_inline)) __m512d ib_digits_w(const FFTContext& ctx, size_t j0, __m512i v, __m512i f, __m512i b,
                                                                 __mmask8 c, int64_t h_prev, int64_t* h_last, bool top_last) {
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i bm1 = _mm512_sub_epi64(b, one);
    __m512i hb = _mm512_and_si512(_mm512_srlv_epi64(v, bm1), one);
    hb = _mm512_mask_and_epi64(hb, 0x80, _mm512_srlv_epi64(f, bm1), one);   // lane 7: from the stored digit
    if (top_last) hb = _mm512_mask_mov_epi64(hb, 0x80, _mm512_setzero_si512());
    const __m512i hp = _mm512_alignr_epi64(hb, _mm512_set1_epi64(h_prev), 7);
    const __m512i d = _mm512_add_epi64(_mm512_sub_epi64(v, _mm512_sllv_epi64(hb, b)), hp);
    if (h_last) {
        alignas(64) int64_t hbs[8];
        _mm512_store_si512((void*)hbs, hb);
        *h_last = hbs[7];
    }
    const size_t h = j0 >> 6, l = j0 & 63;
    __m512d wv = _mm512_mul_pd(_mm512_set1_pd(ctx.ib_WH[h]), _mm512_loadu_pd(ctx.ib_WL.data() + l));
    wv = _mm512_mask_mul_pd(wv, c, wv, _mm512_set1_pd(2.0));
    if (j0 == 0) wv = _mm512_mask_mov_pd(wv, 0x01, _mm512_set1_pd(1.0));   // digit 0: a = 1
    return _mm512_mul_pd(_mm512_cvtepi64_pd(d), wv);
}

// Index of run i (digits 8i .. 8i+7) in ib_rc / ib_h7 / ib_bm - block-major: the runs of one block of 8
// columns of pass B' (rows K = 0 .. R-1) are consecutive, so a block reads them in whole cache lines:
// j = 8i = half*N + K*C + 8*cb -> half*N/8 + cb*R + K (R = n1, C = n2 = 2^ib_lgC, set by ibdwt_square). The top run
// L/8 - 1 keeps its index.
static inline size_t ib_ri(const FFTContext& ctx, size_t run) {
    const size_t j = run << 3;
    const size_t hh = (j >= ctx.fft_len) ? 1 : 0;
    const size_t p = j - hh * ctx.fft_len;
    const size_t K = p >> ctx.ib_lgC, cb = (p & ((size_t(1) << ctx.ib_lgC) - 1)) >> 3;
    return hh * (ctx.fft_len >> 3) + cb * ctx.ib_R + K;
}

// 8 runs j0[0..7] at once: stored digits f, sizes b, weight masks c, and v = f with the carries in applied (ctx.ib_rc_live):
// the runs as 8 lanes after an 8x8 transpose, 7 fixed carry steps (no data-dependent loop), the rest stays in digit 7.
// The same with the stored digits f[0..7] already loaded (ib_runs8 below loads them from ctx.ib_w; the overlaid
// layout of the giants loads them from re, batch8_gather_ib).
static inline __attribute__((always_inline)) void ib_runs8_core(const FFTContext& ctx, const uint8_t* bsz, const size_t* j0,
                                                                const __m512i* f, __m512i* v, __m512i* b, __mmask8* c) {
    for (int r = 0; r < 8; ++r) {
        const __m512i bc = _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(bsz + j0[r])));
        b[r] = _mm512_and_si512(bc, _mm512_set1_epi64(0x7F));
        c[r] = _mm512_test_epi64_mask(bc, _mm512_set1_epi64(0x80));
        v[r] = f[r];
    }
    if (!ctx.ib_rc_live) return;
    alignas(64) int64_t cin[8];
    const size_t mrun = ctx.ib_m >> 3;
    for (int r = 0; r < 8; ++r) {
        const size_t run = j0[r] >> 3;
        cin[r] = run ? ctx.ib_rc[ib_ri(ctx, run - 1)] : ctx.ib_cin0;
        if (ctx.ib_madd != 0 && run == mrun)
            v[r] = _mm512_mask_add_epi64(v[r], (__mmask8)(1u << (ctx.ib_m & 7)), v[r], _mm512_set1_epi64(ctx.ib_madd));
    }
    __m512d x[8], bt[8];
    for (int r = 0; r < 8; ++r) { x[r] = _mm512_castsi512_pd(v[r]); bt[r] = _mm512_castsi512_pd(b[r]); }
    st_transpose8x8(x);    // x[s] lane r = digit s of run r
    st_transpose8x8(bt);
    __m512i carry = _mm512_load_si512((const void*)cin);
    for (int s = 0; s < 7; ++s) {
        const __m512i bs = _mm512_castpd_si512(bt[s]);
        const __m512i vv = _mm512_add_epi64(_mm512_castpd_si512(x[s]), carry);
        carry = _mm512_srav_epi64(vv, bs);
        x[s] = _mm512_castsi512_pd(_mm512_sub_epi64(vv, _mm512_sllv_epi64(carry, bs)));
    }
    x[7] = _mm512_castsi512_pd(_mm512_add_epi64(_mm512_castpd_si512(x[7]), carry));
    st_transpose8x8(x);
    for (int r = 0; r < 8; ++r) v[r] = _mm512_castpd_si512(x[r]);
}
static inline __attribute__((always_inline)) void ib_runs8(const FFTContext& ctx, const int32_t* w, const uint8_t* bsz,
                                                           const size_t* j0, __m512i* v, __m512i* f, __m512i* b, __mmask8* c) {
    for (int r = 0; r < 8; ++r) f[r] = _mm512_cvtepi32_epi64(_mm256_loadu_si256((const __m256i*)(w + j0[r])));
    ib_runs8_core(ctx, bsz, j0, f, v, b, c);
}

// Single run j0 (the rows left over by the groups of 8 in batch8_gather_ib): the same as ib_runs8 + ib_digits_w,
// f = the stored digits.
static inline __m512d ib_state_dig8_f(const FFTContext& ctx, const uint8_t* bsz, __m512i f,
                                      size_t j0, int64_t h_prev, int64_t* h_last, bool top_last) {
    const __m512i bc = _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(bsz + j0)));
    const __m512i b = _mm512_and_si512(bc, _mm512_set1_epi64(0x7F));
    const __mmask8 c = _mm512_test_epi64_mask(bc, _mm512_set1_epi64(0x80));
    __m512i v = f;
    if (ctx.ib_rc_live) {
        const size_t run = j0 >> 3;
        const int64_t cin = run ? ctx.ib_rc[ib_ri(ctx, run - 1)] : ctx.ib_cin0;
        v = _mm512_mask_add_epi64(v, 0x01, v, _mm512_set1_epi64(cin));
        if (ctx.ib_madd != 0 && (ctx.ib_m >> 3) == run)
            v = _mm512_mask_add_epi64(v, (__mmask8)(1u << (ctx.ib_m & 7)), v, _mm512_set1_epi64(ctx.ib_madd));
        alignas(64) int64_t vs[8], bs[8];
        _mm512_store_si512((void*)vs, v);
        _mm512_store_si512((void*)bs, b);
        int64_t carry = 0;
        for (int s = 0; s < 7; ++s) {
            const int64_t vv = vs[s] + carry;
            carry = vv >> bs[s];   // arithmetic shift (floor)
            vs[s] = vv - carry * ((int64_t)1 << bs[s]);
        }
        vs[7] += carry;
        v = _mm512_load_si512((const void*)vs);
    }
    return ib_digits_w(ctx, j0, v, f, b, c, h_prev, h_last, top_last);
}
static inline __m512d ib_state_dig8(const FFTContext& ctx, const int32_t* w, const uint8_t* bsz,
                                    size_t j0, int64_t h_prev, int64_t* h_last, bool top_last) {
    const __m512i f = _mm512_cvtepi32_epi64(_mm256_loadu_si256((const __m256i*)(w + j0)));
    return ib_state_dig8_f(ctx, bsz, f, j0, h_prev, h_last, top_last);
}
// Overlaid digit state (ctx.ib_ovl): the 64 B of re at positions j0 .. j0+7 hold (lower digit j, upper digit N + j)
// as int32 pairs -> both runs as int64 lanes.
static inline __attribute__((always_inline)) void ib_ovl_load(const double* p, __m512i& f_lo, __m512i& f_hi) {
    const __m512i x = _mm512_loadu_si512((const void*)p);
    f_lo = _mm512_srai_epi64(_mm512_slli_epi64(x, 32), 32);
    f_hi = _mm512_srai_epi64(x, 32);
}
// Top bit of digit j of the IBDWT digit state (the top digit L-1: 0).
static inline int64_t ib_state_hbit(const int32_t* w, const uint8_t* bsz, size_t j, size_t L) {
    if (j + 1 >= L) return 0;
    return (w[j] >> ((bsz[j] & 0x7F) - 1)) & 1;
}
// Right-angle twist theta^j for j0 .. j0+7 (two-level tables TH * TL).
static inline __attribute__((always_inline)) void ib_twist8(const FFTContext& ctx, size_t j0, __m512d& tr, __m512d& ti) {
    const size_t h = j0 >> 6, l = j0 & 63;
    const __m512d Hr = _mm512_set1_pd(ctx.ib_THr[h]), Hi = _mm512_set1_pd(ctx.ib_THi[h]);
    const __m512d Lr = _mm512_loadu_pd(ctx.ib_TLr.data() + l), Li = _mm512_loadu_pd(ctx.ib_TLi.data() + l);
    tr = _mm512_fmsub_pd(Hr, Lr, _mm512_mul_pd(Hi, Li));
    ti = _mm512_fmadd_pd(Hr, Li, _mm512_mul_pd(Hi, Lr));
}

// IBDWT pass A with the pack fused in (no separate pack pass over the data): columns
// cstart..cstart+7 of the complex input x (R rows of C positions, element (r, c) = position j = r*C + c) computed
// straight from the digit state: x_j = (a_j d_j + i a_(j+N) d_(j+N)) * theta^j. Rows [r_begin, r_end); gapped as in
// batch8_gather.
// ctx.ib_ovl: the digits are read from xsrc (re, row stride xs: row r, column cstart) instead of ib_w, and the h_prev
// of each run from ib_h7 (the neighbouring column block may already hold FFT data).
static inline void batch8_gather_ib(const FFTContext& ctx, size_t R, size_t C, size_t cstart,
                                    double* __restrict g_re, double* __restrict g_im,
                                    size_t r_begin = 0, size_t r_end = ~(size_t)0, bool gapped = false,
                                    const double* xsrc = nullptr, size_t xs = 0) {
    if (r_end > R) r_end = R;
    const size_t N = ctx.fft_len, L = 2 * N;
    const int32_t* const w = ctx.ib_w.data();
    const uint8_t* const bsz = ctx.ib_b.data();
    if (ctx.ib_ovl && xsrc) {
        const uint8_t* const h7 = ctx.ib_h7.data();
        const size_t gR2o = std::max<size_t>(R >> 3, 1), gSo = b8_S(R);
        size_t gqo = r_begin / gR2o, gko = r_begin % gR2o;
        auto put_row_o = [&](size_t r, __m512d x0, __m512d x1) {
            const size_t gpos = gapped ? gqo * gSo + gko : r;
            if (++gko == gR2o) { gko = 0; ++gqo; }
            __m512d tr, ti;
            ib_twist8(ctx, r * C + cstart, tr, ti);
            _mm512_store_pd(g_re + 8 * gpos, _mm512_fmsub_pd(x0, tr, _mm512_mul_pd(x1, ti)));
            _mm512_store_pd(g_im + 8 * gpos, _mm512_fmadd_pd(x0, ti, _mm512_mul_pd(x1, tr)));
        };
        size_t r = r_begin;
        for (; r + 8 <= r_end; r += 8) {
            size_t jl[8], ju[8];
            __m512i fl[8], fh[8], v[8], b[8];
            __mmask8 c[8];
            __m512d x0[8], x1[8];
            for (int i = 0; i < 8; ++i) {
                jl[i] = (r + i) * C + cstart;
                ju[i] = N + jl[i];
                ib_ovl_load(xsrc + (r + i) * xs + cstart, fl[i], fh[i]);
            }
            ib_runs8_core(ctx, bsz, jl, fl, v, b, c);
            for (int i = 0; i < 8; ++i) {
                const int64_t hl = (jl[i] == 0) ? 0 : (int64_t)h7[ib_ri(ctx, (jl[i] >> 3) - 1)];
                x0[i] = ib_digits_w(ctx, jl[i], v[i], fl[i], b[i], c[i], hl, nullptr, false);
            }
            ib_runs8_core(ctx, bsz, ju, fh, v, b, c);
            for (int i = 0; i < 8; ++i)
                x1[i] = ib_digits_w(ctx, ju[i], v[i], fh[i], b[i], c[i], (int64_t)h7[ib_ri(ctx, (ju[i] >> 3) - 1)], nullptr, ju[i] + 8 == L);
            for (int i = 0; i < 8; ++i) put_row_o(r + (size_t)i, x0[i], x1[i]);
        }
        for (; r < r_end; ++r) {
            const size_t j0 = r * C + cstart;
            __m512i fl, fh;
            ib_ovl_load(xsrc + r * xs + cstart, fl, fh);
            const int64_t hl = (j0 == 0) ? 0 : (int64_t)h7[ib_ri(ctx, (j0 >> 3) - 1)];
            const int64_t hu = (int64_t)h7[ib_ri(ctx, ((N + j0) >> 3) - 1)];
            put_row_o(r, ib_state_dig8_f(ctx, bsz, fl, j0, hl, nullptr, false),
                      ib_state_dig8_f(ctx, bsz, fh, N + j0, hu, nullptr, N + j0 + 8 == L));
        }
        return;
    }
    const size_t gR2 = std::max<size_t>(R >> 3, 1), gS = b8_S(R);
    size_t gq = r_begin / gR2, gk = r_begin % gR2;
    auto put_row = [&](size_t r, __m512d x0, __m512d x1) {
        const size_t gpos = gapped ? gq * gS + gk : r;
        if (++gk == gR2) { gk = 0; ++gq; }
        __m512d tr, ti;
        ib_twist8(ctx, r * C + cstart, tr, ti);
        _mm512_store_pd(g_re + 8 * gpos, _mm512_fmsub_pd(x0, tr, _mm512_mul_pd(x1, ti)));
        _mm512_store_pd(g_im + 8 * gpos, _mm512_fmadd_pd(x0, ti, _mm512_mul_pd(x1, tr)));
    };
    size_t r = r_begin;
    for (; r + 8 <= r_end; r += 8) {   // groups of 8 rows: the carries in of their 8 runs at once (ib_runs8)
        if (r + 16 <= r_end) {   // rows are C*4 B apart (a power of two): prefetch into L2, as in batch8_gather
            for (int i = 0; i < 8; ++i) {
                _mm_prefetch((const char*)(w + (r + 8 + i) * C + cstart), _MM_HINT_T1);
                _mm_prefetch((const char*)(w + N + (r + 8 + i) * C + cstart), _MM_HINT_T1);
            }
        }
        size_t jl[8], ju[8];
        for (int i = 0; i < 8; ++i) { jl[i] = (r + i) * C + cstart; ju[i] = N + jl[i]; }
        __m512i v[8], f[8], b[8];
        __mmask8 c[8];
        __m512d x0[8], x1[8];
        ib_runs8(ctx, w, bsz, jl, v, f, b, c);
        for (int i = 0; i < 8; ++i) {
            const int64_t hl = (jl[i] == 0) ? 0 : ib_state_hbit(w, bsz, jl[i] - 1, L);
            x0[i] = ib_digits_w(ctx, jl[i], v[i], f[i], b[i], c[i], hl, nullptr, false);
        }
        ib_runs8(ctx, w, bsz, ju, v, f, b, c);
        for (int i = 0; i < 8; ++i) {
            const int64_t hu = ib_state_hbit(w, bsz, ju[i] - 1, L);
            x1[i] = ib_digits_w(ctx, ju[i], v[i], f[i], b[i], c[i], hu, nullptr, ju[i] + 8 == L);
        }
        for (int i = 0; i < 8; ++i) put_row(r + (size_t)i, x0[i], x1[i]);
    }
    for (; r < r_end; ++r) {   // rows left over (a range of a thread group in the coop path)
        const size_t j0 = r * C + cstart;
        const int64_t hl = (j0 == 0) ? 0 : ib_state_hbit(w, bsz, j0 - 1, L);
        const int64_t hu = ib_state_hbit(w, bsz, N + j0 - 1, L);
        const __m512d x0 = ib_state_dig8(ctx, w, bsz, j0, hl, nullptr, false);
        const __m512d x1 = ib_state_dig8(ctx, w, bsz, N + j0, hu, nullptr, N + j0 + 8 == L);
        put_row(r, x0, x1);
    }
}

// IBDWT unweight (k / a_j) and round, negated (-W == -k*Y^2), of 8 values of digits jj .. jj+7 -> int64 lanes.
static inline __attribute__((always_inline)) __m512i ib_round8(const FFTContext& ctx, size_t jj, __m512d cval, __m512d& vmax) {
    const uint8_t* const bsz = ctx.ib_b.data();
    const double kd = (double)ctx.ib_k;
    const __m512i bc = _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(bsz + jj)));
    const __mmask8 c = _mm512_test_epi64_mask(bc, _mm512_set1_epi64(0x80));
    const size_t hh = jj >> 6, ll = jj & 63;
    __m512d kia = _mm512_mul_pd(_mm512_set1_pd(kd * ctx.ib_iWH[hh]), _mm512_loadu_pd(ctx.ib_iWL.data() + ll));
    kia = _mm512_mask_mul_pd(kia, c, kia, _mm512_set1_pd(0.5));
    if (jj == 0) kia = _mm512_mask_mov_pd(kia, 0x01, _mm512_set1_pd(kd));   // digit 0: a = 1
    const __m512d q = _mm512_mul_pd(cval, kia);
    const __m512d r = _mm512_roundscale_pd(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    vmax = _mm512_max_pd(vmax, _mm512_abs_pd(_mm512_sub_pd(q, r)));
    return _mm512_sub_epi64(_mm512_setzero_si512(), _mm512_cvtpd_epi64(r));
}

// IBDWT normalization fused into pass B': rows K0 .. K0+7 of the block of columns c0 .. c0+7 (complex
// positions j = K*C + c0 + lane; vector K at pos(K) = two ? posL(K, S) : K in v_re/v_im) -> untwist, unweight, round,
// carries inside each run of 8 digits (the 8 rows at once after an 8x8 transpose), int32 digits into ctx.ib_w and
// the carry out of each run into ctx.ib_rc (lower digits j, upper digits N + j). Replaces the store and the unpack.
// ovl_re != nullptr (overlaid digit state): the digits of position j go as an int32 pair (lower, upper) into the 8 bytes
// of ovl_re at row K, column c0 + lane (row stride dp) - the positions this block of B' has just read - and the top bit
// of digit 7 of every run into ctx.ib_h7.
static inline void ib_norm_rows8(FFTContext& ctx, const double* v_re, const double* v_im, bool two, size_t S,
                                 size_t K0, size_t c0, size_t C, __m512d& vmax, double* ovl_re = nullptr, size_t dp = 0) {
    const size_t N = ctx.fft_len;
    const uint8_t* const bsz = ctx.ib_b.data();
    int32_t* const w = ctx.ib_w.data();
    int64_t* const rc = ctx.ib_rc.data();
    uint8_t* const h7 = ctx.ib_h7.data();
    const __m512i one = _mm512_set1_epi64(1);
    __m512d xl[8], xh[8];
    for (int r = 0; r < 8; ++r) {
        const size_t K = K0 + (size_t)r;
        const size_t pos = two ? batch8_posL(K, S) : K;
        const __m512d xr = _mm512_load_pd(v_re + 8 * pos), xi = _mm512_load_pd(v_im + 8 * pos);
        const size_t j0 = K * C + c0;
        __m512d tr, ti;
        ib_twist8(ctx, j0, tr, ti);
        // (xr + i xi) * conj(theta^j)
        const __m512d cr = _mm512_fmadd_pd(xr, tr, _mm512_mul_pd(xi, ti));
        const __m512d ci = _mm512_fmsub_pd(xi, tr, _mm512_mul_pd(xr, ti));
        xl[r] = _mm512_castsi512_pd(ib_round8(ctx, j0, cr, vmax));
        xh[r] = _mm512_castsi512_pd(ib_round8(ctx, N + j0, ci, vmax));
    }
    auto chain = [&](__m512d* x, size_t off) {   // off = 0 (lower digits) or N (upper)
        __m512d bb[8];
        for (int r = 0; r < 8; ++r) {
            const size_t j = off + (K0 + (size_t)r) * C + c0;
            bb[r] = _mm512_castsi512_pd(_mm512_and_si512(_mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(bsz + j))),
                                                         _mm512_set1_epi64(0x7F)));
        }
        st_transpose8x8(x);    // x[s] lane r = digit s of the run of row K0 + r
        st_transpose8x8(bb);
        __m512i carry = _mm512_setzero_si512();
        for (int s = 0; s < 8; ++s) {
            const __m512i b = _mm512_castpd_si512(bb[s]);
            const __m512i vv = _mm512_add_epi64(_mm512_castpd_si512(x[s]), carry);
            carry = _mm512_srav_epi64(vv, b);
            x[s] = _mm512_castsi512_pd(_mm512_and_si512(vv, _mm512_sub_epi64(_mm512_sllv_epi64(one, b), one)));
        }
        alignas(64) int64_t hb7[8];
        if (ovl_re) {   // top bit of digit 7 of each run (lane r = run of row K0 + r)
            const __m512i b7 = _mm512_castpd_si512(bb[7]);
            _mm512_store_si512((void*)hb7, _mm512_and_si512(_mm512_srlv_epi64(_mm512_castpd_si512(x[7]), _mm512_sub_epi64(b7, one)), one));
        }
        st_transpose8x8(x);
        alignas(64) int64_t co[8];
        _mm512_store_si512((void*)co, carry);
        const size_t ri0 = (off ? (N >> 3) : 0) + (c0 >> 3) * ctx.ib_R + K0;   // ib_ri of the run of row K0
        for (int r = 0; r < 8; ++r) {
            const size_t j = off + (K0 + (size_t)r) * C + c0;
            if (ovl_re) h7[ri0 + r] = (uint8_t)hb7[r];
            else        _mm256_storeu_si256((__m256i*)(w + j), _mm512_cvtepi64_epi32(_mm512_castpd_si512(x[r])));
            rc[ri0 + r] = co[r];
        }
    };
    chain(xl, 0);
    chain(xh, N);
    if (ovl_re) {
        const __m512i lo32 = _mm512_set1_epi64(0xFFFFFFFFLL);
        for (int r = 0; r < 8; ++r) {
            const __m512i pair = _mm512_or_si512(_mm512_slli_epi64(_mm512_castpd_si512(xh[r]), 32),
                                                 _mm512_and_si512(_mm512_castpd_si512(xl[r]), lo32));
            _mm512_storeu_si512((void*)(ovl_re + (K0 + (size_t)r) * dp + c0), pair);
        }
    }
}

// Store the 8 result columns as rows: dst[cols[b]*R + k] (8x8 transpose in registers).
// L = true: result in the blocked layout of fft_batch8_2l (vector k+t is at position t*R/8 + k/8).
// G = true: result in the gapped natural layout of the in-place form (vector k+t at b8_gpos(k+t)).
// tw_re/im (natural order, 8*k): vector k is multiplied by conj(tw[k]) before the transpose (the inverse twiddle of
// the four-step FFT, saved by the gather of the same block: batch8_gather tw_out).
static inline void batch8_store_rows(const double* __restrict v_re, const double* __restrict v_im, size_t R, const size_t* cols,
                                     double* __restrict dst_re, double* __restrict dst_im, bool L = false, size_t dp = 0,
                                     size_t k_begin = 0, size_t k_end = ~(size_t)0, bool G = false,
                                     const double* __restrict tw_re = nullptr, const double* __restrict tw_im = nullptr) {
    const size_t S = b8_S(R);           // block stride of the two-level layout (L = true)
    const size_t R2 = R >> 3;
    const size_t pitch = dp ? dp : R;   // destination row stride (padded or R)
    if (k_end > R) k_end = R;           // k in [k_begin, k_end), multiples of 8 (a block shared by several threads)
    for (size_t k = k_begin; k < k_end; k += 8) {
        __m512d xr[8], xi[8];
        const size_t gbase = G ? b8_gpos(k, R2, S) : 0;   // k .. k+7 lie in one block (R2 is a multiple of 8)
#pragma GCC unroll 8
        for (int t = 0; t < 8; ++t) {
            const size_t pos = L ? ((size_t)t * S + (k >> 3)) : (G ? gbase + t : k + t);
            xr[t] = _mm512_load_pd(v_re + 8 * pos);
            xi[t] = _mm512_load_pd(v_im + 8 * pos);
        }
        if (tw_re) {
#pragma GCC unroll 8
            for (int t = 0; t < 8; ++t) {
                // (x_r + i x_i) * (w_r - i w_i)
                const __m512d wr = _mm512_loadu_pd(tw_re + 8 * (k + t)), wi = _mm512_loadu_pd(tw_im + 8 * (k + t));
                const __m512d yr = _mm512_fmadd_pd(xr[t], wr, _mm512_mul_pd(xi[t], wi));
                const __m512d yi = _mm512_fmsub_pd(xi[t], wr, _mm512_mul_pd(xr[t], wi));
                xr[t] = yr; xi[t] = yi;
            }
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

// Column pass (batch-8): same result as bailey_colfft_pass, without a transpose on load
// and (in TRANSP mode) on store. w1r/w1i: R-th roots of unity (r_w1* or c_w1*).
// rows_nz > 0: only source rows 0 .. rows_nz-1 are nonzero, the rest is not read (may hold garbage). With
// rows_nz == R/2 and the two-level FFT, the first radix-8 stage also skips the zero half (fft_batch8_2l_t<true>).
// src and dst may be the same buffer with transposed_out (in-place layout of the IBDWT): every block (or coop group)
// gathers its columns completely before it stores them back to the same positions.

static void bailey_colfft_pass8(const double* src_re, const double* src_im,
    double* dst_re, double* dst_im,
    size_t R, size_t C, bool twiddle, bool transposed_out, FFTContext& ctx, bool inverse,
    const std::vector<double>& w1r, const std::vector<double>& w1i, size_t sp = 0, size_t dp = 0, size_t rows_nz = 0,
    int prof_base = -1, const DigitSrc* dsrc = nullptr, const FFTContext* ibsrc = nullptr, double* ib_err = nullptr) {
    // ib_err != nullptr (IBDWT pass B', transposed_out): the result is not stored; ib_norm_rows8 normalizes it into the
    // digit state (ctx.ib_w / ib_rc) and ib_err[thread] gets the max rounding error.
    (void)prof_base;
    const Tw2 tw2(ctx);
    const bool two = batch8_two_level(R);
    const size_t R2 = R >> 3;
    if (rows_nz == 0 || rows_nz > R) rows_nz = R;
    const bool half = two && (rows_nz == R / 2);   // the first stage reads only q < 4 (rows < R/2)
    // row strides: source (R rows of C columns) and destination (TRANSP: R rows of C; ROWS: C rows of R); 0 = no padding
    if (sp == 0) sp = C;
    if (dp == 0) dp = transposed_out ? C : R;
    SubAcc sub[64];
    int sub_nt = 1;
    // shared buffers of the leftover blocks computed by groups of threads (48R doubles per block, see below)
    const size_t BL = b8_len(R);   // doubles of one re/im array of a block (two-level layout with block gaps)
    const size_t S = b8_S(R);
    const bool ip = b8_inplace(R);   // in-place two-level form (gapped input, see b8_inplace)
    double* const coop_buf = two ? coop_scratch((size_t)omp_get_max_threads() * 6 * BL) : nullptr;
#pragma omp parallel
    {
        double* const scr = batch8_scratch(R, 1, 0);
        double* const a_re = scr;          double* const a_im = scr + BL;
        double* const b_re = scr + 2 * BL; double* const b_im = scr + 3 * BL;
        double* const c_re = scr + 4 * BL; double* const c_im = scr + 5 * BL;
        const int tid = omp_get_thread_num();
        if (tid == 0) sub_nt = omp_get_num_threads();
        SubAcc& acc = sub[tid < 64 ? tid : 63];
        (void)acc;
        __m512d ib_vmax = _mm512_setzero_pd();   // ib_err: this thread's max rounding error
        // one whole block of 8 columns c0 .. c0+7 (gather, FFT, store) by this thread
        auto do_block = [&](size_t c0) {
            SUB_T(t0);
            if (ibsrc)     batch8_gather_ib(*ibsrc, R, C, c0, a_re, a_im, 0, R, ip, src_re, sp);   // IBDWT pack fused into pass A
            else if (dsrc) batch8_gather_digits(*dsrc, R, C, c0, a_re, a_im, rows_nz, !half, 0, ~(size_t)0, ip);   // pack fused into pass A
            else {
                batch8_gather(src_re, src_im, R, sp, c0, false, 0, twiddle, inverse, ctx, tw2, a_re, a_im, rows_nz, !half,
                              0, ~(size_t)0, ip);
            }
            SUB_T(t1);
            SUB_ADD(acc.c[0], t0, t1);
            const double* res_re;
            const double* res_im;
            uint64_t* const st1 =
#ifdef LOTR_PROFILE
                &acc.c[1];
#else
                nullptr;
#endif
            if (ip) {
                // in place: stage 1 on a, the sub-FFTs through c, spectrum back in a (blocked layout)
                if (inverse) {
                    if (half) b8_ip_stage1<true, true>(a_re, a_im, R, w1r.data(), w1i.data(), 0, R2);
                    else      b8_ip_stage1<false, true>(a_re, a_im, R, w1r.data(), w1i.data(), 0, R2);
                }
                else {
                    if (half) b8_ip_stage1<true, false>(a_re, a_im, R, w1r.data(), w1i.data(), 0, R2);
                    else      b8_ip_stage1<false, false>(a_re, a_im, R, w1r.data(), w1i.data(), 0, R2);
                }
#ifdef LOTR_PROFILE
                *st1 += __rdtsc() - t1;
#endif
                b8_ip_subs(a_re, a_im, c_re, c_im, R, w1r.data(), w1i.data(), inverse, 0, 8);
                res_re = a_re; res_im = a_im;
            }
            else if (two) {
                if (half) fft_batch8_2l_t<true>(a_re, a_im, false, b_re, b_im, c_re, c_im, R, w1r.data(), w1i.data(), inverse, st1);
                else      fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, w1r.data(), w1i.data(), inverse, st1);
                res_re = b_re; res_im = b_im;
            }
            else {
                (void)st1;
                const bool in_b = fft_batch8(a_re, a_im, b_re, b_im, R, w1r.data(), w1i.data(), inverse);
                res_re = in_b ? b_re : a_re;
                res_im = in_b ? b_im : a_im;
            }
            SUB_T(t2);
            SUB_ADD(acc.c[2], t1, t2);
            if (transposed_out && ib_err) {
                for (size_t K0 = 0; K0 < R; K0 += 8)
                    ib_norm_rows8(ctx, res_re, res_im, two, S, K0, c0, C, ib_vmax, ctx.ib_ovl_write ? dst_re : nullptr, dp);
            }
            else if (transposed_out) {
                if (two) {
                    // position p*S + k' holds X[p + 8k']: rows K written in ascending order (one sweep through the
                    // destination instead of 8 interleaved ones), reading position (K & 7)*S + (K >> 3) from the block
                    for (size_t K = 0; K < R; ++K) {
                        const size_t pos = batch8_posL(K, S);
                        _mm512_storeu_pd(dst_re + K * dp + c0, _mm512_load_pd(res_re + 8 * pos));
                        _mm512_storeu_pd(dst_im + K * dp + c0, _mm512_load_pd(res_im + 8 * pos));
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
            SUB_T(t3);
            SUB_ADD(acc.c[3], t2, t3);
        };

        // Blocks over the threads. With the two-level FFT and NB % nt != 0 (e.g. 32 blocks on 6 threads: 6,6,5,5,5,5)
        // every thread takes NB / nt whole blocks and the rem leftover blocks are computed by groups of threads
        // (thread t -> group t % rem): gather rows, stage 1 (k), the 8 sub-FFTs and the store are split inside the
        // group, with a barrier between the steps (shared buffers in coop).
        const int nt = omp_get_num_threads();
        const size_t NB = C / 8;
        const size_t q = NB / (size_t)nt, rem = NB % (size_t)nt;
        const bool use_coop = two && coop_buf != nullptr && nt > 1 && rem != 0 && 2 * rem <= (size_t)nt;
        if (!use_coop) {
#pragma omp for schedule(static)
            for (size_t cb = 0; cb < NB; ++cb) do_block(cb * 8);
        }
        else {
            for (size_t cb = (size_t)tid * q; cb < (size_t)(tid + 1) * q; ++cb) do_block(cb * 8);

            const size_t g = (size_t)tid % rem, u = (size_t)tid / rem;
            const size_t gs = ((size_t)nt - g + rem - 1) / rem;   // threads in group g
            const size_t c0 = (q * (size_t)nt + g) * 8;
            double* const in_re = coop_buf + g * 6 * BL;  double* const in_im = in_re + BL;
            double* const out_re = in_re + 2 * BL;    double* const out_im = in_re + 3 * BL;
            double* const tmp_re = in_re + 4 * BL;    double* const tmp_im = in_re + 5 * BL;
            const bool s1_tmp = batch8_2l_s1_in_tmp(R);
            double* const s1_re = ip ? in_re : (s1_tmp ? tmp_re : out_re);
            double* const s1_im = ip ? in_im : (s1_tmp ? tmp_im : out_im);
            double* const s2_re = s1_tmp ? out_re : tmp_re; double* const s2_im = s1_tmp ? out_im : tmp_im;
            // spectrum (blocked layout): in place -> in, else out
            const double* const sp_re = ip ? in_re : out_re;
            const double* const sp_im = ip ? in_im : out_im;

            // a) gather: rows [rb, re) of the rows that are read (half: only the nonzero half)
            SUB_T(c0t);
            {
                const size_t R_used = half ? rows_nz : R;
                const size_t rb = R_used * u / gs, re = R_used * (u + 1) / gs;
                if (ibsrc)     batch8_gather_ib(*ibsrc, R, C, c0, in_re, in_im, rb, re, ip, src_re, sp);
                else if (dsrc) batch8_gather_digits(*dsrc, R, C, c0, in_re, in_im, rows_nz, !half, rb, re, ip);
                else           batch8_gather(src_re, src_im, R, sp, c0, false, 0, twiddle, inverse, ctx, tw2, in_re, in_im, rows_nz, !half,
                                        rb, re, ip);
            }
            SUB_T(c1t);
            SUB_ADD(acc.c[0], c0t, c1t);
#pragma omp barrier
            // b) stage 1 for k in [kb, ke) (in place: each k reads and writes only its own positions q*S + k)
            SUB_T(c2t);
            {
                const size_t kb = R2 * u / gs, ke = R2 * (u + 1) / gs;
                if (ip) {
                    if (inverse) {
                        if (half) b8_ip_stage1<true, true>(in_re, in_im, R, w1r.data(), w1i.data(), kb, ke);
                        else      b8_ip_stage1<false, true>(in_re, in_im, R, w1r.data(), w1i.data(), kb, ke);
                    }
                    else {
                        if (half) b8_ip_stage1<true, false>(in_re, in_im, R, w1r.data(), w1i.data(), kb, ke);
                        else      b8_ip_stage1<false, false>(in_re, in_im, R, w1r.data(), w1i.data(), kb, ke);
                    }
                }
                else if (half) batch8_2l_stage1<true>(in_re, in_im, false, s1_re, s1_im, R, w1r.data(), w1i.data(), inverse, kb, ke);
                else           batch8_2l_stage1<false>(in_re, in_im, false, s1_re, s1_im, R, w1r.data(), w1i.data(), inverse, kb, ke);
            }
            SUB_T(c3t);
            SUB_ADD(acc.c[1], c2t, c3t);
            SUB_ADD(acc.c[2], c2t, c3t);   // c[2] holds the whole FFT until the final "c[2] -= c[1]" below
#pragma omp barrier
            // c) sub-FFTs p = u, u + gs, ... (result in out; in place: back in in, through this thread's own scratch c)
            SUB_T(c4t);
            if (ip) {
                for (size_t p = u; p < 8; p += gs) b8_ip_subs(in_re, in_im, c_re, c_im, R, w1r.data(), w1i.data(), inverse, p, p + 1);
            }
            else {
                for (size_t p = u; p < 8; p += gs) {
                    fft_batch8(s1_re + 8 * p * S, s1_im + 8 * p * S, s2_re + 8 * p * S, s2_im + 8 * p * S,
                               R2, w1r.data(), w1i.data(), inverse, 8);
                }
            }
            SUB_T(c5t);
            SUB_ADD(acc.c[2], c4t, c5t);
#pragma omp barrier
            // d) store of rows K (transposed) or of k-chunks of 8 (rows of the destination) in [sb, se)
            SUB_T(c6t);
            if (transposed_out && ib_err) {
                const size_t sb = 8 * ((R / 8) * u / gs), se = 8 * ((R / 8) * (u + 1) / gs);   // whole groups of 8 rows
                for (size_t K0 = sb; K0 < se; K0 += 8)
                    ib_norm_rows8(ctx, sp_re, sp_im, true, S, K0, c0, C, ib_vmax, ctx.ib_ovl_write ? dst_re : nullptr, dp);
            }
            else if (transposed_out) {
                const size_t sb = R * u / gs, se = R * (u + 1) / gs;
                for (size_t K = sb; K < se; ++K) {
                    const size_t pos = batch8_posL(K, S);
                    _mm512_storeu_pd(dst_re + K * dp + c0, _mm512_load_pd(sp_re + 8 * pos));
                    _mm512_storeu_pd(dst_im + K * dp + c0, _mm512_load_pd(sp_im + 8 * pos));
                }
            }
            else {
                size_t cols[8];
                for (size_t b = 0; b < 8; ++b) cols[b] = c0 + b;
                const size_t sb = 8 * (R2 * u / gs), se = 8 * (R2 * (u + 1) / gs);
                batch8_store_rows(sp_re, sp_im, R, cols, dst_re, dst_im, true, dp, sb, se);
            }
            SUB_T(c7t);
            SUB_ADD(acc.c[3], c6t, c7t);
        }
#ifdef LOTR_PROFILE
        acc.c[2] -= acc.c[1];   // "rest of the FFT" without stage 1
#endif
        if (ib_err) ib_err[tid < 64 ? tid : 63] = _mm512_reduce_max_pd(ib_vmax);
    }
#ifdef LOTR_PROFILE
    sub_merge(ctx, prof_base, 4, sub, sub_nt);
#endif
}

// IBDWT top carry wrap (k*2^n+1): the carry C out of the top run, C*2^(e_L) = C*2^d*2^n == -q + r*2^n with
// C*2^d = q*k + r (0 <= r < k) -> ib_cin0 = -q (into run 0), ib_madd = r << msh (into digit ib_m), for the next pack.
static void ib_top_wrap(FFTContext& ctx) {
    const size_t L = 2 * ctx.fft_len;
    const __int128 C = ctx.ib_rc[L / 8 - 1];
    const __int128 kk = (__int128)ctx.ib_k;
    ctx.ib_cin0 = 0;
    ctx.ib_madd = 0;
    if (C != 0) {
        const __int128 tt = C * ((__int128)1 << ctx.ib_d);
        __int128 qq = tt / kk;
        __int128 rr = tt % kk;
        if (rr < 0) { rr += kk; qq -= 1; }
        ctx.ib_cin0 = (int64_t)(-qq);
        ctx.ib_madd = (int64_t)(rr << ctx.ib_msh);
    }
}

// Size masks of 8 consecutive runs (ctx.ib_bm, ib_ri order) -> big (byte r = the digits of q0 + 1 bits of run r) and
// cw (byte r = the digits whose weight carries a factor 2): low and high bytes of the 8 uint16 entries.
static inline void ib_masks8(const uint16_t* bm, uint64_t& big, uint64_t& cw) {
    const __m128i v = _mm_loadu_si128((const __m128i*)bm);
    const __m128i s = _mm_shuffle_epi8(v, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
    big = (uint64_t)_mm_cvtsi128_si64(s);
    cw = (uint64_t)_mm_extract_epi64(s, 1);
}

// 8x8 bit matrix transpose (Hacker's Delight): bit 8r + c -> bit 8c + r.
static inline uint64_t bits_transpose8(uint64_t x) {
    uint64_t t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;  x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL; x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL; x = x ^ t ^ (t << 28);
    return x;
}

// The normalization of B' and the pack of the next pass A in one step (ibdwt_pass_BA; for a block whose left neighbour
// c0 - 8 is already done by this thread): rows K0 .. K0+7 of the block c0 (B' result v_re/v_im as in ib_norm_rows8) ->
// untwist, unweight, round; per half one transposed carry chain over the 8 digits of each run that STARTS with the carry
// out of the run to the left (ib_rc), so the digits are fully normalized; the local balancing in the transposed form
// (digit s-1 = the previous vector, h_prev of digit 0 = ib_h7 of the run to the left), back to rows, weights, the same
// twist theta^j -> rows K0 .. K0+7 of pass A's gather buffer g (gapped as batch8_gather). The carry out and the top bit
// of digit 7 of every run go to ib_rc / ib_h7 for the block to the right; the digits themselves are not stored.
// The normalization in floating point: per digit q = round(t * 2^-b), d = t - q * 2^b gives the balanced digit d in
// [-2^(b-1), 2^(b-1)] and the carry q directly - no top bits h, no variable shifts, no int64 <-> double conversions of
// the digits (all values are integers < 2^53, so every step is exact). The digit sizes are q0 or q0 + 1 bits
// (q0 = floor(beta)), the "q0 + 1" bits as an 8x8 bit matrix transposed in a scalar register (mask per digit s, lanes =
// runs). The runs this writes lend no top bit to the right (ib_h7 = 0); the run to the left may (an old-path block: its
// ib_h7 is added to the carry in).
// All per-run / per-row data in the block-major tables (ib_ri order: rows K0 .. K0+7 of the block consecutive):
// ib_bm (size masks, 2 bytes per run), ib_rc / ib_h7, ib_wT (WH, iWH lower, TH, WH, iWH upper) - whole cache lines.
// Where the run carries of one block of ib_norm_pack_rows8 come from and go (per half, pointers at row 0 of the block):
// rcL/h7L = the left block's carries / top bits (h7L nullptr: all 0), rcO = own carries, rcG/h7G = also the global
// tables (nullptr: not needed - no deferred block reads them).
struct IbCarryIO {
    const int64_t* rcL[2]; const uint8_t* h7L[2];
    int64_t* rcO[2]; int64_t* rcG[2]; uint8_t* h7G[2];
};
// out_blocked (one buffer in ibdwt_pass_BA): row K goes to its own blocked position batch8_posL(K, S) -
// the very positions the 8 rows were read from (g may be v: all 8 rows are loaded before anything is stored), so the
// input of pass A lands in the blocked layout of the B' spectrum (pass A then runs DIT, b8_ip_ditf).
static inline void ib_norm_pack_rows8(FFTContext& ctx, const double* v_re, const double* v_im, bool two, size_t S,
                                      size_t K0, size_t c0, size_t C, size_t R, __m512d& vmax,
                                      double* g_re, double* g_im, bool gapped,
                                      const IbCarryIO* io = nullptr, bool out_blocked = false) {
    const size_t N = ctx.fft_len;
    const size_t NR = N >> 3;                        // runs per half
    const size_t ri0 = (c0 >> 3) * R + K0;           // ib_ri of the lower run of row K0
    int64_t* const rc = ctx.ib_rc.data();
    uint8_t* const h7 = ctx.ib_h7.data();
    const size_t M = (C >> 6) * R;                   // ib_wT: 6 arrays of M, index (c0 / 64) * R + K
    const double* const wT = ctx.ib_wT.data() + (c0 >> 6) * R + K0;
    const int q0 = (int)ctx.ib_IL[1];   // floor(beta)
    const __m512d B0 = _mm512_set1_pd(std::ldexp(1.0, q0)), B1 = _mm512_set1_pd(std::ldexp(1.0, q0 + 1));
    const __m512d I0 = _mm512_set1_pd(std::ldexp(1.0, -q0)), I1 = _mm512_set1_pd(std::ldexp(1.0, -q0 - 1));
    const double nkd = -(double)ctx.ib_k;   // negated: -k/a_j, the result is -k*Y^2
    // size masks of the 8 runs (ib_bm), byte r of the masks = row K0 + r
    uint64_t bgl, cml, bgh, cmh;
    ib_masks8(ctx.ib_bm.data() + ri0, bgl, cml);
    ib_masks8(ctx.ib_bm.data() + NR + ri0, bgh, cmh);
    // the low parts of weights and twist: l = c0 & 63 for every row and both halves (N is a multiple of 64)
    const size_t l = c0 & 63;
    const __m512d WL = _mm512_loadu_pd(ctx.ib_WL.data() + l), iWL = _mm512_loadu_pd(ctx.ib_iWL.data() + l);
    const __m512d TLr = _mm512_loadu_pd(ctx.ib_TLr.data() + l), TLi = _mm512_loadu_pd(ctx.ib_TLi.data() + l);
    // -k/a_j * value, rounded (vmax: rounding error); iwh = iWH of the row, weight carries a factor 2 where c
    auto round8 = [&](double iwh, __m512d cval, __mmask8 c) -> __m512d {
        __m512d kia = _mm512_mul_pd(_mm512_set1_pd(nkd * iwh), iWL);
        kia = _mm512_mask_mul_pd(kia, c, kia, _mm512_set1_pd(0.5));
        const __m512d q = _mm512_mul_pd(cval, kia);
        const __m512d rr = _mm512_roundscale_pd(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        vmax = _mm512_max_pd(vmax, _mm512_abs_pd(_mm512_sub_pd(q, rr)));
        return rr;
    };
    __m512d tr[8], ti[8], xl[8], xh[8];
    for (int r = 0; r < 8; ++r) {
        const size_t K = K0 + (size_t)r;
        const size_t pos = two ? batch8_posL(K, S) : K;
        const __m512d xr = _mm512_load_pd(v_re + 8 * pos), xi = _mm512_load_pd(v_im + 8 * pos);
        // theta^j = TH * TL
        const __m512d Hr = _mm512_set1_pd(wT[2 * M + r]), Hi = _mm512_set1_pd(wT[3 * M + r]);
        tr[r] = _mm512_fmsub_pd(Hr, TLr, _mm512_mul_pd(Hi, TLi));
        ti[r] = _mm512_fmadd_pd(Hr, TLi, _mm512_mul_pd(Hi, TLr));
        // (xr + i xi) * conj(theta^j)
        const __m512d cr = _mm512_fmadd_pd(xr, tr[r], _mm512_mul_pd(xi, ti[r]));
        const __m512d ci = _mm512_fmsub_pd(xi, tr[r], _mm512_mul_pd(xr, ti[r]));
        xl[r] = round8(wT[1 * M + r], cr, (__mmask8)(cml >> (8 * r)));
        xh[r] = round8(wT[5 * M + r], ci, (__mmask8)(cmh >> (8 * r)));
    }
    // Carries of both halves (hh = 0: lower digits, ri = ri0; hh = 1: upper, ri = NR + ri0).
    // carry in = carry out of the run to the left (block c0 - 8, same rows; block 0 never takes this path)
    // + its top bit (0 unless the left block took the old path). io (ibdwt_pass_BA): the left carries from the
    // thread's local buffer or the global table, own carries into the local buffer (+ global where a deferred
    // block will read them); without io: the global tables ib_rc / ib_h7 as before.
    auto carry_in = [&](size_t ri, int hh) -> __m512d {
        const int64_t* const rcL = io ? io->rcL[hh] + K0 : rc + ri - R;
        const uint8_t* const h7L = io ? (io->h7L[hh] ? io->h7L[hh] + K0 : nullptr) : h7 + ri - R;
        const __m512i hl = h7L ? _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)h7L)) : _mm512_setzero_si512();
        return _mm512_cvtepi64_pd(_mm512_add_epi64(_mm512_loadu_si512((const void*)rcL), hl));
    };
    auto carry_out = [&](__m512d carry, size_t ri, int hh) {
        const __m512i co = _mm512_cvtpd_epi64(carry);
        if (io) {
            _mm512_storeu_si512((void*)(io->rcO[hh] + K0), co);
            if (io->rcG[hh]) {
                _mm512_storeu_si512((void*)(io->rcG[hh] + K0), co);
                _mm_storel_epi64((__m128i*)(io->h7G[hh] + K0), _mm_setzero_si128());
            }
        }
        else {
            _mm512_storeu_si512((void*)(rc + ri), co);
            _mm_storel_epi64((__m128i*)(h7 + ri), _mm_setzero_si128());
        }
    };
    // The two halves' carry chains interleaved: each step depends on the previous one (add, mul, round: ~10 cycles), the
    // halves do not - run one after the other, the core would mostly wait.
    __m512d cl = carry_in(ri0, 0), ch = carry_in(NR + ri0, 1);
    st_transpose8x8(xl);    // x[s] lane r = digit s of the run of row K0 + r
    st_transpose8x8(xh);
    const uint64_t btl = bits_transpose8(bgl), bth = bits_transpose8(bgh);   // byte s: the runs whose digit s has q0+1 bits
    for (int s = 0; s < 8; ++s) {
        const __mmask8 ml = (__mmask8)(btl >> (8 * s)), mh = (__mmask8)(bth >> (8 * s));
        const __m512d tl = _mm512_add_pd(xl[s], cl), th = _mm512_add_pd(xh[s], ch);
        const __m512d ql = _mm512_roundscale_pd(_mm512_mul_pd(tl, _mm512_mask_blend_pd(ml, I0, I1)),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m512d qh = _mm512_roundscale_pd(_mm512_mul_pd(th, _mm512_mask_blend_pd(mh, I0, I1)),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        xl[s] = _mm512_fnmadd_pd(ql, _mm512_mask_blend_pd(ml, B0, B1), tl);
        xh[s] = _mm512_fnmadd_pd(qh, _mm512_mask_blend_pd(mh, B0, B1), th);
        cl = ql; ch = qh;
    }
    carry_out(cl, ri0, 0);
    carry_out(ch, NR + ri0, 1);
    st_transpose8x8(xl);    // back: x[r] lane i = balanced digit i of the run of row K0 + r
    st_transpose8x8(xh);
    for (int r = 0; r < 8; ++r) {
        __m512d wl = _mm512_mul_pd(_mm512_set1_pd(wT[r]), WL);           // WH (lower) * WL
        __m512d wh = _mm512_mul_pd(_mm512_set1_pd(wT[4 * M + r]), WL);   // WH (upper) * WL
        wl = _mm512_mask_mul_pd(wl, (__mmask8)(cml >> (8 * r)), wl, _mm512_set1_pd(2.0));
        wh = _mm512_mask_mul_pd(wh, (__mmask8)(cmh >> (8 * r)), wh, _mm512_set1_pd(2.0));
        xl[r] = _mm512_mul_pd(xl[r], wl);
        xh[r] = _mm512_mul_pd(xh[r], wh);
    }
    const size_t gR2 = std::max<size_t>(R >> 3, 1), gS = b8_S(R);
    for (int r = 0; r < 8; ++r) {
        const size_t K = K0 + (size_t)r;
        const size_t gpos = out_blocked ? batch8_posL(K, S) : (gapped ? b8_gpos(K, gR2, gS) : K);
        _mm512_store_pd(g_re + 8 * gpos, _mm512_fmsub_pd(xl[r], tr[r], _mm512_mul_pd(xh[r], ti[r])));
        _mm512_store_pd(g_im + 8 * gpos, _mm512_fmadd_pd(xl[r], ti[r], _mm512_mul_pd(xh[r], tr[r])));
    }
}

// Pass A of a DEFERRED block (ibdwt_pass_BA: the first block of a thread's range, the block of digit ib_m) from the overlaid
// digit state. Rows K0 .. K0+7 of the block c0: the stored digits w (int32 pairs in re, normalized by ib_norm_rows8 WITHOUT
// the carry in) get the carry in of the run to the left (ib_rc + its top bit ib_h7, global tables; run 0: ib_cin0) and
// ib_madd at digit ib_m, then a floating-point chain over digits 0..6 (balanced, as ib_norm_pack_rows8); digit 7 keeps the
// rest unnormalized and gives away its own top bit h7 (top bit of the stored w7; 0 for the top digit L-1): the block to
// the right has already added that h7 and taken this run's carry out, so the value is exactly that of the old path.
// Weights, twist -> rows K0 .. K0+7 of the gather buffer g (gapped as batch8_gather). Block-major tables as
// ib_norm_pack_rows8; no carries are written.
static inline void ib_pack_deferred_rows8(const FFTContext& ctx, const double* xsrc, size_t xs, size_t K0, size_t c0, size_t C,
                                          size_t R, double* __restrict g_re, double* __restrict g_im, bool gapped) {
    const size_t N = ctx.fft_len, L = 2 * N;
    const size_t NR = N >> 3;
    const size_t ri0 = (c0 >> 3) * R + K0;
    const int64_t* const rc = ctx.ib_rc.data();
    const uint8_t* const h7 = ctx.ib_h7.data();
    const size_t M = (C >> 6) * R;
    const double* const wT = ctx.ib_wT.data() + (c0 >> 6) * R + K0;
    const int q0 = (int)ctx.ib_IL[1];
    const __m512d B0 = _mm512_set1_pd(std::ldexp(1.0, q0)), B1 = _mm512_set1_pd(std::ldexp(1.0, q0 + 1));
    const __m512d I0 = _mm512_set1_pd(std::ldexp(1.0, -q0)), I1 = _mm512_set1_pd(std::ldexp(1.0, -q0 - 1));
    const __m512d H0 = _mm512_set1_pd(std::ldexp(1.0, q0 - 1)), H1 = _mm512_set1_pd(std::ldexp(1.0, q0));   // 2^(b-1)
    uint64_t bgl, cml, bgh, cmh;
    ib_masks8(ctx.ib_bm.data() + ri0, bgl, cml);
    ib_masks8(ctx.ib_bm.data() + NR + ri0, bgh, cmh);
    const size_t l = c0 & 63;
    const __m512d WL = _mm512_loadu_pd(ctx.ib_WL.data() + l);
    const __m512d TLr = _mm512_loadu_pd(ctx.ib_TLr.data() + l), TLi = _mm512_loadu_pd(ctx.ib_TLi.data() + l);
    __m512d xl[8], xh[8];
    for (int r = 0; r < 8; ++r) {
        __m512i fl, fh;
        ib_ovl_load(xsrc + (K0 + (size_t)r) * xs + c0, fl, fh);
        xl[r] = _mm512_cvtepi64_pd(fl);
        xh[r] = _mm512_cvtepi64_pd(fh);
    }
    auto half = [&](__m512d* x, size_t off, const double* wh, uint64_t big, uint64_t cm) {
        // carries in (+ top bit of the run to the left), per row; the row whose run holds digit ib_m
        alignas(64) int64_t cin[8];
        int madd_row = -1;
        for (int r = 0; r < 8; ++r) {
            const size_t run = (off + (K0 + (size_t)r) * C + c0) >> 3;
            if (run == 0) cin[r] = ctx.ib_cin0;
            else {
                const size_t ri = ib_ri(ctx, run - 1);
                cin[r] = rc[ri] + (int64_t)h7[ri];
            }
            if (ctx.ib_madd != 0 && (ctx.ib_m >> 3) == run) madd_row = r;
        }
        st_transpose8x8(x);    // x[s] lane r = stored digit s of the run of row K0 + r
        const uint64_t bt = bits_transpose8(big);
        // own top bit of digit 7 (from the STORED digit, before ib_madd; 0 for the top digit L-1)
        const __mmask8 m7 = (__mmask8)(bt >> 56);
        __mmask8 h7own = _mm512_cmp_pd_mask(x[7], _mm512_mask_blend_pd(m7, H0, H1), _CMP_GE_OQ);
        if (off == N && c0 + 8 == C && K0 + 8 == R) h7own &= 0x7F;   // run L/8 - 1: lane 7 holds the top digit
        if (madd_row >= 0) {
            const int s = (int)(ctx.ib_m & 7);
            x[s] = _mm512_mask_add_pd(x[s], (__mmask8)(1u << madd_row), x[s], _mm512_set1_pd((double)ctx.ib_madd));
        }
        __m512d carry = _mm512_cvtepi64_pd(_mm512_load_si512((const void*)cin));
        for (int s = 0; s < 7; ++s) {
            const __mmask8 m = (__mmask8)(bt >> (8 * s));
            const __m512d t = _mm512_add_pd(x[s], carry);
            const __m512d q = _mm512_roundscale_pd(_mm512_mul_pd(t, _mm512_mask_blend_pd(m, I0, I1)),
                                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            x[s] = _mm512_fnmadd_pd(q, _mm512_mask_blend_pd(m, B0, B1), t);
            carry = q;
        }
        x[7] = _mm512_mask_sub_pd(_mm512_add_pd(x[7], carry), h7own, _mm512_add_pd(x[7], carry), _mm512_mask_blend_pd(m7, B0, B1));
        st_transpose8x8(x);
        for (int r = 0; r < 8; ++r) {
            const size_t j0 = off + (K0 + (size_t)r) * C + c0;
            __m512d wv = _mm512_mul_pd(_mm512_set1_pd(wh[r]), WL);
            wv = _mm512_mask_mul_pd(wv, (__mmask8)(cm >> (8 * r)), wv, _mm512_set1_pd(2.0));
            if (j0 == 0) wv = _mm512_mask_mov_pd(wv, 0x01, _mm512_set1_pd(1.0));   // digit 0: a = 1
            x[r] = _mm512_mul_pd(x[r], wv);
        }
    };
    half(xl, 0, wT, bgl, cml);
    half(xh, N, wT + 4 * M, bgh, cmh);
    (void)L;
    const size_t gR2 = std::max<size_t>(R >> 3, 1), gS = b8_S(R);
    for (int r = 0; r < 8; ++r) {
        const size_t K = K0 + (size_t)r;
        const __m512d Hr = _mm512_set1_pd(wT[2 * M + r]), Hi = _mm512_set1_pd(wT[3 * M + r]);
        const __m512d tr = _mm512_fmsub_pd(Hr, TLr, _mm512_mul_pd(Hi, TLi));
        const __m512d ti = _mm512_fmadd_pd(Hr, TLi, _mm512_mul_pd(Hi, TLr));
        const size_t gpos = gapped ? b8_gpos(K, gR2, gS) : K;
        _mm512_store_pd(g_re + 8 * gpos, _mm512_fmsub_pd(xl[r], tr, _mm512_mul_pd(xh[r], ti)));
        _mm512_store_pd(g_im + 8 * gpos, _mm512_fmadd_pd(xl[r], ti, _mm512_mul_pd(xh[r], tr)));
    }
}

// IBDWT pass B' fused with pass A of the NEXT square (one sweep through the data
// instead of two). In-place layout, overlaid digit state: for every block of 8 columns c0, the inverse column FFT,
// the normalization into the digits (ib_norm_rows8, digits into re, run carries into ib_rc, top bits into ib_h7),
// then at once - the block still in L2 - pass A of the next square on the same columns (batch8_gather_ib from re,
// forward column FFT, store back into re/im). Pass A of block c0 needs the carries and top bits of block c0 - 8
// (the runs to its left), so every thread takes a contiguous range of blocks in ascending order; pass A of the first
// block of each range (its left neighbour belongs to the previous thread; block 0: the last block, rows r - 1) waits
// for a barrier, as does the block holding digit ib_m (ib_madd comes from the top run, the very last block).
// errs[thread] = max rounding error. R = n1, C = n2, row stride rs = n2 + pad.
static void ibdwt_pass_BA(FFTContext& ctx, size_t R, size_t C, size_t rs, double* errs) {
    const Tw2 tw2(ctx);
    const bool two = batch8_two_level(R);
    const size_t R2 = R >> 3;
    const size_t BL = b8_len(R);
    const size_t S = b8_S(R);
    const bool ip = b8_inplace(R);
    const double* const w1r = ctx.r_w1r.data();
    const double* const w1i = ctx.r_w1i.data();
    double* const re = ctx.re;
    double* const im = ctx.im;
    const size_t NB = C / 8;
    const size_t N = ctx.fft_len;
    const size_t mblock = ((ctx.ib_m % N) % C) / 8;   // block whose pass A adds ib_madd
    // pass A reads the overlaid digits with the carries in: state as after B'
    ctx.ib_ovl = true;
    ctx.ib_rc_live = true;
    // The B' gathers take their twiddles from a base every 8 rows (ctx.tw_fast, as the classic square): only every 8th row
    // of lane_g is read (49 instead of 393 KB of L2 at R = 3072). Max Err measured unchanged on a hard case.
    ctx.tw_fast = true;
    // Adaptive ranges: with threads on both CCDs the busiest thread made the others wait at the barrier (the CCDs differ,
    // e.g. 96 vs 32 MB L3 for the tables). Each thread's contiguous range is sized by its speed (blocks per second) in the
    // previous passes (ctx.ba_w, smoothed); the ranges stay contiguous, so the deferred blocks are the same as before.
    const int ntm = std::min(omp_get_max_threads(), 64);
    if ((int)ctx.ba_w.size() != ntm) ctx.ba_w.assign((size_t)ntm, 1.0);
    size_t bnd[65];
    {
        double tot = 0.0;
        for (int t = 0; t < ntm; ++t) tot += ctx.ba_w[(size_t)t];
        double cum = 0.0;
        bnd[0] = 0;
        for (int t = 1; t < ntm; ++t) {
            cum += ctx.ba_w[(size_t)(t - 1)];
            size_t b = (size_t)std::llround((double)NB * cum / tot);
            b = std::max(b, bnd[t - 1] + 1);                       // at least one block per thread
            b = std::min(b, NB - (size_t)(ntm - t));
            bnd[t] = b;
        }
        bnd[ntm] = NB;
    }
    double ba_time[64] = {};
    int ba_nt = 0;
    // profile (LOTR_PROFILE): [SUBPASS] A = gather_ib / - / FFT / store, B' = gather / - / FFT / normalization (in "store")
    SubAcc subA[64], subB[64];
    int sub_nt = 1;
    (void)subA; (void)subB; (void)sub_nt;
#pragma omp parallel
    {
        double* const scr = batch8_scratch(R, 1, 0);
        double* const a_re = scr;          double* const a_im = scr + BL;
        double* const b_re = scr + 2 * BL; double* const b_im = scr + 3 * BL;
        double* const c_re = scr + 4 * BL; double* const c_im = scr + 5 * BL;
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        if (tid == 0) sub_nt = nt;
        SubAcc& accA = subA[tid < 64 ? tid : 63];
        SubAcc& accB = subB[tid < 64 ? tid : 63];
        (void)accA; (void)accB;
        __m512d vmax = _mm512_setzero_pd();
        // column FFT of the gathered block a (as do_block in bailey_colfft_pass8, all rows); in place: of x (a or b)
        auto col_fft = [&](bool inv, const double*& rr, const double*& ri, double* x_re = nullptr, double* x_im = nullptr) {
            if (ip) {
                if (!x_re) { x_re = a_re; x_im = a_im; }
                if (inv) b8_ip_stage1<false, true>(x_re, x_im, R, w1r, w1i, 0, R2);
                else     b8_ip_stage1<false, false>(x_re, x_im, R, w1r, w1i, 0, R2);
                b8_ip_subs(x_re, x_im, c_re, c_im, R, w1r, w1i, inv, 0, 8);
                rr = x_re; ri = x_im;
            }
            else if (two) {
                fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, w1r, w1i, inv);
                rr = b_re; ri = b_im;
            }
            else {
                const bool in_b = fft_batch8(a_re, a_im, b_re, b_im, R, w1r, w1i, inv);
                rr = in_b ? b_re : a_re;
                ri = in_b ? b_im : a_im;
            }
        };
        auto do_B = [&](size_t c0) {   // inverse column FFT (twiddled gather) + normalization into the digit state
            SUB_T(t0);
            batch8_gather(re, im, R, rs, c0, false, 0, true, true, ctx, tw2, a_re, a_im, R, true, 0, ~(size_t)0, ip);
            SUB_T(t1);
            const double* rr; const double* ri;
            col_fft(true, rr, ri);
            SUB_T(t2);
            for (size_t K0 = 0; K0 < R; K0 += 8) ib_norm_rows8(ctx, rr, ri, two, S, K0, c0, C, vmax, re, rs);
            SUB_T(t3);
            SUB_ADD(accB.c[0], t0, t1);
            SUB_ADD(accB.c[2], t1, t2);
            SUB_ADD(accB.c[3], t2, t3);
        };
        // store of pass A's spectrum rr/ri (blocked layout if two) back into re/im, columns c0 .. c0+7
        auto store_A = [&](size_t c0, const double* rr, const double* ri) {
            SUB_T(t2);
            if (two) {
                for (size_t K = 0; K < R; ++K) {
                    const size_t pos = batch8_posL(K, S);
                    _mm512_storeu_pd(re + K * rs + c0, _mm512_load_pd(rr + 8 * pos));
                    _mm512_storeu_pd(im + K * rs + c0, _mm512_load_pd(ri + 8 * pos));
                }
            }
            else {
                for (size_t k = 0; k < R; ++k) {
                    _mm512_storeu_pd(re + k * rs + c0, _mm512_load_pd(rr + 8 * k));
                    _mm512_storeu_pd(im + k * rs + c0, _mm512_load_pd(ri + 8 * k));
                }
            }
            SUB_T(t3);
            SUB_ADD(accA.c[3], t2, t3);
        };
        auto do_A = [&](size_t c0) {   // pass A of the next square from the overlaid digits in re -> back into re/im
            SUB_T(t0);
            if (ip) {
                for (size_t K0 = 0; K0 < R; K0 += 8) ib_pack_deferred_rows8(ctx, re, rs, K0, c0, C, R, a_re, a_im, ip);
            }
            else {
                batch8_gather_ib(ctx, R, C, c0, a_re, a_im, 0, R, ip, re, rs);
            }
            SUB_T(t1);
            const double* rr; const double* ri;
            col_fft(false, rr, ri);
            SUB_T(t2);
            SUB_ADD(accA.c[0], t0, t1);
            SUB_ADD(accA.c[2], t1, t2);
            store_A(c0, rr, ri);
        };
        // B' and A of a block whose left neighbour is done (in-place form): the normalized digits go straight into
        // pass A's gather buffer b (ib_norm_pack_rows8), no digits in re
        auto do_BA = [&](size_t c0, const IbCarryIO* io) {
            SUB_T(t0);
            batch8_gather(re, im, R, rs, c0, false, 0, true, true, ctx, tw2, a_re, a_im, R, true, 0, ~(size_t)0, ip);
            SUB_T(t1);
            const double* rr; const double* ri;
            col_fft(true, rr, ri);
            SUB_T(t2);
            {   // (do_BA runs only in place: rr == a_re) one buffer: pass A's input over the B' spectrum (blocked layout, in place), its FFT as DIT
                // (blocked -> natural gapped, b8_ip_ditf), the store reads the natural rows. Buffer b stays unused: the
                // thread's L2 set at R = 3072 drops by 393 KB (a + b + lane_g were ~1.2 MB on 1 MB of L2).
                for (size_t K0 = 0; K0 < R; K0 += 8)
                    ib_norm_pack_rows8(ctx, rr, ri, two, S, K0, c0, C, R, vmax, a_re, a_im, ip, io, true);
                SUB_T(t3o);
                b8_ip_ditf(a_re, a_im, c_re, c_im, R, w1r, w1i, false);
                SUB_T(t4o);
                SUB_ADD(accB.c[0], t0, t1);
                SUB_ADD(accB.c[2], t1, t2);
                SUB_ADD(accB.c[3], t2, t3o);
                SUB_ADD(accA.c[2], t3o, t4o);
                const size_t gR2 = std::max<size_t>(R >> 3, 1);
                SUB_T(t5o);
                for (size_t K = 0; K < R; ++K) {
                    const size_t pos = b8_gpos(K, gR2, S);
                    _mm512_storeu_pd(re + K * rs + c0, _mm512_load_pd(a_re + 8 * pos));
                    _mm512_storeu_pd(im + K * rs + c0, _mm512_load_pd(a_im + 8 * pos));
                }
                SUB_T(t6o);
                SUB_ADD(accA.c[3], t5o, t6o);
            }
        };
        const bool np = ip;
        const bool adapt = (nt == ntm);
        const size_t cb0 = adapt ? bnd[tid] : NB * (size_t)tid / (size_t)nt;
        const size_t cb1 = adapt ? bnd[tid + 1] : NB * (size_t)(tid + 1) / (size_t)nt;
        if (tid == 0) ba_nt = nt;
        // Run carries between this thread's consecutive blocks: in a local buffer (2R carries, L2) instead of the global
        // ib_rc / ib_h7 (much RAM traffic per pass on large lengths); the global tables only where a deferred block reads them:
        // the last block of the range (the next thread's first) and the block before ib_m.
        thread_local std::vector<int64_t> rcloc_v;
        if (rcloc_v.size() < 2 * R) rcloc_v.assign(2 * R, 0);
        int64_t* const rcloc = rcloc_v.data();
        int64_t* const rcg = ctx.ib_rc.data();
        uint8_t* const h7g = ctx.ib_h7.data();
        const size_t NR = N >> 3;
        const auto tw0 = std::chrono::steady_clock::now();
        for (size_t cb = cb0; cb < cb1; ++cb) {
            if (cb == cb0 || cb == mblock) do_B(cb * 8);   // pass A after the barrier
            else if (np) {
                const bool left_old = (cb - 1 == cb0) || (cb - 1 == mblock);   // the left block wrote the global tables
                const bool need_g = (cb + 1 == cb1) || (cb + 1 == mblock);      // a deferred block reads ours
                IbCarryIO io;
                for (int hh = 0; hh < 2; ++hh) {
                    const size_t G = (size_t)hh * NR + cb * R, Gl = G - R;
                    io.rcL[hh] = left_old ? rcg + Gl : rcloc + (size_t)hh * R;
                    io.h7L[hh] = left_old ? h7g + Gl : nullptr;
                    io.rcO[hh] = rcloc + (size_t)hh * R;
                    io.rcG[hh] = need_g ? rcg + G : nullptr;
                    io.h7G[hh] = need_g ? h7g + G : nullptr;
                }
                do_BA(cb * 8, &io);
            }
            else { do_B(cb * 8); do_A(cb * 8); }
        }
        if (tid < 64) ba_time[tid] = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        SUB_T(w0);
#pragma omp barrier
#pragma omp single
        ib_top_wrap(ctx);   // the top run (last block) is done: cin0 / madd for run 0 and digit m
        SUB_T(w1);
        SUB_ADD(accA.c[1], w0, w1);   // [SUBPASS] A "st1" = waiting at the barrier
        if (cb0 < cb1) do_A(cb0 * 8);
        if (mblock > cb0 && mblock < cb1) do_A(mblock * 8);
        errs[tid < 64 ? tid : 63] = _mm512_reduce_max_pd(vmax);
    }
    ctx.tw_fast = false;
    // speeds of this pass -> range weights of the next (blocks per second relative to the mean, smoothed 50/50)
    if (ba_nt == ntm) {
        double sp[64], mean = 0.0;
        bool ok = true;
        for (int t = 0; t < ntm; ++t) {
            const double nb = (double)(bnd[t + 1] - bnd[t]);
            if (ba_time[t] <= 0.0) { ok = false; break; }
            sp[t] = nb / ba_time[t];
            mean += sp[t];
        }
        if (ok && mean > 0.0) {
            mean /= (double)ntm;
            for (int t = 0; t < ntm; ++t) ctx.ba_w[(size_t)t] = 0.5 * ctx.ba_w[(size_t)t] + 0.5 * (sp[t] / mean);
        }
    }
#ifdef LOTR_PROFILE
    sub_merge(ctx, 0, 4, subA, sub_nt);
    sub_merge(ctx, 11, 4, subB, sub_nt);
#endif
}

// =====================================================================
// KERNEL MICROBENCHMARK (LOTR_KBENCH=1, cli.cpp): the batch-8 column FFT on one block of 8 columns, data hot in L1/L2,
// one thread. Reports ns per call and GFlop/s (5 * R * log2(R) per column, 8 columns), plus stage 1 of the two-level
// FFT alone and a plain copy of the same data for scale. The input is restored from a copy before every call (values
// stay bounded); the copy time is measured separately and subtracted.
// =====================================================================
void fft_kernel_bench() {
    using clk = std::chrono::steady_clock;
    std::printf("[KBENCH] batch-8 column FFT kernels, 1 thread, 8 columns per call\n");
    for (const size_t R : { (size_t)256, (size_t)448, (size_t)512, (size_t)768, (size_t)896, (size_t)1024, (size_t)2048, (size_t)3072 }) {
        const bool two = batch8_two_level(R);
        const size_t BL = b8_len(R);
        double* const buf = static_cast<double*>(std::aligned_alloc(64, 8 * BL * sizeof(double)));
        double* const a_re = buf;          double* const a_im = buf + BL;
        double* const b_re = buf + 2 * BL; double* const b_im = buf + 3 * BL;
        double* const c_re = buf + 4 * BL; double* const c_im = buf + 5 * BL;
        double* const o_re = buf + 6 * BL; double* const o_im = buf + 7 * BL;   // pristine input
        uint64_t st = 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < 8 * R; ++i) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            o_re[i] = (double)(int32_t)(st >> 40) / 65536.0;
            o_im[i] = (double)(int32_t)(st >> 20 & 0xFFFFF) / 65536.0;
        }
        std::vector<double> wr(R), wi(R);
        for (size_t i = 0; i < R; ++i) { wr[i] = std::cos(2.0 * M_PI * (double)i / (double)R); wi[i] = -std::sin(2.0 * M_PI * (double)i / (double)R); }   // e^(-2 pi i/R), as ctx.w1
        const size_t bytes = 8 * R * sizeof(double);
        const size_t reps = std::max<size_t>(2000, (size_t)(400000000.0 / (8.0 * (double)R * std::log2((double)R))));
        auto restore = [&]() { std::memcpy(a_re, o_re, bytes); std::memcpy(a_im, o_im, bytes); };
        auto timed = [&](auto&& body) -> double {   // ns per call
            for (int w = 0; w < 50; ++w) body();
            const auto t0 = clk::now();
            for (size_t r = 0; r < reps; ++r) body();
            return std::chrono::duration<double, std::nano>(clk::now() - t0).count() / (double)reps;
        };
        const double t_copy = timed([&]() { restore(); });
        const double t_fwd = timed([&]() {
            restore();
            if (two) fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, wr.data(), wi.data(), false);
            else     fft_batch8(a_re, a_im, b_re, b_im, R, wr.data(), wi.data(), false);
        }) - t_copy;
        const double t_inv = timed([&]() {
            restore();
            if (two) fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, wr.data(), wi.data(), true);
            else     fft_batch8(a_re, a_im, b_re, b_im, R, wr.data(), wi.data(), true);
        }) - t_copy;
        double t_st1 = 0.0;
        if (two) {
            t_st1 = timed([&]() {
                restore();
                batch8_2l_stage1<false>(a_re, a_im, false, b_re, b_im, R, wr.data(), wi.data(), false, 0, R >> 3);
            }) - t_copy;
        }
        const double flops = 8.0 * 5.0 * (double)R * std::log2((double)R);
        std::printf("[KBENCH] R=%4zu %s: fwd %8.0f ns (%5.1f GFlop/s)  inv %8.0f ns  stage1 %7.0f ns  copy(8R cplx) %6.0f ns  stages %d%s\n",
                    R, two ? "2-level" : "1-level", t_fwd, flops / t_fwd, t_inv, t_st1, t_copy, batch8_stages(two ? (R >> 3) : R),
                    two ? " (+ stage 1)" : "");
        {
            // against a naive DFT X_k = sum_n x_n w^(nk) (checks the butterflies themselves, radix-7 too):
            // one-level fft_batch8 (natural output) and the two-level form (blocked output, batch8_posL)
            std::vector<double> nr(8 * R), ni(8 * R);
            double mv = 0.0;
            for (size_t k = 0; k < R; ++k) {
                for (int l = 0; l < 8; ++l) {
                    long double sr = 0.0L, si = 0.0L;
                    for (size_t n = 0; n < R; ++n) {
                        const size_t e = (n * k) % R;
                        const long double xr = o_re[8 * n + l], xi = o_im[8 * n + l];
                        sr += xr * (long double)wr[e] - xi * (long double)wi[e];
                        si += xr * (long double)wi[e] + xi * (long double)wr[e];
                    }
                    nr[8 * k + l] = (double)sr; ni[8 * k + l] = (double)si;
                    mv = std::max(mv, std::fabs((double)sr));
                }
            }
            restore();
            const bool in_b = fft_batch8(a_re, a_im, b_re, b_im, R, wr.data(), wi.data(), false);
            const double* const fr = in_b ? b_re : a_re;
            const double* const fi = in_b ? b_im : a_im;
            double d1 = 0.0;
            for (size_t i = 0; i < 8 * R; ++i) d1 = std::max(d1, std::max(std::fabs(fr[i] - nr[i]), std::fabs(fi[i] - ni[i])));
            double d2 = 0.0;
            if (two) {
                restore();
                fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, wr.data(), wi.data(), false);
                const size_t S = b8_S(R);
                for (size_t k = 0; k < R; ++k) {
                    const size_t pos = batch8_posL(k, S);
                    for (int l = 0; l < 8; ++l)
                        d2 = std::max(d2, std::max(std::fabs(b_re[8 * pos + l] - nr[8 * k + l]), std::fabs(b_im[8 * pos + l] - ni[8 * k + l])));
                }
            }
            const bool ok = d1 <= 1e-12 * mv && d2 <= 1e-12 * mv;
            std::printf("[KBENCH]        vs naive DFT: 1-level max diff %.2e, 2-level %.2e (of %.2e)  %s\n", d1, d2, mv, ok ? "OK" : "MISMATCH");
        }
        if (b8_inplace(R)) {
            // in-place form (b8_ip_dif / b8_ip_ditf) against fft_batch8_2l: same spectrum, same inverse, then timings
            const size_t R2 = R >> 3, S = b8_S(R);
            double* const xb = static_cast<double*>(std::aligned_alloc(64, (4 * BL + 4 * 8 * R2) * sizeof(double)));
            double* const x_re = xb;          double* const x_im = xb + BL;
            double* const g_re = xb + 2 * BL; double* const g_im = xb + 3 * BL;   // pristine input, gapped layout
            double* const sc_re = xb + 4 * BL; double* const sc_im = sc_re + 2 * 8 * R2;   // 2 x R/8 vectors each (b8_ip_subs)
            auto vdiff = [](const double* p, const double* q, double& md, double& mv) {
                for (int l = 0; l < 8; ++l) { md = std::max(md, std::fabs(p[l] - q[l])); mv = std::max(mv, std::fabs(q[l])); }
            };
            auto make_g = [&](bool half) {
                for (size_t r = 0; r < R; ++r) {
                    const size_t gp = b8_gpos(r, R2, S);
                    for (int l = 0; l < 8; ++l) {
                        const bool z = half && r >= R / 2;   // half: garbage in the gapped copy (must not be read), 0 in a
                        g_re[8 * gp + l] = z ? 1e300 : o_re[8 * r + l];
                        g_im[8 * gp + l] = z ? -1e300 : o_im[8 * r + l];
                        a_re[8 * r + l] = z ? 0.0 : o_re[8 * r + l];
                        a_im[8 * r + l] = z ? 0.0 : o_im[8 * r + l];
                    }
                }
            };
            for (int half = 0; half < 2; ++half) {
                make_g(half != 0);
                fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, wr.data(), wi.data(), false);   // spectrum (blocked) in b
                std::memcpy(x_re, g_re, BL * sizeof(double)); std::memcpy(x_im, g_im, BL * sizeof(double));
                b8_ip_dif(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), false, half != 0);
                double md = 0, mv = 0;
                for (size_t K = 0; K < R; ++K) {
                    const size_t pos = batch8_posL(K, S);
                    vdiff(x_re + 8 * pos, b_re + 8 * pos, md, mv); vdiff(x_im + 8 * pos, b_im + 8 * pos, md, mv);
                }
                // inverse of the spectrum b: out of place (result blocked, element n at posL(n)) vs DIT in place (gapped)
                fft_batch8_2l(b_re, b_im, true, a_re, a_im, c_re, c_im, R, wr.data(), wi.data(), true);
                std::memcpy(x_re, b_re, BL * sizeof(double)); std::memcpy(x_im, b_im, BL * sizeof(double));
                b8_ip_ditf(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), true);
                double md2 = 0, mv2 = 0;
                for (size_t n = 0; n < R; ++n) {
                    const size_t pa = batch8_posL(n, S), px = b8_gpos(n, R2, S);
                    vdiff(x_re + 8 * px, a_re + 8 * pa, md2, mv2); vdiff(x_im + 8 * px, a_im + 8 * pa, md2, mv2);
                }
                const bool ok = md <= 1e-12 * mv && md2 <= 1e-12 * mv2;
                std::printf("[KBENCH]        in place%s vs 2-level: fwd max diff %.2e (of %.2e), inv %.2e (of %.2e)  %s\n",
                            half ? " (half)" : "", md, mv, md2, mv2, ok ? "OK" : "MISMATCH");
            }
            // round trips (no reference needed): forward then inverse must give R * x
            {
                make_g(false);
                const double sR = (double)R;
                std::memcpy(x_re, g_re, BL * sizeof(double)); std::memcpy(x_im, g_im, BL * sizeof(double));
                b8_ip_dif(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), false, false);
                b8_ip_ditf(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), true);
                double e_ip = 0, e_dd = 0, e_2l = 0, mv = 0;
                for (size_t n = 0; n < R; ++n) {
                    const size_t px = b8_gpos(n, R2, S);
                    for (int l = 0; l < 8; ++l) {
                        e_ip = std::max(e_ip, std::fabs(x_re[8 * px + l] - sR * o_re[8 * n + l]));
                        e_ip = std::max(e_ip, std::fabs(x_im[8 * px + l] - sR * o_im[8 * n + l]));
                        mv = std::max(mv, sR * std::fabs(o_re[8 * n + l]));
                    }
                }
                // in-place forward, then the in-place DIF inverse (the path of pass B') read in the blocked layout
                std::memcpy(x_re, g_re, BL * sizeof(double)); std::memcpy(x_im, g_im, BL * sizeof(double));
                b8_ip_dif(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), false, false);
                // blocked spectrum -> gapped natural order (X[K] at posL(K) -> gpos(K)) for the DIF inverse
                for (size_t K = 0; K < R; ++K)
                    for (int l = 0; l < 8; ++l) {
                        c_re[8 * b8_gpos(K, R2, S) + l] = x_re[8 * batch8_posL(K, S) + l];
                        c_im[8 * b8_gpos(K, R2, S) + l] = x_im[8 * batch8_posL(K, S) + l];
                    }
                std::memcpy(x_re, c_re, BL * sizeof(double)); std::memcpy(x_im, c_im, BL * sizeof(double));
                b8_ip_dif(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), true, false);
                for (size_t n = 0; n < R; ++n)
                    for (int l = 0; l < 8; ++l) {
                        e_dd = std::max(e_dd, std::fabs(x_re[8 * batch8_posL(n, S) + l] - sR * o_re[8 * n + l]));
                        e_dd = std::max(e_dd, std::fabs(x_im[8 * batch8_posL(n, S) + l] - sR * o_im[8 * n + l]));
                    }
                // the old out-of-place pair: fft_batch8_2l forward, fft_batch8_2l inverse from the blocked layout (in_L)
                fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, wr.data(), wi.data(), false);
                fft_batch8_2l(b_re, b_im, true, a_re, a_im, c_re, c_im, R, wr.data(), wi.data(), true);
                for (size_t n = 0; n < R; ++n)
                    for (int l = 0; l < 8; ++l) {
                        e_2l = std::max(e_2l, std::fabs(a_re[8 * batch8_posL(n, S) + l] - sR * o_re[8 * n + l]));
                        e_2l = std::max(e_2l, std::fabs(a_im[8 * batch8_posL(n, S) + l] - sR * o_im[8 * n + l]));
                    }
                auto st = [&](double e) { return e <= 1e-9 * mv ? "OK" : "BAD"; };
                std::printf("[KBENCH]        round trip (of %.2e): in place DIF+DIT %.2e %s | in place DIF+DIF %.2e %s | 2-level+2-level(in_L) %.2e %s\n",
                            mv, e_ip, st(e_ip), e_dd, st(e_dd), e_2l, st(e_2l));
            }
            make_g(false);
            auto restore_g = [&]() { std::memcpy(x_re, g_re, BL * sizeof(double)); std::memcpy(x_im, g_im, BL * sizeof(double)); };
            const double t_gcopy = timed([&]() { restore_g(); });
            const double t_ipf = timed([&]() { restore_g(); b8_ip_dif(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), false, false); }) - t_gcopy;
            const double t_ipi = timed([&]() { restore_g(); b8_ip_ditf(x_re, x_im, sc_re, sc_im, R, wr.data(), wi.data(), true); }) - t_gcopy;
            std::printf("[KBENCH]        in place: fwd %8.0f ns (%5.1f GFlop/s)  inv %8.0f ns  (vs 2-level fwd %.0f, inv %.0f)\n",
                        t_ipf, flops / t_ipf, t_ipi, t_fwd, t_inv);
            std::free(xb);
        }
        std::free(buf);
    }

    // --- what one radix-8 butterfly (16 vectors in, 16 out) costs in isolation ---
    // core clock: a chain of dependent integer adds (1 cycle each)
    double ns_per_cycle = 0.0;
    {
        const uint64_t n = 200000000ULL;
        uint64_t x = 1;
        const auto t0 = clk::now();
        for (uint64_t i = 0; i < n; ++i) asm volatile("add %1, %0" : "+r"(x) : "r"(i));
        ns_per_cycle = std::chrono::duration<double, std::nano>(clk::now() - t0).count() / (double)n;
        std::printf("[KBENCH] core clock ~ %.2f GHz (dependent adds; x=%llu)\n", 1.0 / ns_per_cycle, (unsigned long long)(x & 1));
    }
    const __m512d v_C = _mm512_set1_pd(0.70710678118654752440);
    const __m512d v_eighth = _mm512_set1_pd(0.125);
    alignas(64) static double lbuf[2][16][8];
    for (int s = 0; s < 2; ++s) for (int v = 0; v < 16; ++v) for (int l = 0; l < 8; ++l) lbuf[s][v][l] = 1.0 + 0.01 * (v * 8 + l + s);
    const size_t iters = 4000000;
    auto report = [&](const char* name, double ns_total) {
        const double ns = ns_total / (double)iters;
        std::printf("[KBENCH] %-58s %6.2f ns = %5.1f cycles per butterfly\n", name, ns, ns / ns_per_cycle);
    };
    // (a) registers only: y = core(x) / 8 fed back (latency of one butterfly + 16 muls)
    {
        __m512d r[8], i[8], yr[8], yi[8];
        for (int q = 0; q < 8; ++q) { r[q] = _mm512_load_pd(lbuf[0][q]); i[q] = _mm512_load_pd(lbuf[0][8 + q]); }
        const auto t0 = clk::now();
        for (size_t it = 0; it < iters; ++it) {
            r8_core<false>(r, i, yr, yi, v_C);
            for (int q = 0; q < 8; ++q) { r[q] = _mm512_mul_pd(yr[q], v_eighth); i[q] = _mm512_mul_pd(yi[q], v_eighth); }
        }
        const double t = std::chrono::duration<double, std::nano>(clk::now() - t0).count();
        double sink = 0; for (int q = 0; q < 8; ++q) sink += _mm512_reduce_add_pd(r[q]) + _mm512_reduce_add_pd(i[q]);
        report("(a) core, registers only (dependent, +16 mul)", t);
        if (sink == 12345.678) std::printf(" ");
    }
    // (b) the same through L1: load 16 vectors from one buffer, store 16 to the other, swap (contiguous)
    {
        const auto t0 = clk::now();
        for (size_t it = 0; it < iters; ++it) {
            const int s = (int)(it & 1);
            __m512d r[8], i[8], yr[8], yi[8];
            for (int q = 0; q < 8; ++q) { r[q] = _mm512_load_pd(lbuf[s][q]); i[q] = _mm512_load_pd(lbuf[s][8 + q]); }
            r8_core<false>(r, i, yr, yi, v_C);
            for (int q = 0; q < 8; ++q) {
                _mm512_store_pd(lbuf[s ^ 1][q], _mm512_mul_pd(yr[q], v_eighth));
                _mm512_store_pd(lbuf[s ^ 1][8 + q], _mm512_mul_pd(yi[q], v_eighth));
            }
        }
        report("(b) core + 16 loads + 16 stores (L1, contiguous, dependent)", std::chrono::duration<double, std::nano>(clk::now() - t0).count());
    }
    // (c) independent butterflies, 64 KB in -> 64 KB out (like one stage of R = 512: 64 butterflies), no feedback
    {
        const size_t NB = 64;
        double* const in = static_cast<double*>(std::aligned_alloc(64, NB * 16 * 8 * sizeof(double)));
        double* const out = static_cast<double*>(std::aligned_alloc(64, NB * 16 * 8 * sizeof(double)));
        for (size_t x = 0; x < NB * 16 * 8; ++x) in[x] = 1.0 + 1e-6 * (double)(x % 1000);
        const size_t rounds = iters / NB;
        const auto t0 = clk::now();
        for (size_t rd = 0; rd < rounds; ++rd) {
            for (size_t b = 0; b < NB; ++b) {
                const double* src = in + b * 128;
                double* dst = out + b * 128;
                __m512d r[8], i[8], yr[8], yi[8];
                for (int q = 0; q < 8; ++q) { r[q] = _mm512_load_pd(src + 8 * q); i[q] = _mm512_load_pd(src + 64 + 8 * q); }
                r8_core<false>(r, i, yr, yi, v_C);
                for (int q = 0; q < 8; ++q) { _mm512_store_pd(dst + 8 * q, yr[q]); _mm512_store_pd(dst + 64 + 8 * q, yi[q]); }
            }
        }
        report("(c) core, independent, contiguous (64 KB in, 64 KB out)", std::chrono::duration<double, std::nano>(clk::now() - t0).count());
        // (d) the same with the 7 twiddle multiplications (twiddles in registers)
        __m512d wr[8], wi[8];
        for (int q = 0; q < 8; ++q) { wr[q] = _mm512_set1_pd(std::cos(0.1 * q)); wi[q] = _mm512_set1_pd(std::sin(0.1 * q)); }
        const auto t1 = clk::now();
        for (size_t rd = 0; rd < rounds; ++rd) {
            for (size_t b = 0; b < NB; ++b) {
                const double* src = in + b * 128;
                double* dst = out + b * 128;
                __m512d r[8], i[8], yr[8], yi[8];
                for (int q = 0; q < 8; ++q) { r[q] = _mm512_load_pd(src + 8 * q); i[q] = _mm512_load_pd(src + 64 + 8 * q); }
                r8_tw<false>(r, i, wr, wi, yr, yi, v_C);
                for (int q = 0; q < 8; ++q) { _mm512_store_pd(dst + 8 * q, yr[q]); _mm512_store_pd(dst + 64 + 8 * q, yi[q]); }
            }
        }
        report("(d) as (c) + 7 twiddle multiplications", std::chrono::duration<double, std::nano>(clk::now() - t1).count());
        // (e) as (c) but strided like stage 1 of R = 512: input vector q at q*64, output block p at p*65 (two-level gaps)
        const size_t SI = 64, SO = 65;
        double* const in2 = static_cast<double*>(std::aligned_alloc(64, 2 * 8 * SO * 8 * sizeof(double)));
        double* const out2 = static_cast<double*>(std::aligned_alloc(64, 2 * 8 * SO * 8 * sizeof(double)));
        for (size_t x = 0; x < 2 * 8 * SO * 8; ++x) { in2[x] = 1.0 + 1e-6 * (double)(x % 1000); out2[x] = 0.0; }
        double* const in2i = in2 + 8 * SO * 8;
        double* const out2i = out2 + 8 * SO * 8;
        const size_t rounds2 = iters / SI;
        const auto t2 = clk::now();
        for (size_t rd = 0; rd < rounds2; ++rd) {
            for (size_t k = 0; k < SI; ++k) {
                __m512d r[8], i[8], yr[8], yi[8];
                for (int q = 0; q < 8; ++q) { r[q] = _mm512_load_pd(in2 + 8 * (k + q * SI)); i[q] = _mm512_load_pd(in2i + 8 * (k + q * SI)); }
                r8_core<false>(r, i, yr, yi, v_C);
                for (int p = 0; p < 8; ++p) { _mm512_store_pd(out2 + 8 * (p * SO + k), yr[p]); _mm512_store_pd(out2i + 8 * (p * SO + k), yi[p]); }
            }
        }
        report("(e) as (c), strided loads (4 KB) / stores (4160 B) like stage 1", std::chrono::duration<double, std::nano>(clk::now() - t2).count() * (double)iters / (double)(rounds2 * SI));
        std::free(in); std::free(out); std::free(in2); std::free(out2);
    }

    // (f) two-level R = 512 in an L1-friendlier form: input already in the gapped layout (row r at (r / 64)*65 + r % 64),
    // stage 1 IN PLACE (it reads positions q*S + k and writes p*S + k: the same set), then each 64-point sub-FFT
    // through one 16 KB scratch (two stages: the result comes back into the block). Compared with fft_batch8_2l above.
    {
        const size_t R = 512, R2 = 64, S = b8_S(R);
        const size_t AL = 64 * S;   // doubles per re or im array (8 blocks of S vectors)
        double* const x = static_cast<double*>(std::aligned_alloc(64, 4 * AL * sizeof(double) + 4 * 8 * R2 * sizeof(double)));
        double* const x_re = x; double* const x_im = x + AL;
        double* const o_re = x + 2 * AL; double* const o_im = x + 3 * AL;    // pristine input
        double* const sc_re = x + 4 * AL; double* const sc_im = sc_re + 8 * R2;
        uint64_t st = 12345;
        for (size_t i = 0; i < 2 * AL; ++i) { st = st * 6364136223846793005ULL + 1; o_re[i] = (double)(int32_t)(st >> 40) / 65536.0; }
        std::vector<double> wr(R), wi(R);
        for (size_t i = 0; i < R; ++i) { wr[i] = std::cos(2.0 * M_PI * (double)i / (double)R); wi[i] = -std::sin(2.0 * M_PI * (double)i / (double)R); }   // e^(-2 pi i/R), as ctx.w1
        const __m512d vC = _mm512_set1_pd(0.70710678118654752440);
        auto restore = [&]() { std::memcpy(x_re, o_re, 2 * AL * sizeof(double)); };
        auto stage1_inplace = [&]() {
            for (size_t k = 0; k < R2; ++k) {
                __m512d r[8], i[8], yr[8], yi[8];
                for (int q = 0; q < 8; ++q) { r[q] = _mm512_load_pd(x_re + 8 * (q * S + k)); i[q] = _mm512_load_pd(x_im + 8 * (q * S + k)); }
                r8_core<false>(r, i, yr, yi, vC);
                _mm512_store_pd(x_re + 8 * k, yr[0]);
                _mm512_store_pd(x_im + 8 * k, yi[0]);
                for (int p = 1; p < 8; ++p) {
                    const __m512d w_r = _mm512_set1_pd(wr[(size_t)p * k]), w_i = _mm512_set1_pd(wi[(size_t)p * k]);
                    _mm512_store_pd(x_re + 8 * (p * S + k), _mm512_fmsub_pd(yr[p], w_r, _mm512_mul_pd(yi[p], w_i)));
                    _mm512_store_pd(x_im + 8 * (p * S + k), _mm512_fmadd_pd(yr[p], w_i, _mm512_mul_pd(yi[p], w_r)));
                }
            }
        };
        const size_t reps = 200000;
        auto timed = [&](auto&& body) -> double {
            for (int w = 0; w < 50; ++w) body();
            const auto t0 = clk::now();
            for (size_t r = 0; r < reps; ++r) body();
            return std::chrono::duration<double, std::nano>(clk::now() - t0).count() / (double)reps;
        };
        const double t_copy = timed([&]() { restore(); });
        const double t_st1 = timed([&]() { restore(); stage1_inplace(); }) - t_copy;
        const double t_all = timed([&]() {
            restore();
            stage1_inplace();
            for (size_t p = 0; p < 8; ++p)
                fft_batch8(x_re + 8 * p * S, x_im + 8 * p * S, sc_re, sc_im, R2, wr.data(), wi.data(), false, 8);
        }) - t_copy;
        std::printf("[KBENCH] (f) R=512 two-level, stage 1 in place + 16 KB scratch: fwd %6.0f ns  stage1 %6.0f ns  (copy %5.0f ns)\n",
                    t_all, t_st1, t_copy);
        std::free(x);
    }

    // --- RAM bandwidth (all threads of the process, as the giant FFTs run): what the passes over > L3 data can reach.
    // MB = src + dst together: 24 MB fits every L3 (32 MB), 64 MB only the V-cache CCD (96 MB), 256 MB none (RAM) ---
    for (const size_t MB : { (size_t)24, (size_t)64, (size_t)256 }) {
        const size_t nd = MB * 1024 * 1024 / 2 / sizeof(double);
        double* const src = static_cast<double*>(std::aligned_alloc(64, nd * sizeof(double)));
        double* const dst = static_cast<double*>(std::aligned_alloc(64, nd * sizeof(double)));
        const int nt = omp_get_max_threads();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nd; ++i) { src[i] = (double)(i & 1023); dst[i] = 0.0; }
        auto best_of = [&](auto&& body) -> double {   // seconds, best of 5
            double best = 1e30;
            for (int rep = 0; rep < 5; ++rep) {
                const auto t0 = clk::now();
                body();
                best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
            }
            return best;
        };
        const double bytes = (double)nd * sizeof(double);
        // (1) copy: read + write
        const double t_copy = best_of([&]() {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < nd; i += 8) _mm512_store_pd(dst + i, _mm512_load_pd(src + i));
        });
        // (2) read only
        double sink = 0.0;
        const double t_read = best_of([&]() {
            double s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : s)
            for (size_t i = 0; i < nd; i += 8) s += _mm512_reduce_add_pd(_mm512_load_pd(src + i));
            sink += s;
        });
        // (3) the column pattern of the passes: a matrix of rows of C = 2048 + 8 doubles; each thread takes blocks of 8
        // columns and reads 64 B per row (one line), writes them to a contiguous 8-column buffer and back (like gather
        // + store of a column pass, here without the FFT)
        const size_t C = 2048 + 8, Rr = nd / C;
        double* const blk = static_cast<double*>(std::aligned_alloc(64, (size_t)nt * Rr * 8 * sizeof(double)));
        const double t_col = best_of([&]() {
#pragma omp parallel
            {
                double* const b = blk + (size_t)omp_get_thread_num() * Rr * 8;
#pragma omp for schedule(static)
                for (size_t c0 = 0; c0 < 2048; c0 += 8) {
                    for (size_t r = 0; r < Rr; ++r) _mm512_store_pd(b + 8 * r, _mm512_loadu_pd(src + r * C + c0));
                    for (size_t r = 0; r < Rr; ++r) _mm512_storeu_pd(dst + r * C + c0, _mm512_load_pd(b + 8 * r));
                }
            }
        });
        const double col_bytes = 2.0 * (double)Rr * 2048 * sizeof(double);
        std::printf("[KBENCH] RAM %zu MB, %d threads: copy %.1f GB/s (read+write)  read %.1f GB/s  column pattern %.1f GB/s (read+write)%s\n",
                    MB, nt, 2.0 * bytes / t_copy * 1e-9, bytes / t_read * 1e-9, col_bytes / t_col * 1e-9, sink == 1.0 ? " " : "");
        std::free(blk);
        std::free(src);
        std::free(dst);
    }
}

// The column pass (batch-8 kernels)
static void bailey_colpass(const double* src_re, const double* src_im, double* dst_re, double* dst_im,
    size_t R, size_t C, bool twiddle, bool transposed_out, FFTContext& ctx, bool inverse,
    const std::vector<double>& w1r, const std::vector<double>& w1i) {
    bailey_colfft_pass8(src_re, src_im, dst_re, dst_im, R, C, twiddle, transposed_out, ctx, inverse, w1r, w1i);
}

// Bailey 2D dimensions (N = n1 * n2, n1 >= n2) and twiddle tables for the row FFTs (r_*, length n1) and column FFTs (c_*, length n2).
// N = 2^p or 3*2^p: n2 = the largest power of two with n1 = N/n2 >= n2, so the factor 3 goes into n1 (passes A and B')
// and n2 (the fused middle and the padded row length) stays a power of two. For N = 2^p this gives the old split
// (p even: n1 = n2; p odd: n1 = 2*n2). Returns p = log2 of the power-of-two part of N.
// Split N = n1 * n2 (n2 a power of two): the largest n2 with n1 >= n2.
static void bailey_split(size_t N, size_t& n1, size_t& n2) {
    n2 = 1;
    while ((N / (n2 * 2)) >= n2 * 2 && (N % (n2 * 2)) == 0) n2 *= 2;
    n1 = N / n2;
}

static unsigned bailey_setup(FFTContext& ctx, size_t& n1, size_t& n2) {
    size_t N = ctx.fft_len;
    unsigned p = 0;
    size_t temp_for_p = N;
    while ((temp_for_p & 1) == 0 && temp_for_p > 1) { temp_for_p >>= 1; p++; }

    bailey_split(N, n1, n2);

    if (ctx.r_w1r.size() != n1) {
        ctx.r_w1r.assign(n1, 0); ctx.r_w1i.assign(n1, 0);

        size_t tw_stride_col = n2;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n1; ++i) {
            size_t idx = i * tw_stride_col;
            ctx.r_w1r[i] = ctx.w1_re[idx]; ctx.r_w1i[i] = ctx.w1_im[idx];
        }
    }

    if (ctx.c_w1r.size() != n2) {
        ctx.c_w1r.assign(n2, 0); ctx.c_w1i.assign(n2, 0);

        size_t tw_stride_row = n1;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n2; ++i) {
            size_t idx = i * tw_stride_row;
            ctx.c_w1r[i] = ctx.w1_re[idx]; ctx.c_w1i[i] = ctx.w1_im[idx];
        }
    }

    if (ctx.tw2_N != N) {
        const unsigned sh = (p + 1) / 2;   // low table 2^sh entries, high table N / 2^sh entries (N divisible by 2^sh)
        const size_t L = size_t(1) << sh;
        const size_t H = N >> sh;
        ctx.tw2_lo_re.assign(L, 0); ctx.tw2_lo_im.assign(L, 0);
        ctx.tw2_hi_re.assign(H, 0); ctx.tw2_hi_im.assign(H, 0);
        for (size_t i = 0; i < L; ++i) { ctx.tw2_lo_re[i] = ctx.w1_re[i]; ctx.tw2_lo_im[i] = ctx.w1_im[i]; }
        for (size_t i = 0; i < H; ++i) { ctx.tw2_hi_re[i] = ctx.w1_re[i << sh]; ctx.tw2_hi_im[i] = ctx.w1_im[i << sh]; }
        ctx.tw2_shift = sh;
        ctx.tw2_N = N;
    }

    // lane_g[8r + b] = w^(r*b) for every row r of a gathered matrix: n1 rows (passes A, B') and n2 rows (the fusion),
    // so max(n1, n2) rows
    const size_t g_rows = std::max(n1, n2);
    if (ctx.lane_g_N != N || ctx.lane_g_re.size() != 8 * g_rows) {
        ctx.lane_g_re.assign(8 * g_rows, 0.0);
        ctx.lane_g_im.assign(8 * g_rows, 0.0);
        for (size_t r = 0; r < g_rows; ++r) {
            for (size_t b = 0; b < 8; ++b) {
                const size_t e = r * b;   // r < n1, b < 8: below N
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
    bailey_setup(ctx, n1, n2);
    if (!inverse) {
        // A: x (n1 rows x n2 columns) -> length-n1 FFT over columns -> Y1 (n2 rows x n1)
        { LOTR_PASS(ctx, 0); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, false, false, ctx, false, ctx.r_w1r, ctx.r_w1i); }
        // B: Y1 (n2 x n1) -> twiddle w^(b*k1), length-n2 FFT over columns, transposed store -> natural order
        { LOTR_PASS(ctx, 1); bailey_colpass(ctx.re2, ctx.im2, ctx.re, ctx.im, n2, n1, true, true, ctx, false, ctx.c_w1r, ctx.c_w1i); }
    }
    else {
        // A': spectrum X (n2 rows x n1 columns) -> length-n2 FFT over columns -> Y (n1 rows x n2)
        { LOTR_PASS(ctx, 0); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n2, n1, false, false, ctx, true, ctx.c_w1r, ctx.c_w1i); }
        // B': Y (n1 x n2) -> conjugate twiddle w^(k1*b), length-n1 FFT over columns, transposed store -> natural order
        { LOTR_PASS(ctx, 1); bailey_colpass(ctx.re2, ctx.im2, ctx.re, ctx.im, n1, n2, true, true, ctx, true, ctx.r_w1r, ctx.r_w1i); }
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
// instead of 3 passes over memory (store spectrum in natural order, square pass, read the spectrum again).
// Spectrum index k = k1 + n1*k2. The pair N-k is in column n1-k1 (k2 -> n2-1-k2); columns 0 and n1/2 pair with themselves.
// Block b: columns A = [8b, 8b+8) and mirrored M = [n1-8b-7, n1-8b]; lane t of block A <-> lane 7-t of block M.
// For b = 0: M = [n1-7, n1-1] + column n1/2 in lane 7; the self-paired columns (0 and n1/2) are computed separately in scalar code.
// Input src: Y1 after pass A (n2 rows x n1 columns). Output dst: layout as after A' (n1 rows x n2).
// =====================================================================
// FUSED MIDDLE in the batch-8 layout (vector = element k2 of eight columns), without transposes. The partner of column A (lane t, k2) is lane 7-t of block M at n2-1-k2, i.e.
// Z[N-k] = reversed vector M[n2-1-k2]. The self-paired columns (0 = lane 0 of block A, n1/2 = lane 7 of block M at b = 0)
// are computed in scalar code from copies.
// tw_inv: the result rows are stored already multiplied by the inverse twiddle conj(w^(j2*k1)) of pass B' - the same
// values the gather of this block computed (kept in 32 KB per block): pass B' then gathers without computing the
// twiddles a second time.
// twiddles (it computed them a second time). Measured before, 3*2^16: B' gather .0142 vs A's .0078 ms/iter.
static void bailey_fused_square_mid8(FFTContext& ctx, const double* __restrict src_re, const double* __restrict src_im,
                                     double* __restrict dst_re, double* __restrict dst_im, size_t n1, size_t n2,
                                     size_t sp = 0, size_t dp = 0, bool tw_inv = false) {
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
    SubAcc sub[64];
    int sub_nt = 1;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (tid == 0) sub_nt = omp_get_num_threads();
        SubAcc& acc = sub[tid < 64 ? tid : 63];
        (void)acc;
        // block A: arrays a, b, c (re, im) = scr[0 .. 6 BL), block M: scr[6 BL .. 12 BL), copies of the self-paired
        // columns: 4R (BL = b8_len(R): one re or im array of a block, two-level layout with block gaps)
        const size_t BL = b8_len(R);
        double* const scr = batch8_scratch(R, 2, 4 * R + 32 * R);
        double* const A_a_re = scr;           double* const A_a_im = scr + BL;
        double* const A_b_re = scr + 2 * BL;  double* const A_b_im = scr + 3 * BL;
        double* const A_c_re = scr + 4 * BL;  double* const A_c_im = scr + 5 * BL;
        double* const M_a_re = scr + 6 * BL;  double* const M_a_im = scr + 7 * BL;
        double* const M_b_re = scr + 8 * BL;  double* const M_b_im = scr + 9 * BL;
        double* const M_c_re = scr + 10 * BL; double* const M_c_im = scr + 11 * BL;
        double* const selfA_re = scr + 12 * BL;
        double* const selfA_im = selfA_re + R;
        double* const selfM_re = selfA_re + 2 * R;
        double* const selfM_im = selfA_re + 3 * R;
        // twiddles of the two blocks (tw_inv): 8 per row, natural row order
        double* const twA_re = tw_inv ? selfA_re + 4 * R : nullptr;
        double* const twA_im = tw_inv ? selfA_re + 12 * R : nullptr;
        double* const twM_re = tw_inv ? selfA_re + 20 * R : nullptr;
        double* const twM_im = tw_inv ? selfA_re + 28 * R : nullptr;

        const __m512i rev_idx = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512d v_s = _mm512_set1_pd(s);
        // R >= 512: two-level FFT, spectrum in the blocked layout -> address of vector k2 via P()
        const bool two = batch8_two_level(R);
        const size_t S = b8_S(R);
        auto P = [&](size_t K) -> size_t { return two ? batch8_posL(K, S) : K; };
        const bool ip = b8_inplace_mid(R);   // in-place two-level form: spectrum stays in a, buffer c = 16 KB scratch

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
            SUB_T(t0);
            batch8_gather(src_re, src_im, R, C, c0, false, 0, true, false, ctx, tw2, A_a_re, A_a_im, 0, true, 0, ~(size_t)0, ip,
                          twA_re, twA_im);
            batch8_gather(src_re, src_im, R, C, mstart, sp, spc, true, false, ctx, tw2, M_a_re, M_a_im, 0, true, 0, ~(size_t)0, ip,
                          twM_re, twM_im);
            SUB_T(t1);
            SUB_ADD(acc.c[0], t0, t1);
            bool A_in_b = true, M_in_b = true;
            if (ip) {
                b8_ip_dif(A_a_re, A_a_im, A_c_re, A_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false, false);
                b8_ip_dif(M_a_re, M_a_im, M_c_re, M_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false, false);
                A_in_b = M_in_b = false;
            }
            else if (two) {
                fft_batch8_2l(A_a_re, A_a_im, false, A_b_re, A_b_im, A_c_re, A_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
                fft_batch8_2l(M_a_re, M_a_im, false, M_b_re, M_b_im, M_c_re, M_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
            }
            else {
                A_in_b = fft_batch8(A_a_re, A_a_im, A_b_re, A_b_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
                M_in_b = fft_batch8(M_a_re, M_a_im, M_b_re, M_b_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
            }
            double* const XA_re = A_in_b ? A_b_re : A_a_re; double* const XA_im = A_in_b ? A_b_im : A_a_im;
            double* const XM_re = M_in_b ? M_b_re : M_a_re; double* const XM_im = M_in_b ? M_b_im : M_a_im;
            SUB_T(t2);
            SUB_ADD(acc.c[1], t1, t2);

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

            SUB_T(t3);
            SUB_ADD(acc.c[2], t2, t3);

            // ---- inv A': length-n2 FFT^-1 of both blocks and row store dst[col*n2 + c] ----
            if (ip) {
                // blocked spectrum in a -> gapped natural result in a (DIT), buffer c as scratch
                b8_ip_ditf(A_a_re, A_a_im, A_c_re, A_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                SUB_T(t4i);
                batch8_store_rows(A_a_re, A_a_im, R, cols, dst_re, dst_im, false, DP, 0, ~(size_t)0, true, twA_re, twA_im);
                SUB_T(t5i);
                b8_ip_ditf(M_a_re, M_a_im, M_c_re, M_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                SUB_T(t6i);
                batch8_store_rows(M_a_re, M_a_im, R, cols + 8, dst_re, dst_im, false, DP, 0, ~(size_t)0, true, twM_re, twM_im);
                SUB_T(t7i);
                SUB_ADD(acc.c[3], t3, t4i); SUB_ADD(acc.c[3], t5i, t6i);
                SUB_ADD(acc.c[4], t4i, t5i); SUB_ADD(acc.c[4], t6i, t7i);
            }
            else if (two) {
                // spectrum in the blocked layout (b) -> result in the blocked layout (a), buffer c as scratch
                fft_batch8_2l(A_b_re, A_b_im, true, A_a_re, A_a_im, A_c_re, A_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                SUB_T(t4a);
                batch8_store_rows(A_a_re, A_a_im, R, cols, dst_re, dst_im, true, DP, 0, ~(size_t)0, false, twA_re, twA_im);
                SUB_T(t5a);
                fft_batch8_2l(M_b_re, M_b_im, true, M_a_re, M_a_im, M_c_re, M_c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                SUB_T(t6a);
                batch8_store_rows(M_a_re, M_a_im, R, cols + 8, dst_re, dst_im, true, DP, 0, ~(size_t)0, false, twM_re, twM_im);
                SUB_T(t7a);
                SUB_ADD(acc.c[3], t3, t4a); SUB_ADD(acc.c[3], t5a, t6a);
                SUB_ADD(acc.c[4], t4a, t5a); SUB_ADD(acc.c[4], t6a, t7a);
            }
            else {
                {
                    double* const y_re = A_in_b ? A_a_re : A_b_re; double* const y_im = A_in_b ? A_a_im : A_b_im;
                    const bool in_y = fft_batch8(XA_re, XA_im, y_re, y_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                    SUB_T(t4b);
                    batch8_store_rows(in_y ? y_re : XA_re, in_y ? y_im : XA_im, R, cols, dst_re, dst_im, false, DP, 0, ~(size_t)0,
                                      false, twA_re, twA_im);
                    SUB_T(t5b);
                    SUB_ADD(acc.c[3], t3, t4b); SUB_ADD(acc.c[4], t4b, t5b);
                }
                {
                    SUB_T(t6b);
                    double* const y_re = M_in_b ? M_a_re : M_b_re; double* const y_im = M_in_b ? M_a_im : M_b_im;
                    const bool in_y = fft_batch8(XM_re, XM_im, y_re, y_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                    SUB_T(t7b);
                    batch8_store_rows(in_y ? y_re : XM_re, in_y ? y_im : XM_im, R, cols + 8, dst_re, dst_im, false, DP, 0, ~(size_t)0,
                                      false, twM_re, twM_im);
                    SUB_T(t8b);
                    SUB_ADD(acc.c[3], t6b, t7b); SUB_ADD(acc.c[4], t7b, t8b);
                }
            }
        }
    }
#ifdef LOTR_PROFILE
    sub_merge(ctx, 5, 5, sub, sub_nt);
#endif
}

// Can the square use the fused middle (two-pass Bailey for this FFT length); sets n1, n2 and the twiddle tables.
// FFT length 3*2^p (radix-3 in the batch-8 column passes)?
static bool is_len_3x2p(size_t len) {
    return len >= 3 && (len % 3) == 0 && ((len / 3) & (len / 3 - 1)) == 0;
}
// FFT length 7*2^p (radix-7 in the batch-8 column passes: e.g. 7*2^15 = 229376 between 3*2^16 and 2^18,
// N = 4.19M with 19-bit digits instead of 2^18). The column length n1 = 7*2^q, the rows n2 stay a power of two.
// Classic real-FFT square only (not the IBDWT).
static bool is_len_7x2p(size_t len) {
    return len >= 7 && (len % 7) == 0 && ((len / 7) & (len / 7 - 1)) == 0;
}
// 3*2^p or 7*2^p: only through the fused two-pass Bailey with batch-8 kernels
static bool is_len_mixed(size_t len) { return is_len_3x2p(len) || is_len_7x2p(len); }

static bool bailey_fused_ok(FFTContext& ctx, size_t& n1, size_t& n2) {
    const bool r3 = is_len_mixed(ctx.fft_len);
    if (!r3) {
        for (size_t f : ctx.factor_plan.factors) {
            if (f != 2) return false;
        }
    }
    unsigned p = 0; size_t temp = ctx.fft_len;
    while (temp > 1) { temp >>= 1; p++; }
    if (!r3 && !use_bailey_engine(p)) return false;
    bailey_setup(ctx, n1, n2);
    // 3*2^p: n2 >= 128, so the padded rows (8 doubles per row) fit the fft_len/16 headroom of fft_init
    if (r3 && n2 < 128) return false;
    return (n1 % 16 == 0) && (n2 % 8 == 0) && (n2 >= 16);
}

// Lengths the fast squaring path handles: powers of two >= 256 and 3*2^p / 7*2^p, when the fused
// two-pass Bailey with batch-8 kernels is available for them (dcore.cpp asks before calibrating a length).
// Lengths the IBDWT square can use: those with the fused two-pass Bailey (A | middle | B'): 3*2^p as above,
// powers of two where the Bailey engine runs two-pass (not the Stockham ones), min(n1, n2) >= 128.
bool fft_ibdwt_supported(size_t len) {
    if (!fft_len_supported(len)) return false;
    if (is_len_7x2p(len)) return false;   // radix-7: classic square only (not checked with the IBDWT)
    size_t n1, n2;
    bailey_split(len, n1, n2);
    if (n1 < 128 || n2 < 128 || (n1 % 16) != 0) return false;
    if ((len & (len - 1)) == 0) {
        unsigned p = 0;
        for (size_t t = len; t > 1; t >>= 1) ++p;
        return use_bailey_engine(p);
    }
    return true;
}

bool fft_len_supported(size_t len) {
    if (len < 256) return false;
    if ((len & (len - 1)) == 0) return true;
    if (!is_len_mixed(len)) return false;
    size_t n1, n2;
    bailey_split(len, n1, n2);
    return (n1 % 16 == 0) && (n2 >= 128) && (n1 >= 128);   // room for the padded rows
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

        fft_run_inplace(ctx, inverse);

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
// Powers of two only (3*2^p / 7*2^p: the fused square): Bailey 2D or Stockham 1D (use_bailey_engine).
static void fft_run_inplace(FFTContext& ctx, bool inverse) {
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
// The 3-FFT square (1 forward + 2 inverse: A^2 + iB^2 and AB, fft_square_karatsuba3) stays as the fallback (self-test).
// =====================================================================
static void fft_square_real(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    // Two-pass Bailey: fwd A | fusion (fwd B + square + inv A') | inv B'. In the profile: fwd = A, square = fusion, inv = B'.
    size_t n1 = 0, n2 = 0;
    const bool fused = bailey_fused_ok(ctx, n1, n2);
    if (!fused && (ctx.fft_len & (ctx.fft_len - 1)) != 0) {
        // 3*2^p exists only in the fused batch-8 path (the plain FFT below handles powers of two only)
        static std::atomic<bool> warned{ false };
        if (!warned.exchange(true))
            std::fprintf(stderr, "[ERROR] FFT length %zu needs the fused two-pass Bailey\n", ctx.fft_len);
        ctx.last_max_diff = 1.0e30;
        return;
    }

    // Row padding (8 doubles = 64 B) in the matrices between passes: rows whose length is a power of two (e.g. 4 KB)
    // map to the same L1 set (conflicts when gathering columns and in transposed stores).
    // Only with min(n1, n2) >= 128: fft_init leaves fft_len/16 of headroom per array, the padded matrix needs 8 doubles
    // per row (n1*(n2+8) or n2*(n1+8)); smaller matrices (e.g. 64 x 64) would need more than the headroom.
    const size_t pad = (fused && std::min(n1, n2) >= 128) ? 8 : 0;
    ctx.lay_pad = pad;
    if (pad) {
        unsigned lg = 0; while ((size_t(1) << lg) < n2) ++lg;
        ctx.lay_lg = lg;   // a row of the x and result layout = n2 elements
    }

    // The digits fill only the lower half (16-bit: j < 2*n_limbs, n_limbs = fft_len/4; wide digits: real_pack_used);
    // the upper half is room for the square.
    // Padded path: pass A reads only the rows of x that hold digits (rounded up to n1/2, so its first radix-8 stage
    // skips the zero half too), and pack does not zero the rows A never reads.
    size_t rows_nz = 0;
    if (pad) {
        const size_t j_used = real_pack_used(n_limbs, ctx.digit_bits);   // complex positions holding digits
        const size_t rows_used = (j_used + n2 - 1) / n2;                // x = n1 rows of n2 elements
        rows_nz = (rows_used <= n1 / 2) ? n1 / 2 : rows_used;
        if (rows_nz >= n1) rows_nz = 0;                          // the whole matrix is used: nothing to skip
    }
    ctx.pack_zero_end = rows_nz * n2;

    // Pack fused into pass A (padded fused path): only y = x + K here, pass A takes the digits of its columns straight
    // from y instead of reading the unpacked re/im matrix written by the pack.
    // Only with a two-level pass A (n1 >= 256): the lengths that were tested this way (a precaution; negligible cost at
    // the small lengths).
    const bool pack_in_a = (pad != 0) && batch8_two_level(n1);
    DigitSrc dsrc;

    auto t_pack_0 = clock::now();
    if (pack_in_a) pack_prepare_digits(v, n_limbs, ctx, dsrc);
    else           direct_pack_square_real(v, n_limbs, ctx);
    ctx.pack_zero_end = 0;
    auto t_pack_1 = clock::now();

    if (fused) {
        auto t0 = clock::now();
        if (pad) {
            // A: x (n1 rows of n2, stride n2+pad) -> Y1 (n2 rows of n1, stride n1+pad); rows >= rows_nz are zero (not read)
            { LOTR_PASS(ctx, 0); bailey_colfft_pass8(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, false, false, ctx, false,
                ctx.r_w1r, ctx.r_w1i, n2 + pad, n1 + pad, rows_nz, 0, pack_in_a ? &dsrc : nullptr); }
        }
        else {
            { LOTR_PASS(ctx, 0); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, false, false, ctx, false,
                ctx.r_w1r, ctx.r_w1i); }
        }
        auto t1 = clock::now();
        // twiddles of the gathers from a base every 8 rows (batch8_gather, ctx.tw_fast)
        ctx.tw_fast = true;
        {
            LOTR_PASS(ctx, 1);
            // fusion: Y1 (stride n1+pad) -> n1 rows of n2 (stride n2+pad)
            bailey_fused_square_mid8(ctx, ctx.re2, ctx.im2, ctx.re, ctx.im, n1, n2, pad ? n1 + pad : 0, pad ? n2 + pad : 0);
        }
        auto t2 = clock::now();
        // B' in place: every block (or coop group) stores its columns back to the positions it has just gathered from re/im
        // (lines already in this core's cache) instead of into re2, whose lines the middle's cores read last.
        const bool bp_inplace = pad != 0;
        if (pad) {
            // B': n1 rows of n2 (stride n2+pad) -> result in the x layout (n1 rows of n2, stride n2+pad)
            { LOTR_PASS(ctx, 2); bailey_colfft_pass8(ctx.re, ctx.im, bp_inplace ? ctx.re : ctx.re2, bp_inplace ? ctx.im : ctx.im2,
                n1, n2, true, true, ctx, true, ctx.r_w1r, ctx.r_w1i, n2 + pad, n2 + pad, 0, 11); }
        }
        else {
            { LOTR_PASS(ctx, 2); bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, true, true, ctx, true,
                ctx.r_w1r, ctx.r_w1i); }
        }
        ctx.tw_fast = false;
        if (!bp_inplace) {   // the result is in re2/im2: the unpack reads re/im
            std::swap(ctx.re, ctx.re2);
            std::swap(ctx.im, ctx.im2);
        }
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

// =====================================================================
// IBDWT SQUARE for k*2^n+1 (no zero padding), in the scaled form: v = Y  ->  -k*Y^2 mod N.
// Checked against a reference with exact integers (also 20-step chains). Steps:
//   setup (per k, n): digit positions e_j = ceil(j*beta), beta = (n + log2 k)/L (exact, 128-bit fixed point),
//                     weights a_j = 2^(e_j - j*beta) in [1, 2), twist theta^j = exp(i*pi*j/L), L = 2*fft_len
//   pack:   local balanced digits d_j of Y (top digit not balanced),
//           z_j = (a_j d_j + i a_(j+N) d_(j+N)) * theta^j  (j < N = fft_len) into the x layout
//   FFT:    pass A | complex fused middle (fwd B, Z^2 / N, inv A') | pass B'   (A and B' as in the real square)
//   unpack: c_j = out_j / theta^j, w_m = round(k * c_m / a_m) (the outputs are in (1/k)Z), W = sum w_m 2^e_m == k*Y^2,
//           carries over the L digits; the carry C out of the top sits at 2^(e_L) = 2^d * 2^n and
//           C*2^d*2^n == -q + r*2^n  (C*2^d = q*k + r): -q at digit 0, r*2^(n - e_m) at the digit m holding bit n;
//           digits -> limbs V < 2^(e_L) < 2N, V mod N, R = (N - V) mod N.
// LOTR_IBDWT=1 with LOTR_FORCE_FFT_LEN selects it (dcore.cpp).
// =====================================================================

// floor(log2(k) * 2^128): integer part ip and the 128-bit fraction (f_hi, f_lo), by repeated squaring of k / 2^m.
static void log2_fixed128(uint64_t k, uint64_t& ip, uint64_t& f_hi, uint64_t& f_lo) {
    const unsigned m = 63 - (unsigned)__builtin_clzll(k);
    const unsigned P = 128 + 64;   // fractional bits of y (64 guard bits)
    mpz_t y, two;
    mpz_inits(y, two, nullptr);
    mpz_set_ui(y, (unsigned long)k);
    mpz_mul_2exp(y, y, P);
    mpz_fdiv_q_2exp(y, y, m);      // y = k / 2^m in [1, 2)
    mpz_set_ui(two, 1);
    mpz_mul_2exp(two, two, P + 1); // 2.0
    unsigned __int128 f = 0;
    for (int i = 0; i < 128; ++i) {
        mpz_mul(y, y, y);
        mpz_fdiv_q_2exp(y, y, P);
        f <<= 1;
        if (mpz_cmp(y, two) >= 0) { mpz_fdiv_q_2exp(y, y, 1); f |= 1; }
    }
    mpz_clears(y, two, nullptr);
    ip = m;
    f_hi = (uint64_t)(f >> 64);
    f_lo = (uint64_t)f;
}

// Tables for (k, n); false if the layout is not usable (digit sizes out of range).
static bool ibdwt_setup(FFTContext& ctx, unsigned n, uint64_t k) {
    if (ctx.ib_k == k && ctx.ib_n == n && !ctx.ib_e.empty()) return true;
    const size_t N = ctx.fft_len, L = 2 * N;
    uint64_t ip, fh, fl;
    log2_fixed128(k, ip, fh, fl);
    // beta * 2^128 = floor(((n + ip) * 2^128 + frac) / L) = q0 * 2^128 + F
    mpz_t num, t;
    mpz_inits(num, t, nullptr);
    mpz_set_ui(num, (unsigned long)(n + ip));
    mpz_mul_2exp(num, num, 64);
    mpz_add_ui(num, num, (unsigned long)fh);
    mpz_mul_2exp(num, num, 64);
    mpz_add_ui(num, num, (unsigned long)fl);
    mpz_fdiv_q_ui(num, num, (unsigned long)L);
    mpz_fdiv_q_2exp(t, num, 128);
    const uint64_t q0 = (uint64_t)mpz_get_ui(t);
    mpz_fdiv_r_2exp(t, num, 128);
    const uint64_t F_lo = (uint64_t)mpz_get_ui(t);
    mpz_fdiv_q_2exp(t, t, 64);
    const uint64_t F_hi = (uint64_t)mpz_get_ui(t);
    mpz_clears(num, t, nullptr);
    const unsigned __int128 F = ((unsigned __int128)F_hi << 64) | F_lo;

    // The full weight / twist tables are read only by the reference unpack (LOTR_VERIFY_UNPACK=1); the run uses the
    // two-level tables below. Not built otherwise: 3*L doubles = 2.3 GB on the giant 195*2^1073000000+1 (FFT 3*2^24).
    const bool full = (std::getenv("LOTR_VERIFY_UNPACK") != nullptr);
    // j*beta = ipart + acc/2^128 incrementally; e_j = ceil(j*beta), a_j = 2^(e_j - j*beta) from the same acc (consistent)
    ctx.ib_e.assign(L + 1, 0);
    if (full) {
        ctx.ib_a.assign(L, 1.0);
        ctx.ib_kia.assign(L, (double)k);
    }
    unsigned __int128 acc = 0;
    uint64_t ipart = 0;
    for (size_t j = 0; j <= L; ++j) {
        if (j > 0) {
            const unsigned __int128 old = acc;
            acc += F;
            ipart += q0 + (acc < old ? 1 : 0);
        }
        const uint64_t e = ipart + (acc != 0 ? 1 : 0);
        if (e > 0xFFFFFFFFull) return false;
        ctx.ib_e[j] = (uint32_t)e;
        if (full && j < L && acc != 0) {
            const double fr = (double)(uint64_t)(acc >> 64) * 0x1p-64;   // fraction of j*beta
            const double a = std::exp2(1.0 - fr);
            ctx.ib_a[j] = a;
            ctx.ib_kia[j] = (double)k / a;
        }
    }
    for (size_t j = 0; j < L; ++j) {
        const uint32_t b = ctx.ib_e[j + 1] - ctx.ib_e[j];
        if (b < 2 || b > 30) return false;   // the digit state is int32 (ctx.ib_w)
    }
    const uint32_t eL = ctx.ib_e[L];
    if (eL <= n || eL - n > 64) return false;
    ctx.ib_d = eL - n;
    size_t m = 0;
    while (m + 1 < L && ctx.ib_e[m + 1] <= n) ++m;
    ctx.ib_m = m;
    ctx.ib_msh = n - ctx.ib_e[m];

    if (full) {
        ctx.ib_tw_re.assign(N, 0.0);
        ctx.ib_tw_im.assign(N, 0.0);
        for (size_t j = 0; j < N; ++j) {
            const double ang = M_PI * (double)j / (double)L;
            ctx.ib_tw_re[j] = std::cos(ang);
            ctx.ib_tw_im[j] = std::sin(ang);
        }
    }

    // Two-level tables (digit j = 64h + l): (64h)*beta and l*beta as integer part + 128-bit fraction, exactly
    // like the incremental acc above (j*F mod 2^128 = 64h*F + l*F mod 2^128), so e_j matches ib_e bit for bit.
    {
        const size_t NH = L / 64 + 1;
        ctx.ib_AHlo.assign(NH, 0); ctx.ib_AHhi.assign(NH, 0); ctx.ib_IH.assign(NH, 0);
        ctx.ib_WH.assign(NH, 1.0); ctx.ib_iWH.assign(NH, 1.0);
        ctx.ib_ALlo.assign(64, 0); ctx.ib_ALhi.assign(64, 0); ctx.ib_IL.assign(64, 0);
        ctx.ib_WL.assign(64, 2.0); ctx.ib_iWL.assign(64, 0.5);
        const unsigned __int128 F64 = F << 6;                                    // 64*F mod 2^128
        const uint64_t q64 = 64 * q0 + (uint64_t)(F >> 122);                     // integer part of 64*beta
        unsigned __int128 ah = 0;
        uint64_t ih = 0;
        for (size_t h = 0; h < NH; ++h) {
            if (h > 0) {
                const unsigned __int128 old = ah;
                ah += F64;
                ih += q64 + (ah < old ? 1 : 0);
            }
            ctx.ib_AHlo[h] = (uint64_t)ah;
            ctx.ib_AHhi[h] = (uint64_t)(ah >> 64);
            ctx.ib_IH[h] = ih;
            const double fh = (double)(uint64_t)(ah >> 64) * 0x1p-64;
            ctx.ib_WH[h] = std::exp2(-fh);
            ctx.ib_iWH[h] = std::exp2(fh);
        }
        unsigned __int128 al = 0;
        uint64_t il = 0;
        for (size_t l = 0; l < 64; ++l) {
            if (l > 0) {
                const unsigned __int128 old = al;
                al += F;
                il += q0 + (al < old ? 1 : 0);
            }
            ctx.ib_ALlo[l] = (uint64_t)al;
            ctx.ib_ALhi[l] = (uint64_t)(al >> 64);
            ctx.ib_IL[l] = il;
            const double fl = (double)(uint64_t)(al >> 64) * 0x1p-64;
            ctx.ib_WL[l] = std::exp2(1.0 - fl);
            ctx.ib_iWL[l] = std::exp2(fl - 1.0);
        }
        const size_t NT = N / 64 + 1;
        ctx.ib_THr.assign(NT, 1.0); ctx.ib_THi.assign(NT, 0.0);
        ctx.ib_TLr.assign(64, 1.0); ctx.ib_TLi.assign(64, 0.0);
        for (size_t h = 0; h < NT; ++h) {
            const double ang = M_PI * (double)(64 * h) / (double)L;
            ctx.ib_THr[h] = std::cos(ang);
            ctx.ib_THi[h] = std::sin(ang);
        }
        for (size_t l = 0; l < 64; ++l) {
            const double ang = M_PI * (double)l / (double)L;
            ctx.ib_TLr[l] = std::cos(ang);
            ctx.ib_TLi[l] = std::sin(ang);
        }
        // consistency with the full table (positions must agree exactly); per digit byte: size b_j | c_j << 7
        ctx.ib_b.assign(L + 64, 0);
        for (size_t j = 0; j <= L; ++j) {
            const size_t h = j >> 6, l = j & 63;
            const uint64_t lo = ctx.ib_AHlo[h] + ctx.ib_ALlo[l];
            const uint64_t c1 = (lo < ctx.ib_AHlo[h]) ? 1 : 0;
            const uint64_t hr = ctx.ib_AHhi[h] + ctx.ib_ALhi[l];
            const uint64_t ov1 = (hr < ctx.ib_AHhi[h]) ? 1 : 0;
            const uint64_t hi = hr + c1;
            const uint64_t ov2 = (c1 && hi == 0) ? 1 : 0;
            const uint64_t ej = ctx.ib_IH[h] + ctx.ib_IL[l] + ov1 + ov2 + (((lo | hi) != 0) ? 1 : 0);
            if (ej != ctx.ib_e[j]) {
                std::fprintf(stderr, "[ERROR] IBDWT two-level positions differ at j=%zu (%llu vs %u)\n",
                             j, (unsigned long long)ej, ctx.ib_e[j]);
                return false;
            }
            if (j < L) ctx.ib_b[j] = (uint8_t)((ctx.ib_e[j + 1] - ctx.ib_e[j]) | ((ov1 | ov2) << 7));
        }
    }
    ctx.ib_k = k;
    ctx.ib_n = n;
    return true;
}

// Position e_j from the two-level tables (scalar; the vector code does the same per lane).
static inline uint64_t ib_e_at(const FFTContext& ctx, size_t j) {
    const size_t h = j >> 6, l = j & 63;
    const uint64_t lo = ctx.ib_AHlo[h] + ctx.ib_ALlo[l];
    const uint64_t c1 = (lo < ctx.ib_AHlo[h]) ? 1 : 0;
    const uint64_t hr = ctx.ib_AHhi[h] + ctx.ib_ALhi[l];
    const uint64_t ov1 = (hr < ctx.ib_AHhi[h]) ? 1 : 0;
    const uint64_t hi = hr + c1;
    const uint64_t ov2 = (c1 && hi == 0) ? 1 : 0;
    return ctx.ib_IH[h] + ctx.ib_IL[l] + ov1 + ov2 + (((lo | hi) != 0) ? 1 : 0);
}

// 8 consecutive digits j0 .. j0+7 (j0 a multiple of 8, one 64-group): positions e (lanes), carry c of the fraction
// sum and nz (fraction != 0) as masks.
static inline __attribute__((always_inline)) void ib_pos8(const FFTContext& ctx, size_t j0, __m512i& e, __mmask8& c, __mmask8& nz) {
    const size_t h = j0 >> 6, l = j0 & 63;
    const __m512i ahlo = _mm512_set1_epi64((long long)ctx.ib_AHlo[h]);
    const __m512i ahhi = _mm512_set1_epi64((long long)ctx.ib_AHhi[h]);
    const __m512i allo = _mm512_loadu_si512((const void*)(ctx.ib_ALlo.data() + l));
    const __m512i alhi = _mm512_loadu_si512((const void*)(ctx.ib_ALhi.data() + l));
    const __m512i lo = _mm512_add_epi64(ahlo, allo);
    const __mmask8 c1 = _mm512_cmplt_epu64_mask(lo, ahlo);
    const __m512i hr = _mm512_add_epi64(ahhi, alhi);
    const __mmask8 ov1 = _mm512_cmplt_epu64_mask(hr, ahhi);
    const __m512i hi = _mm512_mask_add_epi64(hr, c1, hr, _mm512_set1_epi64(1));
    const __mmask8 ov2 = c1 & _mm512_cmpeq_epi64_mask(hi, _mm512_setzero_si512());
    c = ov1 | ov2;
    nz = _mm512_test_epi64_mask(_mm512_or_si512(lo, hi), _mm512_or_si512(lo, hi));
    __m512i ev = _mm512_add_epi64(_mm512_set1_epi64((long long)ctx.ib_IH[h]),
                                  _mm512_loadu_si512((const void*)(ctx.ib_IL.data() + l)));
    ev = _mm512_mask_add_epi64(ev, c, ev, _mm512_set1_epi64(1));
    ev = _mm512_mask_add_epi64(ev, nz, ev, _mm512_set1_epi64(1));
    e = ev;
}

// Top bit h of digit j of x (scalar; the top digit L-1 has h = 0).
static inline int64_t ib_hbit(const FFTContext& ctx, const uint64_t* x, size_t j) {
    if (j + 1 >= 2 * ctx.fft_len) return 0;
    const uint64_t pos = ib_e_at(ctx, j);
    const unsigned b = (unsigned)(ib_e_at(ctx, j + 1) - pos);
    const size_t li = pos >> 6;
    const unsigned sh = pos & 63;
    uint64_t f = x[li] >> sh;
    if (sh != 0) f |= x[li + 1] << (64 - sh);
    return (int64_t)((f >> (b - 1)) & 1);
}

// Digits j0 .. j0+7 of x (j0 a multiple of 8) as local balanced digits times the weights a_j. h_prev = top bit of
// digit j0-1 (in), *h_last = top bit of digit j0+7 (out); top_last: digit j0+7 is the top digit (h = 0, not balanced).
// The limbs x[e_j/64 + 1] must exist (n_limbs*64 >= n + 192).
static inline __attribute__((always_inline)) __m512d ib_digits8_w(const FFTContext& ctx, const uint64_t* x, size_t j0,
                                                                  int64_t h_prev, int64_t* h_last, bool top_last) {
    __m512i e; __mmask8 c, nz;
    ib_pos8(ctx, j0, e, c, nz);
    const int64_t e8 = (int64_t)ib_e_at(ctx, j0 + 8);
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i en = _mm512_alignr_epi64(_mm512_set1_epi64(e8), e, 1);   // e_(j+1)
    const __m512i b = _mm512_sub_epi64(en, e);
    const __m512i li = _mm512_srli_epi64(e, 6);
    const __m512i sh = _mm512_and_si512(e, _mm512_set1_epi64(63));
    const __m512i lo = _mm512_i64gather_epi64(li, (const void*)x, 8);
    const __m512i hi = _mm512_i64gather_epi64(_mm512_add_epi64(li, one), (const void*)x, 8);
    // shift counts >= 64 give 0 (sh = 0: nothing from hi)
    __m512i f = _mm512_or_si512(_mm512_srlv_epi64(lo, sh), _mm512_sllv_epi64(hi, _mm512_sub_epi64(_mm512_set1_epi64(64), sh)));
    f = _mm512_and_si512(f, _mm512_sub_epi64(_mm512_sllv_epi64(one, b), one));
    __m512i hb = _mm512_and_si512(_mm512_srlv_epi64(f, _mm512_sub_epi64(b, one)), one);
    if (top_last) hb = _mm512_mask_mov_epi64(hb, 0x80, _mm512_setzero_si512());
    const __m512i hp = _mm512_alignr_epi64(hb, _mm512_set1_epi64(h_prev), 7);   // lane 0 = h_prev, lane i = hb[i-1]
    const __m512i d = _mm512_add_epi64(_mm512_sub_epi64(f, _mm512_sllv_epi64(hb, b)), hp);
    alignas(64) int64_t hbs[8];
    _mm512_store_si512((void*)hbs, hb);
    *h_last = hbs[7];
    const size_t h = j0 >> 6, l = j0 & 63;
    __m512d wv = _mm512_mul_pd(_mm512_set1_pd(ctx.ib_WH[h]), _mm512_loadu_pd(ctx.ib_WL.data() + l));
    wv = _mm512_mask_mul_pd(wv, c, wv, _mm512_set1_pd(2.0));
    wv = _mm512_mask_mov_pd(_mm512_set1_pd(1.0), nz, wv);   // j = 0: a = 1
    return _mm512_mul_pd(_mm512_cvtepi64_pd(d), wv);
}

// Pack: local balanced digits of v and the weighted, twisted complex input in the x layout (row stride n2 + pad).
// 8 complex positions per step (vector digits from the two-level tables), the threads take ranges of positions.
static void ibdwt_pack(FFTContext& ctx, const uint64_t* v, size_t n_limbs) {
    (void)n_limbs;
    const size_t N = ctx.fft_len, L = 2 * N;
    double* const re = ctx.re;
    double* const im = ctx.im;
    const size_t pad = ctx.lay_pad;
    const unsigned lg = ctx.lay_lg;
#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const size_t cb = 8 * ((N / 8) * (size_t)t / (size_t)nt), ce = 8 * ((N / 8) * (size_t)(t + 1) / (size_t)nt);
        int64_t hl = (cb == 0) ? 0 : ib_hbit(ctx, v, cb - 1);   // top bit of the digit before (lower stream)
        int64_t hu = ib_hbit(ctx, v, N + cb - 1);                // (upper stream; digit N-1 before N)
        for (size_t j0 = cb; j0 < ce; j0 += 8) {
            const __m512d x0 = ib_digits8_w(ctx, v, j0, hl, &hl, false);
            const __m512d x1 = ib_digits8_w(ctx, v, N + j0, hu, &hu, N + j0 + 8 == L);
            const size_t h = j0 >> 6, l = j0 & 63;
            const __m512d Hr = _mm512_set1_pd(ctx.ib_THr[h]), Hi = _mm512_set1_pd(ctx.ib_THi[h]);
            const __m512d Lr = _mm512_loadu_pd(ctx.ib_TLr.data() + l), Li = _mm512_loadu_pd(ctx.ib_TLi.data() + l);
            const __m512d tr = _mm512_fmsub_pd(Hr, Lr, _mm512_mul_pd(Hi, Li));   // theta^j = TH*TL
            const __m512d ti = _mm512_fmadd_pd(Hr, Li, _mm512_mul_pd(Hi, Lr));
            const size_t p = j0 + (j0 >> lg) * pad;
            _mm512_store_pd(re + p, _mm512_fmsub_pd(x0, tr, _mm512_mul_pd(x1, ti)));
            _mm512_store_pd(im + p, _mm512_fmadd_pd(x0, ti, _mm512_mul_pd(x1, tr)));
        }
    }
}

// Pack from the digit state of the previous square (ctx.ib_w: normalized digits in [0, 2^b_j), ctx.ib_b: b_j):
// the fields are the stored digits themselves, so only the local balancing, weights and twist remain.
static void ibdwt_pack_digits(FFTContext& ctx) {
    const size_t N = ctx.fft_len, L = 2 * N;
    const int32_t* const w = ctx.ib_w.data();
    const uint8_t* const bsz = ctx.ib_b.data();
    double* const re = ctx.re;
    double* const im = ctx.im;
    const size_t pad = ctx.lay_pad;
    const unsigned lg = ctx.lay_lg;
#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        // ranges of whole groups of 8 runs (64 complex positions)
        const size_t cb = 64 * ((N / 64) * (size_t)t / (size_t)nt), ce = 64 * ((N / 64) * (size_t)(t + 1) / (size_t)nt);
        int64_t hl = (cb == 0) ? 0 : ib_state_hbit(w, bsz, cb - 1, L);
        int64_t hu = ib_state_hbit(w, bsz, N + cb - 1, L);
        for (size_t g0 = cb; g0 < ce; g0 += 64) {
            size_t jl[8], ju[8];
            for (int r = 0; r < 8; ++r) { jl[r] = g0 + 8 * (size_t)r; ju[r] = N + jl[r]; }
            __m512i v[8], f[8], b[8];
            __mmask8 c[8];
            __m512d x0[8], x1[8];
            ib_runs8(ctx, w, bsz, jl, v, f, b, c);
            for (int r = 0; r < 8; ++r) x0[r] = ib_digits_w(ctx, jl[r], v[r], f[r], b[r], c[r], hl, &hl, false);
            ib_runs8(ctx, w, bsz, ju, v, f, b, c);
            for (int r = 0; r < 8; ++r) x1[r] = ib_digits_w(ctx, ju[r], v[r], f[r], b[r], c[r], hu, &hu, ju[r] + 8 == L);
            for (int r = 0; r < 8; ++r) {
                __m512d tr, ti;
                ib_twist8(ctx, jl[r], tr, ti);
                const size_t p = jl[r] + (jl[r] >> lg) * pad;
                _mm512_store_pd(re + p, _mm512_fmsub_pd(x0[r], tr, _mm512_mul_pd(x1[r], ti)));
                _mm512_store_pd(im + p, _mm512_fmadd_pd(x0[r], ti, _mm512_mul_pd(x1[r], tr)));
            }
        }
    }
}

// Complex fused middle: per block of 8 columns of Y1 (n2 rows of n1, stride sp): twiddle + length-n2 FFT (fwd B),
// Z -> Z^2 / N, inverse length-n2 FFT (A'), rows to dst (n1 rows of n2, stride dp). No pairing of k and N-k
// (that belongs to the real-FFT square), every block is independent.
// nat_src (in-place layout): src is the natural matrix x (n1 rows of n2, row stride sp), block cb = its rows 8cb ..
// 8cb+7 (batch8_gather_nat); dst may then be src (each block reads its 8 rows completely before writing them back).
static void bailey_fused_cplx_mid8(FFTContext& ctx, const double* src_re, const double* src_im,
                                   double* dst_re, double* dst_im, size_t n1, size_t n2,
                                   size_t sp, size_t dp, bool nat_src = false) {
    const size_t R = n2;
    const size_t C = (sp != 0) ? sp : n1;
    const size_t DP = (dp != 0) ? dp : n2;
    const Tw2 tw2(ctx);
    const __m512d v_s = _mm512_set1_pd(1.0 / (double)ctx.fft_len);
    const bool two = batch8_two_level(R);
    const size_t S = b8_S(R);
    const size_t BL = b8_len(R);
    const bool ip = b8_inplace_mid(R);   // in-place two-level form: spectrum stays in a, buffer c = scratch
    // profile (LOTR_PROFILE): [SUBPASS] mid = gather / fwd FFT / square / inv FFT / store
    SubAcc sub[64];
    int sub_nt = 1;
    (void)sub; (void)sub_nt;
#pragma omp parallel
    {
        double* const scr = batch8_scratch(R, 1, 0);
        double* const a_re = scr;          double* const a_im = scr + BL;
        double* const b_re = scr + 2 * BL; double* const b_im = scr + 3 * BL;
        double* const c_re = scr + 4 * BL; double* const c_im = scr + 5 * BL;
        const int tid = omp_get_thread_num();
        if (tid == 0) sub_nt = omp_get_num_threads();
        SubAcc& acc = sub[tid < 64 ? tid : 63];
        (void)acc;
#pragma omp for schedule(static)
        for (size_t cb = 0; cb < n1 / 8; ++cb) {
            const size_t c0 = 8 * cb;
            size_t cols[8];
            for (size_t t = 0; t < 8; ++t) cols[t] = c0 + t;
            SUB_T(t0);
            if (nat_src) {
                batch8_gather_nat(src_re, src_im, R, C, c0, ctx, tw2, a_re, a_im, ip);
            }
            else         batch8_gather(src_re, src_im, R, C, c0, false, 0, true, false, ctx, tw2, a_re, a_im, 0, true, 0, ~(size_t)0, ip);
            SUB_T(t1);
            SUB_ADD(acc.c[0], t0, t1);
            double* X_re; double* X_im; double* Y_re; double* Y_im;
            if (ip) {
                b8_ip_dif(a_re, a_im, c_re, c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false, false);
                X_re = a_re; X_im = a_im; Y_re = b_re; Y_im = b_im;
            }
            else if (two) {
                fft_batch8_2l(a_re, a_im, false, b_re, b_im, c_re, c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
                X_re = b_re; X_im = b_im; Y_re = a_re; Y_im = a_im;
            }
            else {
                const bool in_b = fft_batch8(a_re, a_im, b_re, b_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), false);
                X_re = in_b ? b_re : a_re; X_im = in_b ? b_im : a_im;
                Y_re = in_b ? a_re : b_re; Y_im = in_b ? a_im : b_im;
            }
            SUB_T(t2);
            SUB_ADD(acc.c[1], t1, t2);
            // Z -> Z^2 / N over all R vectors (two-level: the blocked layout, positions P(K))
            for (size_t K = 0; K < R; ++K) {
                const size_t pos = two ? batch8_posL(K, S) : K;
                const __m512d zr = _mm512_load_pd(X_re + 8 * pos);
                const __m512d zi = _mm512_load_pd(X_im + 8 * pos);
                const __m512d sr = _mm512_mul_pd(_mm512_fmsub_pd(zr, zr, _mm512_mul_pd(zi, zi)), v_s);
                const __m512d si = _mm512_mul_pd(_mm512_add_pd(_mm512_mul_pd(zr, zi), _mm512_mul_pd(zr, zi)), v_s);
                _mm512_store_pd(X_re + 8 * pos, sr);
                _mm512_store_pd(X_im + 8 * pos, si);
            }
            SUB_T(t3);
            SUB_ADD(acc.c[2], t2, t3);
            if (ip) {
                b8_ip_ditf(a_re, a_im, c_re, c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                SUB_T(t4);
                SUB_ADD(acc.c[3], t3, t4);
                batch8_store_rows(a_re, a_im, R, cols, dst_re, dst_im, false, DP, 0, ~(size_t)0, true);
                SUB_T(t5);
                SUB_ADD(acc.c[4], t4, t5);
            }
            else if (two) {
                fft_batch8_2l(X_re, X_im, true, Y_re, Y_im, c_re, c_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                SUB_T(t4);
                SUB_ADD(acc.c[3], t3, t4);
                batch8_store_rows(Y_re, Y_im, R, cols, dst_re, dst_im, true, DP);
                SUB_T(t5);
                SUB_ADD(acc.c[4], t4, t5);
            }
            else {
                const bool in_y = fft_batch8(X_re, X_im, Y_re, Y_im, R, ctx.c_w1r.data(), ctx.c_w1i.data(), true);
                SUB_T(t4);
                SUB_ADD(acc.c[3], t3, t4);
                batch8_store_rows(in_y ? Y_re : X_re, in_y ? Y_im : X_im, R, cols, dst_re, dst_im, false, DP);
                SUB_T(t5);
                SUB_ADD(acc.c[4], t4, t5);
            }
        }
    }
#ifdef LOTR_PROFILE
    sub_merge(ctx, 5, 5, sub, sub_nt);
#endif
}

// Reference unpack (serial; checked): untwist, unweight (k / a_j), round, carries, top carry wrap,
// digits -> limbs, R = (N - V) mod N into v. Used by LOTR_VERIFY_UNPACK=1 to check ibdwt_unpack.
// Returns the max rounding error; < 0 when the top carry did not settle (the caller reports a huge error).
static double ibdwt_unpack_ref(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n, uint64_t k) {
    const size_t N = ctx.fft_len, L = 2 * N;
    static thread_local std::vector<int64_t> tl_w;
    if (tl_w.size() < L) tl_w.resize(L);
    int64_t* const w = tl_w.data();
    const double* const kia = ctx.ib_kia.data();
    const double* const twr = ctx.ib_tw_re.data();
    const double* const twi = ctx.ib_tw_im.data();
    const double* const re = ctx.re;
    const double* const im = ctx.im;
    const size_t pad = ctx.lay_pad;
    const unsigned lg = ctx.lay_lg;
    double err = 0.0;
#pragma omp parallel for schedule(static) reduction(max : err)
    for (size_t j = 0; j < N; ++j) {
        const size_t p = j + (j >> lg) * pad;
        // (re + i im) * conj(theta^j)
        const double cr = re[p] * twr[j] + im[p] * twi[j];
        const double ci = im[p] * twr[j] - re[p] * twi[j];
        const double q0 = cr * kia[j];
        const double q1 = ci * kia[j + N];
        const double r0 = std::nearbyint(q0), r1 = std::nearbyint(q1);
        err = std::max(err, std::max(std::fabs(q0 - r0), std::fabs(q1 - r1)));
        w[j] = (int64_t)r0;
        w[j + N] = (int64_t)r1;
    }

    // carries over the L digits (digits end in [0, 2^b))
    const uint32_t* const e = ctx.ib_e.data();
    __int128 carry = 0;
    for (size_t j = 0; j < L; ++j) {
        const unsigned b = e[j + 1] - e[j];
        const __int128 x = (__int128)w[j] + carry;
        w[j] = (int64_t)(x & (((__int128)1 << b) - 1));
        carry = x >> b;
    }
    // top carry wrap: C*2^(e_L) = C*2^d*2^n == -q + r*2^n (C*2^d = q*k + r, 0 <= r < k)
    auto add_at = [&](size_t j, __int128 val) -> __int128 {
        while (val != 0 && j < L) {
            const unsigned b = e[j + 1] - e[j];
            const __int128 x = (__int128)w[j] + val;
            w[j] = (int64_t)(x & (((__int128)1 << b) - 1));
            val = x >> b;
            ++j;
        }
        return val;
    };
    int rounds = 0;
    while (carry != 0) {
        if (++rounds > 16) return -1.0;
        const __int128 tt = carry * ((__int128)1 << ctx.ib_d);
        __int128 qq = tt / (__int128)k;
        __int128 rr = tt % (__int128)k;
        if (rr < 0) { rr += (__int128)k; qq -= 1; }
        const __int128 c1 = add_at(0, -qq);
        const __int128 c2 = add_at(ctx.ib_m, rr << ctx.ib_msh);
        carry = c1 + c2;
    }

    // digits -> limbs: V = sum w_j 2^(e_j) < 2^(e_L)
    std::memset(v, 0, n_limbs * sizeof(uint64_t));
    for (size_t j = 0; j < L; ++j) {
        const uint64_t dgt = (uint64_t)w[j];
        if (dgt == 0) continue;
        const uint32_t pos = e[j];
        const unsigned b = e[j + 1] - pos;
        const size_t li = pos >> 6;
        const unsigned sh = pos & 63;
        if (li < n_limbs) v[li] |= dgt << sh;
        if (sh != 0 && sh + b > 64 && li + 1 < n_limbs) v[li + 1] |= dgt >> (64 - sh);
    }

    // N = k*2^n + 1 as limbs, V mod N (V < 2N: at most one subtraction), then R = (N - V) mod N
    std::vector<uint64_t> Nl(n_limbs, 0);
    Nl[0] = 1;
    {
        const size_t li = n >> 6;
        const unsigned sh = n & 63;
        if (li < n_limbs) Nl[li] |= k << sh;
        if (sh != 0 && li + 1 < n_limbs) Nl[li + 1] |= k >> (64 - sh);
    }
    auto geq = [&](const uint64_t* x, const uint64_t* y) {
        for (size_t i = n_limbs; i-- > 0;) {
            if (x[i] != y[i]) return x[i] > y[i];
        }
        return true;
    };
    auto sub_into = [&](uint64_t* r, const uint64_t* x, const uint64_t* y) {   // r = x - y (x >= y)
        unsigned char br = 0;
        for (size_t i = 0; i < n_limbs; ++i) {
            unsigned long long d;
            br = _subborrow_u64(br, x[i], y[i], &d);
            r[i] = d;
        }
    };
    if (geq(v, Nl.data())) sub_into(v, v, Nl.data());
    bool zero = true;
    for (size_t i = 0; i < n_limbs && zero; ++i) zero = (v[i] == 0);
    if (!zero) sub_into(v, Nl.data(), v);
    return err;
}

// Parallel unpack (same result as ibdwt_unpack_ref). The result is negated already at the rounding (-W == -k*Y^2),
// so no N - V on limbs:
//   1) per thread, complex positions [cb, ce): untwist, k / a_j, round, w = -round; carries over its two digit
//      segments [cb, ce) and [N+cb, N+ce) from carry 0; its share of the limbs zeroed
//   2) single: segment carries joined in digit order (lower segments, then upper), top carry wrap
//      (C*2^d = q*k + r: -q at digit 0, r*2^(n - e_m) at digit m), V >= N ? V - N (on the few top digits and digit 0)
//   3) per thread: its digits -> limbs (limbs shared with a neighbouring segment by atomic OR)
// to_limbs = false: the normalized digits stay in ctx.ib_w / ctx.ib_b only (the next square packs them directly).
static double ibdwt_unpack(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n, uint64_t k, bool to_limbs = true) {
    const size_t N = ctx.fft_len, L = 2 * N;
    if (ctx.ib_w.size() < L + 8) ctx.ib_w.assign(L + 8, 0);
    int32_t* const w = ctx.ib_w.data();
    const uint8_t* const bsz = ctx.ib_b.data();   // per digit: size b_j (bits 0..6) | c_j << 7 (from the setup)
    const double* const re = ctx.re;
    const double* const im = ctx.im;
    const size_t pad = ctx.lay_pad;
    const unsigned lg = ctx.lay_lg;
    const size_t m = ctx.ib_m;
    const unsigned msh = ctx.ib_msh;
    const double kd = (double)k;

    // adds val at digit j and propagates (digits stay in [0, 2^b)); returns what leaves the top digit
    auto add_at = [&](size_t j, __int128 val) -> __int128 {
        while (val != 0 && j < L) {
            const unsigned b = bsz[j] & 0x7F;
            const __int128 x = (__int128)w[j] + val;
            w[j] = (int32_t)(x & (((__int128)1 << b) - 1));
            val = x >> b;
            ++j;
        }
        return val;
    };
    // digits [jb, je) -> limbs; the first and the last limb of the range may be shared with a neighbour (atomic OR)
    auto put = [&](size_t jb, size_t je) {
        if (jb >= je) return;
        uint64_t pos = ib_e_at(ctx, jb);
        const size_t first_li = pos >> 6;
        const size_t last_li = (ib_e_at(ctx, je) - 1) >> 6;
        auto flush = [&](size_t li, uint64_t val) {
            if (val == 0 || li >= n_limbs) return;
            if (li == first_li || li == last_li) __atomic_fetch_or(&v[li], val, __ATOMIC_RELAXED);
            else v[li] |= val;
        };
        uint64_t acc = 0;
        size_t cur = first_li;
        for (size_t j = jb; j < je; ++j) {
            const uint64_t dgt = (uint64_t)(uint32_t)w[j];
            const unsigned b = bsz[j] & 0x7F;
            const size_t li = pos >> 6;
            const unsigned sh = pos & 63;
            if (li != cur) { flush(cur, acc); acc = 0; cur = li; }
            acc |= dgt << sh;
            if (sh + b > 64) { flush(cur, acc); acc = dgt >> (64 - sh); cur = li + 1; }
            pos += b;
        }
        flush(cur, acc);
    };
    // unweight (k / a_j) and round (negated) 8 values of digits jj .. jj+7 -> int64 lanes (as __m512d bits)
    auto round8 = [&](size_t jj, __m512d cval, __m512d& vmax) -> __m512d {
        const __m512i bc = _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(bsz + jj)));
        const __mmask8 c = _mm512_test_epi64_mask(bc, _mm512_set1_epi64(0x80));
        const size_t hh = jj >> 6, ll = jj & 63;
        __m512d kia = _mm512_mul_pd(_mm512_set1_pd(kd * ctx.ib_iWH[hh]), _mm512_loadu_pd(ctx.ib_iWL.data() + ll));
        kia = _mm512_mask_mul_pd(kia, c, kia, _mm512_set1_pd(0.5));
        if (jj == 0) kia = _mm512_mask_mov_pd(kia, 0x01, _mm512_set1_pd(kd));   // digit 0: a = 1
        const __m512d q = _mm512_mul_pd(cval, kia);
        const __m512d r = _mm512_roundscale_pd(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        vmax = _mm512_max_pd(vmax, _mm512_abs_pd(_mm512_sub_pd(q, r)));
        return _mm512_castsi512_pd(_mm512_sub_epi64(_mm512_setzero_si512(), _mm512_cvtpd_epi64(r)));
    };
    // untwist complex positions j0 .. j0+7: (xr + i xi) * conj(theta^j) -> cr (digits j), ci (digits N + j)
    auto untwist8 = [&](size_t j0, __m512d& cr, __m512d& ci) {
        const size_t p = j0 + (j0 >> lg) * pad;
        const __m512d xr = _mm512_load_pd(re + p), xi = _mm512_load_pd(im + p);
        __m512d tr, ti;
        ib_twist8(ctx, j0, tr, ti);
        cr = _mm512_fmadd_pd(xr, tr, _mm512_mul_pd(xi, ti));
        ci = _mm512_fmsub_pd(xi, tr, _mm512_mul_pd(xr, ti));
    };

    int T = omp_get_max_threads();
    if (T > 64) T = 64;
    if (N < 4096) T = 1;
    int64_t seg_c[2][64][8];   // carries out of the lane segments: [lower/upper half][thread][lane]
    double seg_err[64];
    int used = 1;
    bool ok = true;
    // thread ranges of complex positions: multiples of 64 (8 lanes of whole 8-digit steps)
    auto seg_bound = [N](int t, int nt) -> size_t { return 64 * ((N / 64) * (size_t)t / (size_t)nt); };
    // Rounded values x[l] of digits jb + l*len + t0 .. +7 -> carries in 8 lanes (8 chains at once, variable digit
    // sizes: srav/sllv per lane), normalized digits stored as int32.
    auto carry8 = [&](__m512d* x, size_t jb, size_t len, size_t t0, __m512i& carry) {
        const __m512i one = _mm512_set1_epi64(1);
        __m512d bb[8];
        for (int l = 0; l < 8; ++l) {
            const size_t j = jb + (size_t)l * len + t0;
            bb[l] = _mm512_castsi512_pd(_mm512_and_si512(_mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(bsz + j))),
                                                         _mm512_set1_epi64(0x7F)));
        }
        st_transpose8x8(x);    // x[s] lane l = digit t0+s of lane l
        st_transpose8x8(bb);
        for (int s = 0; s < 8; ++s) {
            const __m512i b = _mm512_castpd_si512(bb[s]);
            const __m512i vv = _mm512_add_epi64(_mm512_castpd_si512(x[s]), carry);
            carry = _mm512_srav_epi64(vv, b);
            x[s] = _mm512_castsi512_pd(_mm512_and_si512(vv, _mm512_sub_epi64(_mm512_sllv_epi64(one, b), one)));
        }
        st_transpose8x8(x);
        for (int l = 0; l < 8; ++l)
            _mm256_storeu_si256((__m256i*)(w + jb + (size_t)l * len + t0), _mm512_cvtepi64_epi32(_mm512_castpd_si512(x[l])));
    };
    // The rounded values (up to ~2^55) go to an int64 scratch in digit order (one sequential pass), the carries then
    // read it in 8 lanes and write the int32 digit state. A single fused pass (untwist + carries in the 8 lanes) was
    // slower (16 scattered streams per thread).
    static thread_local std::vector<int64_t> tl_tmp;   // the calling thread's (one per worker), shared by its team
    if (tl_tmp.size() < L + 8) tl_tmp.assign(L + 8, 0);
    int64_t* const tmp = tl_tmp.data();
    // carries of digits [jb, jb + 8*len) in 8 lanes (lane l = digits jb + l*len .. + len, len a multiple of 8), each
    // from carry 0; out[l] = carry out of lane l
    auto norm_lanes = [&](size_t jb, size_t len, int64_t* out) {
        __m512i carry = _mm512_setzero_si512();
        for (size_t t0 = 0; t0 < len; t0 += 8) {
            __m512d x[8];
            for (int l = 0; l < 8; ++l)
                x[l] = _mm512_castsi512_pd(_mm512_loadu_si512((const void*)(tmp + jb + (size_t)l * len + t0)));
            carry8(x, jb, len, t0, carry);
        }
        _mm512_storeu_si512((void*)out, carry);
    };
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        if (t == 0) used = nt;
        const size_t cb = seg_bound(t, nt), ce = seg_bound(t + 1, nt);
        __m512d vmax = _mm512_setzero_pd();
        // tmp = -round (negated: -W == -k*Y^2), then the carries into the int32 digit state
        for (size_t j0 = cb; j0 < ce; j0 += 8) {
            __m512d cr, ci;
            untwist8(j0, cr, ci);
            _mm512_storeu_pd((double*)(tmp + j0), round8(j0, cr, vmax));
            _mm512_storeu_pd((double*)(tmp + N + j0), round8(N + j0, ci, vmax));
        }
        seg_err[t] = _mm512_reduce_max_pd(vmax);
        norm_lanes(cb, (ce - cb) / 8, seg_c[0][t]);
        norm_lanes(N + cb, (ce - cb) / 8, seg_c[1][t]);
        if (to_limbs) {
            const size_t lb = n_limbs * (size_t)t / (size_t)nt, le = n_limbs * (size_t)(t + 1) / (size_t)nt;
            std::memset(v + lb, 0, (le - lb) * sizeof(uint64_t));
        }
#pragma omp barrier
#pragma omp single
        {
            // carry of segment s goes to the first digit of segment s+1 (lower segments, then upper ones)
            __int128 top = 0;
            // lane segments in digit order: lower half (threads, lanes), then the upper half
            for (int hf = 0; hf < 2; ++hf) {
                for (int s = 0; s < nt; ++s) {
                    const size_t b0 = seg_bound(s, nt), len = (seg_bound(s + 1, nt) - b0) / 8;
                    if (len == 0) continue;
                    for (int l = 0; l < 8; ++l) {
                        const size_t seg_end = (size_t)hf * N + b0 + (size_t)(l + 1) * len;
                        top += add_at(seg_end, seg_c[hf][s][l]);
                    }
                }
            }
            // top carry wrap: C*2^(e_L) = C*2^d*2^n == -q + r*2^n (C*2^d = q*k + r, 0 <= r < k)
            int rounds = 0;
            while (top != 0) {
                if (++rounds > 16) { ok = false; break; }
                const __int128 tt = top * ((__int128)1 << ctx.ib_d);
                __int128 qq = tt / (__int128)k;
                __int128 rr = tt % (__int128)k;
                if (rr < 0) { rr += (__int128)k; qq -= 1; }
                const __int128 c1 = add_at(0, -qq);
                const __int128 c2 = add_at(m, rr << msh);
                top = c1 + c2;
            }
            // V in [0, 2^(e_L)) < 2N: V >= N  <=>  (V >> n) > k, or == k with a nonzero low part; then V -= N
            if (ok) {
                __int128 hi = (__int128)(w[m] >> msh);
                for (size_t j = m + 1; j < L; ++j) hi += (__int128)w[j] << (ib_e_at(ctx, j) - n);
                bool low_nz = (w[m] & (((int64_t)1 << msh) - 1)) != 0;
                for (size_t j = 0; j < m && !low_nz; ++j) low_nz = (w[j] != 0);
                if (hi > (__int128)k || (hi == (__int128)k && low_nz)) {
                    add_at(m, -((__int128)k << msh));
                    add_at(0, -1);
                }
            }
        }
        if (to_limbs) {
            put(cb, ce);
            put(N + cb, N + ce);
        }
    }
    double err = 0.0;
    for (int t = 0; t < used; ++t) err = std::max(err, seg_err[t]);
    if (!ok) return -1.0;

    // LOTR_VERIFY_UNPACK=1: compare with the serial reference
    static const bool verify = (std::getenv("LOTR_VERIFY_UNPACK") != nullptr);
    if (verify && to_limbs) {
        std::vector<uint64_t> par(v, v + n_limbs);
        ibdwt_unpack_ref(ctx, v, n_limbs, n, k);
        if (std::memcmp(par.data(), v, n_limbs * sizeof(uint64_t)) != 0)
            std::fprintf(stderr, "[VERIFY_UNPACK] MISMATCH: IBDWT parallel unpack != serial reference\n");
    }
    return err;
}

// Block-major tables of ib_norm_pack_rows8 (see fft_types.h): ib_bm from ib_b, ib_wT from ib_WH / ib_iWH / ib_TH,
// once per (k, n, n2). R = n1, C = n2.
static void ib_build_blocked(FFTContext& ctx, size_t R, size_t C) {
    if (ctx.ib_T_k == ctx.ib_k && ctx.ib_T_n == ctx.ib_n && ctx.ib_T_C == C && !ctx.ib_bm.empty()) return;
    const size_t N = ctx.fft_len, NR = N >> 3, NB = C >> 3, G = C >> 6, M = G * R;
    ctx.ib_bm.assign(2 * NR, 0);
    ctx.ib_wT.assign(6 * M, 0.0);
    const uint8_t* const bsz = ctx.ib_b.data();
    const unsigned q0 = (unsigned)ctx.ib_IL[1];
#pragma omp parallel for schedule(static)
    for (size_t K = 0; K < R; ++K) {
        for (size_t hh = 0; hh < 2; ++hh)
            for (size_t cb = 0; cb < NB; ++cb) {
                // 2 bytes per run: low = the digits of q0 + 1 bits, high = c (weight x2)
                const uint8_t* const b8 = bsz + hh * N + K * C + 8 * cb;
                unsigned big = 0, cw = 0;
                for (unsigned s = 0; s < 8; ++s) {
                    if ((unsigned)(b8[s] & 0x7F) > q0) big |= 1u << s;
                    if (b8[s] & 0x80) cw |= 1u << s;
                }
                ctx.ib_bm[hh * NR + cb * R + K] = (uint16_t)(big | (cw << 8));
            }
        for (size_t g = 0; g < G; ++g) {
            const size_t h = K * G + g;   // (K*C + 64g) >> 6
            ctx.ib_wT[0 * M + g * R + K] = ctx.ib_WH[h];
            ctx.ib_wT[1 * M + g * R + K] = ctx.ib_iWH[h];
            ctx.ib_wT[2 * M + g * R + K] = ctx.ib_THr[h];
            ctx.ib_wT[3 * M + g * R + K] = ctx.ib_THi[h];
            ctx.ib_wT[4 * M + g * R + K] = ctx.ib_WH[(N >> 6) + h];
            ctx.ib_wT[5 * M + g * R + K] = ctx.ib_iWH[(N >> 6) + h];
        }
    }
    ctx.ib_T_k = ctx.ib_k;
    ctx.ib_T_n = ctx.ib_n;
    ctx.ib_T_C = C;
}

// from_digits: the input is the digit state of the previous square (ctx.ib_w / ib_b), v is not read.
// to_limbs = false: the result stays in the digit state only (v not written).
// chain_next: another square from the digit state follows (fft_square_repeat) - its pass A may be fused into this
// square's B' (ibdwt_pass_BA); that next square then starts at the middle (ctx.ib_a_done).
static void ibdwt_square(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val,
                         bool from_digits = false, bool to_limbs = true, bool chain_next = false) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;
    size_t n1 = 0, n2 = 0;
    if (!bailey_fused_ok(ctx, n1, n2) || !ibdwt_setup(ctx, n_bits, k_val)) {
        static std::atomic<bool> warned{ false };
        if (!warned.exchange(true))
            std::fprintf(stderr, "[ERROR] IBDWT not available for FFT length %zu, n=%u, k=%llu\n",
                         ctx.fft_len, n_bits, (unsigned long long)k_val);
        ctx.last_max_diff = 1.0e30;
        return;
    }
    ctx.ib_R = n1;   // block-major run order (ib_ri)
    {
        unsigned lg = 0; while ((size_t(1) << lg) < n2) ++lg;
        ctx.ib_lgC = lg;
    }
    // min(n1, n2) >= 128: the headroom of fft_init (see fft_square_real; fft_ibdwt_supported already demands it)
    const size_t pad = (std::min(n1, n2) >= 128) ? 8 : 0;
    ctx.lay_pad = pad;
    if (pad) {
        unsigned lg = 0; while ((size_t(1) << lg) < n2) ++lg;
        ctx.lay_lg = lg;
    }
    else {
        ctx.lay_lg = 0;
    }

    // From the digit state the pack can run inside pass A (batch8_gather_ib). With the data in L3 a separate pack is
    // faster; when the data does not fit (RAM-bound, e.g. 2^23) the fused pack wins (one pass less).
    // Default: fused from FFT length new_min (below). LOTR_IB_PACK_FUSE=1 / =0 forces it on / off.
    static const int fuse_mode = []() {
        const char* e = std::getenv("LOTR_IB_PACK_FUSE");
        return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
    }();
    // Default threshold of all the IBDWT modes below (pack in A, normalization in B', in-place layout, overlaid digits,
    // B' + next A): 2^15, i.e. on from FFT 49152 (measured faster there, slower on 16384 / 24576). LOTR_IB_NEWMIN=<len>
    // changes it.
    static const size_t new_min = []() {
        const char* e = std::getenv("LOTR_IB_NEWMIN");
        return e ? (size_t)std::strtoull(e, nullptr, 10) : ((size_t)1 << 15);
    }();
    const bool fuse = from_digits && pad && (fuse_mode == 1 || (fuse_mode < 0 && ctx.fft_len >= new_min));
    // Normalization fused into pass B' (the unpack disappears) unless the limbs are wanted (the last square of a run: full
    // unpack, digits fully normalized again). Pays where the data does not fit L3 (with the data in cache it is the same
    // work). Default from FFT length new_min (as the pack in pass A); LOTR_IB_NORM=1 / =0 forces it on / off.
    static const int norm_mode = []() {
        const char* e = std::getenv("LOTR_IB_NORM");
        return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
    }();
    const bool fnorm = !to_limbs && pad && (norm_mode == 1 || (norm_mode < 0 && ctx.fft_len >= new_min));
    const size_t L = 2 * ctx.fft_len;
    // a_done: the previous square ran this square's pass A inside its B' (ibdwt_pass_BA): re/im already hold the
    // output of pass A, the top carry wrap is applied. Start at the middle.
    const bool a_done = from_digits && ctx.ib_a_done;
    ctx.ib_a_done = false;
    if (from_digits && ctx.ib_rc_live && !a_done) {
        ib_top_wrap(ctx);   // carry out of the top run -> ib_cin0 / ib_madd
    }
    else if (!from_digits) {
        ctx.ib_rc_live = false;
    }
    // In-place layout: the data stays in re/im in the natural order (n1 rows of n2) through all three passes - A stores its
    // columns back (transposed_out), the middle reads rows (batch8_gather_nat) and writes them back, B' reads columns.
    // re2/im2 are not touched: the working set of a giant (3*2^21) drops from ~260 to ~160 MB, and mid-size lengths may fit
    // the L3 (7*2^20267500 on 3*2^18: 2x faster on a 32 MB L3). On small lengths the row gather costs more than the smaller
    // set saves. Default from FFT length new_min; LOTR_IB_INPLACE=1 / =0 forces it on / off.
    static const int inpl_mode = []() {
        const char* e = std::getenv("LOTR_IB_INPLACE");
        return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
    }();
    const bool inpl = pad && (inpl_mode == 1 || (inpl_mode < 0 && ctx.fft_len >= new_min));
    // Overlaid digit state: with the in-place layout, the fused normalization and the pack in pass A all on,
    // the fused B' writes the digits into re itself (ib_norm_rows8) and the next pass A reads them there: no ib_w
    // (8 B per position) in the working set. LOTR_IB_OVL=0 = digits in ib_w.
    static const bool ovl_off = []() { const char* e = std::getenv("LOTR_IB_OVL"); return e && std::atoi(e) == 0; }();
    const bool fuse_next = pad && (fuse_mode == 1 || (fuse_mode < 0 && ctx.fft_len >= new_min));
    const bool ovl = inpl && fnorm && fuse_next && !ovl_off;
    // B' fused with the next pass A (ibdwt_pass_BA): with the overlaid state, whenever another square
    // follows. LOTR_IB_CHAIN=0 = separate passes.
    static const bool chain_off = []() { const char* e = std::getenv("LOTR_IB_CHAIN"); return e && std::atoi(e) == 0; }();
    const bool chain = ovl && chain_next && !chain_off && n2 / 8 >= 2 * (size_t)omp_get_max_threads();
    if (from_digits && ctx.ib_ovl && !fuse && !a_done) {   // cannot happen (ovl only with fuse_next); the digits would be stale
        std::fprintf(stderr, "[ERROR] IBDWT: overlaid digit state without the fused pack\n");
        ctx.last_max_diff = 1.0e30;
        return;
    }
    double* const A_re = inpl ? ctx.re : ctx.re2;   // output of pass A = input of the middle
    double* const A_im = inpl ? ctx.im : ctx.im2;
    double* const B_re = inpl ? ctx.re : ctx.re2;   // output of pass B' (without the fused normalization)
    double* const B_im = inpl ? ctx.im : ctx.im2;
    const auto t0 = clock::now();
    if (from_digits) { if (!fuse && !a_done) ibdwt_pack_digits(ctx); }
    else             ibdwt_pack(ctx, v, n_limbs);
    const auto t1 = clock::now();
    if (a_done) {
        // pass A already done by the previous square's B'
    }
    else if (pad) {
        LOTR_PASS(ctx, 0);
        bailey_colfft_pass8(ctx.re, ctx.im, A_re, A_im, n1, n2, false, inpl, ctx, false,
                            ctx.r_w1r, ctx.r_w1i, n2 + pad, inpl ? n2 + pad : n1 + pad, 0, 0, nullptr, fuse ? &ctx : nullptr);
    }
    else {
        LOTR_PASS(ctx, 0);
        bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, false, false, ctx, false,
                       ctx.r_w1r, ctx.r_w1i);
    }
    const auto t2 = clock::now();
    {
        LOTR_PASS(ctx, 1);
        if (inpl) bailey_fused_cplx_mid8(ctx, ctx.re, ctx.im, ctx.re, ctx.im, n1, n2, n2 + pad, n2 + pad, true);
        else      bailey_fused_cplx_mid8(ctx, ctx.re2, ctx.im2, ctx.re, ctx.im, n1, n2, pad ? n1 + pad : 0, pad ? n2 + pad : 0);
    }
    const auto t3 = clock::now();
    if (fnorm) {
        if (ctx.ib_w.size() < L + 8) ctx.ib_w.assign(L + 8, 0);
        if (ctx.ib_rc.size() < L / 8) ctx.ib_rc.assign(L / 8, 0);
        if (ovl && ctx.ib_h7.size() < L / 8) ctx.ib_h7.assign(L / 8, 0);
        double errs[64] = {};
        LOTR_PASS(ctx, 2);
        if (chain) {
            ib_build_blocked(ctx, n1, n2);
            ibdwt_pass_BA(ctx, n1, n2, n2 + pad, errs);
            ctx.ib_a_done = true;
        }
        else {
            ctx.ib_ovl_write = ovl;
            bailey_colfft_pass8(ctx.re, ctx.im, B_re, B_im, n1, n2, true, true, ctx, true,
                                ctx.r_w1r, ctx.r_w1i, n2 + pad, n2 + pad, 0, 11, nullptr, nullptr, errs);
            ctx.ib_ovl_write = false;
        }
        ctx.ib_ovl = ovl;
        double err = 0.0;
        for (int t = 0; t < 64; ++t) err = std::max(err, errs[t]);
        ctx.last_max_diff = err;
        ctx.ib_rc_live = true;
    }
    else if (pad) {
        LOTR_PASS(ctx, 2);
        bailey_colfft_pass8(ctx.re, ctx.im, B_re, B_im, n1, n2, true, true, ctx, true,
                            ctx.r_w1r, ctx.r_w1i, n2 + pad, n2 + pad, 0, 11);
    }
    else {
        LOTR_PASS(ctx, 2);
        bailey_colpass(ctx.re, ctx.im, ctx.re2, ctx.im2, n1, n2, true, true, ctx, true,
                       ctx.r_w1r, ctx.r_w1i);
    }
    if (!inpl) {   // the result of B' is in re2/im2: the unpack reads re/im
        std::swap(ctx.re, ctx.re2);
        std::swap(ctx.im, ctx.im2);
    }
    const auto t4 = clock::now();
    if (!fnorm) {
        const double err = ibdwt_unpack(ctx, v, n_limbs, n_bits, k_val, to_limbs);
        ctx.last_max_diff = (err < 0.0) ? 1.0e30 : err;
        ctx.ib_rc_live = false;   // the digit state is fully normalized again
        ctx.ib_ovl = false;       // ... and in ib_w
    }
    ctx.lay_pad = 0;
    const auto t5 = clock::now();

    ctx.prof_pack += secd(t1 - t0).count();
    ctx.prof_fwd += secd(t2 - t1).count();
    ctx.prof_square += secd(t3 - t2).count();
    ctx.prof_inv += secd(t4 - t3).count();
    ctx.prof_unpack += secd(t5 - t4).count();
    ctx.prof_calls++;
}

// Self-test of the IBDWT (once per context and (k, n)): (-k * a^2) mod N with GMP for a pseudo-random a < 2^n.
static void ibdwt_selftest(FFTContext& ctx, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    // + 64 zero limbs: the packs read and the unpacks write in groups of 2-4 limbs, so with an n_limbs that is not a
    // multiple of 4 (calibration: (n+63)/64, e.g. 3125 for n = 200000) they went up to 3 limbs past the vector - garbage
    // digits (autotest err 1e30) and a damaged heap ("free(): invalid size"). The runs pass large buffers (fft_len/4).
    std::vector<uint64_t> a(n_limbs + 64, 0);
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
    ibdwt_square(ctx, b.data(), n_limbs, n_bits, k_val);
    const double err = ctx.last_max_diff;

    mpz_t A, N, R, X;
    mpz_inits(A, N, R, X, nullptr);
    mpz_import(A, n_limbs, -1, 8, 0, 0, a.data());
    mpz_set_ui(N, (unsigned long)k_val);
    mpz_mul_2exp(N, N, n_bits);
    mpz_add_ui(N, N, 1);
    mpz_mul(R, A, A);
    mpz_mul_ui(R, R, (unsigned long)k_val);
    mpz_neg(R, R);
    mpz_mod(R, R, N);
    mpz_import(X, n_limbs, -1, 8, 0, 0, b.data());
    const bool ok = (mpz_cmp(R, X) == 0) && (err < 0.4);
    mpz_clears(A, N, R, X, nullptr);

    ctx.sq_real_state = ok ? 1 : -1;
    const double bpd = (double)ctx.ib_e.back() / (double)(2 * ctx.fft_len);
    // calibration: the length is only rejected (a large k, e.g. 35 bits: the products pass 2^53, where every double is an
    // integer - wrong digits with "max err 0"); auto_tune_uncached prints the summary
    if (!ok && ctx.quiet_selftest) return;
    if (!ok && err >= 0.4) {
        // precision limit (too many bits per digit for this k), expected while the calibration tries short lengths
        if (g_detail)
            std::fprintf(stderr, "[CPU IBDWT] FFT %zu: %.2f bits per digit are too many for k=%llu (max err %.3g) - not usable\n",
                ctx.fft_len, bpd, (unsigned long long)k_val, err);
        return;
    }
    static std::atomic<bool> printed{ false };
    if (!ok || (g_detail && !printed.exchange(true))) {
        std::fprintf(stderr, "[CPU IBDWT] autotest %s: FFT %zu (%zu digits of ~%.2f bits), k=%llu, max err %.3g (reference: GMP)\n",
            ok ? "OK" : "MISMATCH (computation error)", ctx.fft_len, 2 * ctx.fft_len, bpd, (unsigned long long)k_val, err);
    }
}

// Self-test (once per FFT context): square a pseudo-random number < 2^n with both paths; the results must be identical.
// On a mismatch the context stays on the old path (3 FFTs) - the test still computes correctly, only slower.
static void sq_real_selftest(FFTContext& ctx, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    // + 64 zero limbs: the packs read and the unpacks write in groups of 2-4 limbs, so with an n_limbs that is not a
    // multiple of 4 (calibration: (n+63)/64, e.g. 3125 for n = 200000) they went up to 3 limbs past the vector - garbage
    // digits (autotest err 1e30) and a damaged heap ("free(): invalid size"). The runs pass large buffers (fft_len/4).
    std::vector<uint64_t> a(n_limbs + 64, 0);
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

// Self-test for wide digits / 3*2^p lengths (fft_real_only): the old 3-FFT path knows only 16-bit digits on powers of two,
// so the reference is GMP: (a^2 mod N) for a pseudo-random a < 2^n must equal the real-FFT result (reduced mod N).
static void sq_real_selftest_wide(FFTContext& ctx, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    // + 64 zero limbs: the packs read and the unpacks write in groups of 2-4 limbs, so with an n_limbs that is not a
    // multiple of 4 (calibration: (n+63)/64, e.g. 3125 for n = 200000) they went up to 3 limbs past the vector - garbage
    // digits (autotest err 1e30) and a damaged heap ("free(): invalid size"). The runs pass large buffers (fft_len/4).
    std::vector<uint64_t> a(n_limbs + 64, 0);
    uint64_t st = 0x9E3779B97F4A7C15ULL ^ (uint64_t)ctx.fft_len ^ (k_val << 20) ^ ctx.digit_bits;
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

    fft_square_real(ctx, b.data(), n_limbs, n_bits, k_val);
    const double err = ctx.last_max_diff;

    mpz_t A, N, R, X;
    mpz_inits(A, N, R, X, nullptr);
    mpz_import(A, n_limbs, -1, 8, 0, 0, a.data());
    mpz_set_ui(N, (unsigned long)k_val);
    mpz_mul_2exp(N, N, n_bits);
    mpz_add_ui(N, N, 1);
    mpz_mul(R, A, A);
    if (unpack_scaled_enabled()) {   // scaled form: the square returns -k*a^2 mod N (see fft_pack_vbmi.h)
        mpz_mul_ui(R, R, (unsigned long)k_val);
        mpz_neg(R, R);
    }
    mpz_mod(R, R, N);
    mpz_import(X, n_limbs, -1, 8, 0, 0, b.data());
    mpz_mod(X, X, N);
    const bool ok = (mpz_cmp(R, X) == 0) && (err < 0.4);
    mpz_clears(A, N, R, X, nullptr);

    ctx.sq_real_state = ok ? 1 : -1;
    static std::atomic<bool> printed{ false };
    if (!ok || (g_detail && !printed.exchange(true))) {
        std::fprintf(stderr, "[SQ_REAL] autotest %s: FFT %zu, %u-bit digits, max err %.3g (reference: GMP)%s\n",
            ok ? "OK" : "MISMATCH", ctx.fft_len, ctx.digit_bits, err,
            ok ? "" : " -> this FFT length / digit width cannot be used (LOTR_DIGIT_BITS=16 turns the wide digits off)");
    }
}

void fft_square_karatsuba(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    if (ctx.ibdwt) {
        // IBDWT (scaled form only): self-test once per (k, n), a failure reports an "infinite" error (test aborts)
        if (ctx.sq_real_state == 0 || ctx.ib_k != k_val || ctx.ib_n != n_bits) ibdwt_selftest(ctx, n_limbs, n_bits, k_val);
        ibdwt_square(ctx, v, n_limbs, n_bits, k_val);
        if (ctx.sq_real_state < 0) ctx.last_max_diff = 1.0e30;
        return;
    }
    if (fft_real_only(ctx)) {
        // Wide digits and 3*2^p lengths exist only in the real-FFT square. A failed self-test (normally caught by the calibration in
        // dcore.cpp) reports an "infinite" error, so the Gerbicz-Li check rejects every segment and the test aborts
        // without writing a result.
        if (ctx.sq_real_state == 0) sq_real_selftest_wide(ctx, n_limbs, n_bits, k_val);
        fft_square_real(ctx, v, n_limbs, n_bits, k_val);
        if (ctx.sq_real_state < 0) ctx.last_max_diff = 1.0e30;
        return;
    }
    if (n_limbs * 4 > ctx.fft_len || ctx.sq_real_state < 0) {
        fft_square_karatsuba3(ctx, v, n_limbs, n_bits, k_val);
        return;
    }
    if (ctx.sq_real_state == 0) sq_real_selftest(ctx, n_limbs, n_bits, k_val);
    if (ctx.sq_real_state > 0) fft_square_real(ctx, v, n_limbs, n_bits, k_val);
    else fft_square_karatsuba3(ctx, v, n_limbs, n_bits, k_val);
}

bool fft_uses_scaled_form() {
    return unpack_scaled_enabled();
}

// count squarings of v in a row (scaled form). IBDWT: the digits stay in the context between them and become limbs
// only after the last one. ctx.last_max_diff = max over the squarings.
void fft_square_repeat(FFTContext& ctx, uint64_t* v, size_t n_limbs, unsigned n_bits, uint64_t k_val, unsigned count) {
    double mx = 0.0;
    if (!ctx.ibdwt) {
        for (unsigned i = 0; i < count; ++i) {
            fft_square_karatsuba(ctx, v, n_limbs, n_bits, k_val);
            mx = std::max(mx, ctx.last_max_diff);
        }
        ctx.last_max_diff = mx;
        return;
    }
    if (ctx.sq_real_state == 0 || ctx.ib_k != k_val || ctx.ib_n != n_bits) ibdwt_selftest(ctx, n_limbs, n_bits, k_val);
    for (unsigned i = 0; i < count; ++i) {
        ibdwt_square(ctx, v, n_limbs, n_bits, k_val, i > 0, i + 1 == count, i + 1 < count);
        mx = std::max(mx, ctx.last_max_diff);
    }
    ctx.last_max_diff = (ctx.sq_real_state < 0) ? 1.0e30 : mx;
}

void fft_mul_karatsuba(FFTContext& ctx, uint64_t* out, const uint64_t* X, const uint64_t* Y, size_t n_limbs, unsigned n_bits, uint64_t k_val) {
    if (fft_real_only(ctx)) {
        // The multiplication knows only 16-bit digits on powers of two; otherwise proth_logic.cpp multiplies through two
        // squarings (x*y = ((x+y)^2 - (x-y)^2) / 4). Reaching this point is a bug: report an "infinite" error.
        std::fprintf(stderr, "[ERROR] fft_mul_karatsuba called with %u-bit digits / FFT length %zu\n", ctx.digit_bits, ctx.fft_len);
        ctx.last_max_diff = 1.0e30;
        return;
    }

    auto run_fft_on_buffer = [&](double*& target_re, double*& target_im, bool inverse) {
        std::swap(ctx.re, target_re);
        std::swap(ctx.im, target_im);

        fft_run_inplace(ctx, inverse);

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
