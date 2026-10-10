// Copyright (C) Damian Koba.

#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct FFTFactorPlan {
    size_t total_n;
    std::vector<size_t> factors;
    bool has_unsupported_primes;
};

enum class FFTBackend {
    INTERNAL
};

struct alignas(64) FFTContext {
    
    
    double* re;
    double* im;
    double* re2;
    double* im2;
    double* w1_re;
    double* w1_im;
    double* w2_re;
    double* w2_im;

    
    double* w3_re;
    double* w3_im;
    double* proth_w_re;
    double* proth_w_im;
    double* k_A_re;
    double* k_A_im;
    double* k_B_re;
    double* k_B_im;

    double* k_AB_re;
    double* k_AB_im;
    double* g_C_re;
    double* g_C_im;
    double* g_D_re;
    double* g_D_im;
    void* base_mem_ptr;
    void* base_ptr;
    

    void* raw_allocation;
    size_t   total_allocation_size;
    size_t   fft_len;
    FFTFactorPlan factor_plan;
    uint64_t current_i;
    uint64_t prof_calls;
    double   last_max_diff;
    double   prof_pack;
    double   prof_fwd;

    
    double   prof_square;
    double   prof_inv;
    double   prof_unpack;
    double   prof_unpack_clear;
    double   prof_unpack_deweight;
    double   prof_unpack_scalar;
    double   prof_unpack_finalfix;
#ifdef LOTR_PROFILE
    // times of the passes inside the FFT (summed over the FFT calls of one iteration):
    //   Bailey 2D, 5 passes: [0] transpose 1, [1] row FFTs 1, [2] transpose+twiddle, [3] row FFTs 2, [4] transpose 2
    //   Bailey 2D two-pass / fused square: [0] pass A, [1] fused middle (or pass B), [2] pass B'
    //   Stockham 1D: consecutive passes (radix-8, ..., radix-2/4 at the end)
    double   prof_pass[8];
#endif
    int      num_workers;      
    
    uint64_t k_val = 0;
    // Real-FFT squaring (2 FFTs instead of 3): 0 = self-test not run yet, 1 = OK, -1 = mismatch (old path is used)
    int sq_real_state = 0;
    double* pre_weights_real;
    double* pre_weights_imag;
    double* post_weights_real;
    double* post_weights_imag;

    
    FFTBackend backend;        
    bool       use_avx2_fallback;
    

    std::vector<double> r_w1r;
    std::vector<double> r_w1i;
    std::vector<double> r_w2r;
    std::vector<double> r_w2i;
    std::vector<double> r_w3r;
    std::vector<double> r_w3i;

    // Two-level twiddle table w^e = hi[e >> tw2_shift] * lo[e & (2^tw2_shift - 1)] (small, stays in L1/L2; replaces scattered reads of w1)
    std::vector<double> tw2_lo_re, tw2_lo_im, tw2_hi_re, tw2_hi_im;
    unsigned tw2_shift = 0;
    size_t tw2_N = 0;

    // Padded row layout of the fused squaring path (pack -> passes -> unpack): element j is stored at position
    // (j >> lay_lg) * (2^lay_lg + lay_pad) + (j & (2^lay_lg - 1)). lay_pad = 0: natural layout (all other paths).
    size_t lay_pad = 0;
    unsigned lay_lg = 0;

    // Per-lane twiddles for the batch-8 Bailey column passes (8 columns per vector): lane_g[8*r + b] = w^(r*b), r < n1
    std::vector<double> lane_g_re, lane_g_im;
    size_t lane_g_N = 0;

    std::vector<double> c_w1r;
    std::vector<double> c_w1i;
    std::vector<double> c_w2r;
    std::vector<double> c_w2i;
    std::vector<double> c_w3r;
    std::vector<double> c_w3i;
};

#endif
