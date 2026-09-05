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
    int      num_workers;      
    
    uint64_t k_val = 0;
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

    std::vector<double> c_w1r;
    std::vector<double> c_w1i;
    std::vector<double> c_w2r;
    std::vector<double> c_w2i;
    std::vector<double> c_w3r;
    std::vector<double> c_w3i;
};

#endif
