// Copyright (C) Damian Koba.

#include "llr_fft_pack.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <immintrin.h> 
#include <vector>

void pack_karatsuba_limbs(const uint64_t* in, size_t n_limbs, FFTContext& ctx) {
    const size_t n_fft = ctx.fft_len;   
    const size_t split_chunks = n_fft / 2;
    
    std::memset(ctx.k_A_re, 0, n_fft * sizeof(double));
    std::memset(ctx.k_A_im, 0, n_fft * sizeof(double));
    std::memset(ctx.k_B_re, 0, n_fft * sizeof(double));
    std::memset(ctx.k_B_im, 0, n_fft * sizeof(double));
    
    const uint16_t* in16 = (const uint16_t*)in;
    size_t total_chunks = n_limbs * 4;   
    for (size_t j = 0; j < split_chunks; j += 8) {        
        __m512d v_A = _mm512_setzero_pd();
        if (j < total_chunks) {
            __m128i v_in16_A = _mm_loadu_si128((const __m128i*) & in16[j]);
            __m256i v_in32_A = _mm256_cvtepu16_epi32(v_in16_A);
            v_A = _mm512_cvtepi32_pd(v_in32_A);
        }        
        _mm512_store_pd(&ctx.k_A_re[j], v_A);       
        __m512d v_B = _mm512_setzero_pd();
        size_t j_B = j + split_chunks;
        if (j_B < total_chunks) {
            __m128i v_in16_B = _mm_loadu_si128((const __m128i*) & in16[j_B]);
            __m256i v_in32_B = _mm256_cvtepu16_epi32(v_in16_B);
            v_B = _mm512_cvtepi32_pd(v_in32_B);
        }        
        _mm512_store_pd(&ctx.k_B_re[j], v_B);
    }    
    _mm_sfence();
}



