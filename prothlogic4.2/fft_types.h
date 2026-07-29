// Copyright (C) Damian Koba.

#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include <cstddef>
#include <cstdint>
enum class FFTBackend {
    INTERNAL
};

struct FFTContext {
    FFTBackend backend;
void* base_mem_ptr; // Do przechowywania oryginalnego adresu z aligned_alloc
void* base_ptr;
void* raw_allocation;
size_t total_allocation_size;
    size_t fft_len;
    int num_workers;
    
    // Bufory robocze (wyrównane do 64 bajtów dla AVX-512)
    double* re;
    double* im;
    double* re2;
    double* im2;

    // LUT dla Stockham radix-4
    double* w1_re;
    double* w1_im;
    double* w2_re;
    double* w2_im;
    double* w3_re;
    double* w3_im;

    // Wagi dla splotu Protha (DWT)
    double* proth_w_re;
    double* proth_w_im;
    double last_max_diff;

    double prof_pack = 0.0;
    double prof_fwd = 0.0;
    double prof_square = 0.0;
    double prof_inv = 0.0;
    double prof_unpack = 0.0;
    uint64_t prof_calls = 0;
    double prof_unpack_clear = 0.0;
    double prof_unpack_deweight = 0.0;
    double prof_unpack_scalar = 0.0;
    double prof_unpack_finalfix = 0.0;

    // --- BUFORY DLA KARATSUBY / OVERLAP-ADD ---
    double* k_A_re;
    double* k_A_im;
    double* k_B_re;
    double* k_B_im;
    double* k_AB_re;
    double* k_AB_im;
    uint64_t current_i = 0; // Dodaj to na końcu struktury

};

#endif
