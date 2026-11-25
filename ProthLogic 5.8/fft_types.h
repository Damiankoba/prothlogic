// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

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
    //   Bailey 2D (two-pass) / fused square: [0] pass A, [1] fused middle (or pass B), [2] pass B'
    //   Stockham 1D: consecutive passes (radix-8, ..., radix-2/4 at the end)
    double   prof_pass[8];
    // parts of the fused-square passes (seconds per call, averaged over threads; *_max = busiest thread):
    //   [0..4]  pass A:  gather, stage 1 (two-level), rest of the FFT, store, busy_max
    //   [5..10] fusion:  gather, fwd FFT, square, inv FFT, store, busy_max
    //   [11..15] pass B': gather, stage 1, rest of the FFT, store, busy_max
    double   prof_sub[16];
#endif
    int      num_workers;      
    
    uint64_t k_val = 0;
    // Real-FFT squaring (2 FFTs instead of 3): 0 = self-test not run yet, 1 = OK, -1 = mismatch (old path is used)
    int sq_real_state = 0;
    // Calibration context (dcore.cpp calibrate_len): a failed IBDWT self-test only rejects the length, the calibration
    // reports it in its summary line
    bool quiet_selftest = false;

    
    FFTBackend backend;
    

    std::vector<double> r_w1r;
    std::vector<double> r_w1i;

    // Two-level twiddle table w^e = hi[e >> tw2_shift] * lo[e & (2^tw2_shift - 1)] (small, stays in L1/L2; replaces scattered reads of w1)
    std::vector<double> tw2_lo_re, tw2_lo_im, tw2_hi_re, tw2_hi_im;
    unsigned tw2_shift = 0;
    size_t tw2_N = 0;

    // Padded row layout of the fused squaring path (pack -> passes -> unpack): element j is stored at position
    // (j >> lay_lg) * (2^lay_lg + lay_pad) + (j & (2^lay_lg - 1)). lay_pad = 0: natural layout (all other paths).
    size_t lay_pad = 0;
    unsigned lay_lg = 0;
    // pack_zero_end > 0: the real-FFT pack zeroes only positions j < pack_zero_end above the digits (the rest of the
    // buffer is never read: the fused forward pass A reads only the lower half). 0 = zero the whole rest (all other paths).
    size_t pack_zero_end = 0;

    // Bits per FFT digit. 16 = classic path (4 digits per limb, n_limbs = fft_len/4). 17..20 = wider digits, only
    // in the real-FFT square (fft_square_real): n_limbs = fft_len*digit_bits/64, so a number of up to
    // fft_len*digit_bits bits fits a shorter FFT (e.g. N = 4.39M on 2^18 instead of 2^19). Chosen in dcore.cpp.
    unsigned digit_bits = 16;

    // Per-lane twiddles for the batch-8 Bailey column passes (8 columns per vector): lane_g[8*r + b] = w^(r*b), r < n1
    std::vector<double> lane_g_re, lane_g_im;
    size_t lane_g_N = 0;

    std::vector<double> c_w1r;
    std::vector<double> c_w1i;

    // IBDWT (irrational-base weighted transform for k*2^n+1, no zero padding; ibdwt_square in fft_engine.cpp).
    // L = 2*fft_len real digits at positions e_j = ceil(j*(n + log2 k)/L), weights a_j = 2^(e_j - j*beta), right-angle
    // negacyclic convolution through the complex FFT of length fft_len. Output k*Y^2, i.e. (negated) the scaled form.
    // digit_bits then only sets the limb count (n_limbs = fft_len*digit_bits/64 >= (n + 192)/64).
    bool ibdwt = false;
    uint64_t ib_k = 0;                     // tables below built for this (k, n)
    unsigned ib_n = 0;
    std::vector<uint32_t> ib_e;            // digit positions e_j, j = 0 .. L
    std::vector<double> ib_a;             // weights a_j
    std::vector<double> ib_kia;           // k / a_j (unweighting and the factor k of the output)
    std::vector<double> ib_tw_re, ib_tw_im;   // twist theta^j = exp(i*pi*j/L), j < fft_len
    unsigned ib_d = 0;                     // 2^(e_L) = 2^d * 2^n (top carry wrap)
    size_t ib_m = 0;                       // digit holding bit n ...
    unsigned ib_msh = 0;                   // ... at shift n - e_m
    // Two-level tables for the fast pack/unpack (digit j = 64*h + l): j*beta = (H part) + (L part) with 128-bit fractions,
    // e_j = IH[h] + IL[l] + c + nz (c = carry of the fraction sum, nz = fraction != 0), a_j = WH[h]*WL[l]*2^c,
    // k/a_j = k*iWH[h]*iWL[l]*2^-c, theta^j = TH[h]*TL[l]. H tables: L/64 + 1 entries, L tables: 64 entries.
    std::vector<uint64_t> ib_AHlo, ib_AHhi, ib_IH, ib_ALlo, ib_ALhi, ib_IL;
    std::vector<double> ib_WH, ib_iWH, ib_WL, ib_iWL;
    std::vector<double> ib_THr, ib_THi, ib_TLr, ib_TLi;
    // Digit state between squarings (fft_square_repeat): normalized digits w_j in [0, 2^b_j), b_j <= 30
    // (int32: half the memory traffic of pass A's gather and of the unpack)
    std::vector<int32_t> ib_w;
    // Normalization fused into pass B': the digits are normalized only inside runs of 8
    // (run i = digits 8i .. 8i+7); ib_rc[i] = carry out of run i, added to run i+1 by the next pack. Run 0 gets
    // ib_cin0 (-q of the top carry wrap) and digit ib_m gets ib_madd (r << ib_msh). ib_rc_live = false: none pending.
    std::vector<int64_t> ib_rc;
    int64_t ib_cin0 = 0;
    int64_t ib_madd = 0;
    bool ib_rc_live = false;
    // Digits overlaid on the FFT data (giants): the fused B' writes the digits of position j (lower j and
    // upper N + j, int32 each) into the 8 bytes of re at j (natural in-place layout) instead of ib_w; ib_h7[i] = top bit
    // of digit 7 of run i (the next run's h_prev: its neighbour in re may already hold FFT data). ib_ovl_write: the
    // fused B' writes this way; ib_ovl: the digit state is in re (pass A's gather reads it there).
    std::vector<uint8_t> ib_h7;
    bool ib_ovl_write = false;
    bool ib_ovl = false;
    // Pass B' fused with the next square's pass A (ibdwt_pass_BA): re/im already hold the output of pass A
    // of the digit state, the next square starts at the middle.
    bool ib_a_done = false;
    // Block-major order of the per-run data (ib_ri in fft_engine.cpp): run of j = half*N + K*C + 8*cb at
    // half*N/8 + cb*ib_R + K (ib_R = n1, C = n2 = 2^ib_lgC; set by ibdwt_square). ib_rc and ib_h7 use it; ib_bm = the size
    // masks of each run in that order (2 bytes), ib_wT = 6 arrays of (C/64)*R doubles at (j_low / 64 % (C/64))*R + K:
    // WH, iWH (lower digit j), THr, THi (twist), WH, iWH (upper N + j) - built by ib_build_blocked for (ib_T_k, ib_T_n, ib_T_C).
    size_t ib_R = 0;
    unsigned ib_lgC = 0;
    // Twiddles of the column gathers from a base every 8 rows (batch8_gather; classic real-FFT square only)
    bool tw_fast = false;
    // Relative speed of each thread in the IBDWT pass B'+A (adaptive contiguous ranges, ibdwt_pass_BA)
    std::vector<double> ba_w;
    std::vector<uint16_t> ib_bm;   // per run (ib_ri order): bit s = digit s has q0 + 1 bits, bit 8 + s = its weight carries x2
    std::vector<double> ib_wT;
    uint64_t ib_T_k = 0;
    unsigned ib_T_n = 0;
    size_t ib_T_C = 0;
    // Per digit, fixed for (k, n): bits 0..6 = size b_j, bit 7 = c_j (the weight carries a factor 2); L + 64 bytes
    std::vector<uint8_t> ib_b;
};

// Context that only the real-FFT square can serve: wide digits or an FFT length that is not a power of two (3*2^p).
// The old 3-FFT square, its self-test and the FFT multiplication know only 16-bit digits on powers of two, so such a
// context uses the GMP self-test and proth_logic.cpp multiplies through squarings. The IBDWT likewise.
inline bool fft_real_only(const FFTContext& c) {
    return c.digit_bits != 16 || (c.fft_len & (c.fft_len - 1)) != 0 || c.ibdwt;
}

#endif
