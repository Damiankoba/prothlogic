#include "fft_pack_vbmi.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <immintrin.h>


void direct_pack_square(const uint64_t* limbs, size_t n_limbs, FFTContext& ctx) {
    size_t split_chunks = ctx.fft_len / 2;
    size_t limbs_per_split = split_chunks / 4;

#pragma omp parallel
    {
        
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < limbs_per_split; i += 4) {
            if (i + 3 < n_limbs) {
                __m256i v_in = _mm256_loadu_si256((const __m256i*) & limbs[i]);
                __m512i v_int32 = _mm512_cvtepu16_epi32(v_in);
                _mm512_store_pd(&ctx.re[i * 4], _mm512_cvtepi32_pd(_mm512_castsi512_si256(v_int32)));
                _mm512_store_pd(&ctx.re[i * 4 + 8], _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v_int32, 1)));
            }
            else {
                for (size_t k = 0; k < 4; k++) {
                    if (i + k < n_limbs) {
                        uint64_t L = limbs[i + k];
                        ctx.re[(i + k) * 4 + 0] = static_cast<double>(L & 0xFFFF);
                        ctx.re[(i + k) * 4 + 1] = static_cast<double>((L >> 16) & 0xFFFF);
                        ctx.re[(i + k) * 4 + 2] = static_cast<double>((L >> 32) & 0xFFFF);
                        ctx.re[(i + k) * 4 + 3] = static_cast<double>((L >> 48) & 0xFFFF);
                    }
                    else if ((i + k) * 4 < split_chunks) {
                        ctx.re[(i + k) * 4 + 0] = 0.0; ctx.re[(i + k) * 4 + 1] = 0.0;
                        ctx.re[(i + k) * 4 + 2] = 0.0; ctx.re[(i + k) * 4 + 3] = 0.0;
                    }
                }
            }
        }

        
#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < limbs_per_split; i += 4) {
            size_t actual_i = limbs_per_split + i;
            if (actual_i + 3 < n_limbs) {
                __m256i v_in = _mm256_loadu_si256((const __m256i*) & limbs[actual_i]);
                __m512i v_int32 = _mm512_cvtepu16_epi32(v_in);
                _mm512_store_pd(&ctx.im[i * 4], _mm512_cvtepi32_pd(_mm512_castsi512_si256(v_int32)));
                _mm512_store_pd(&ctx.im[i * 4 + 8], _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v_int32, 1)));
            }
            else {
                for (size_t k = 0; k < 4; k++) {
                    if (actual_i + k < n_limbs) {
                        uint64_t L = limbs[actual_i + k];
                        ctx.im[(i + k) * 4 + 0] = static_cast<double>(L & 0xFFFF);
                        ctx.im[(i + k) * 4 + 1] = static_cast<double>((L >> 16) & 0xFFFF);
                        ctx.im[(i + k) * 4 + 2] = static_cast<double>((L >> 32) & 0xFFFF);
                        ctx.im[(i + k) * 4 + 3] = static_cast<double>((L >> 48) & 0xFFFF);
                    }
                    else if ((i + k) * 4 < split_chunks) {
                        ctx.im[(i + k) * 4 + 0] = 0.0; ctx.im[(i + k) * 4 + 1] = 0.0;
                        ctx.im[(i + k) * 4 + 2] = 0.0; ctx.im[(i + k) * 4 + 3] = 0.0;
                    }
                }
            }
        }

        
#pragma omp for schedule(static) nowait
        for (size_t i = split_chunks; i < ctx.fft_len; i += 8) {
            _mm512_store_pd(&ctx.re[i], _mm512_setzero_pd());
            _mm512_store_pd(&ctx.im[i], _mm512_setzero_pd());
        }
    }
}

void pack_to_fft_buffers(const uint64_t* limbs, size_t n_limbs, double* out, size_t n_fft) {
    size_t i = 0;
    for (; i + 3 < n_limbs; i += 4) {
        __m256i v_in = _mm256_loadu_si256((const __m256i*) & limbs[i]);
        __m512i v_int32 = _mm512_cvtepu16_epi32(v_in);
        _mm512_storeu_pd(&out[i * 4], _mm512_cvtepi32_pd(_mm512_castsi512_si256(v_int32)));
        _mm512_storeu_pd(&out[i * 4 + 8], _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v_int32, 1)));
    }
    for (; i < n_limbs; i++) {
        uint64_t L = limbs[i];
        out[i * 4 + 0] = static_cast<double>(L & 0xFFFF);
        out[i * 4 + 1] = static_cast<double>((L >> 16) & 0xFFFF);
        out[i * 4 + 2] = static_cast<double>((L >> 32) & 0xFFFF);
        out[i * 4 + 3] = static_cast<double>((L >> 48) & 0xFFFF);
    }
    for (size_t j = n_limbs * 4; j < n_fft; j++) {
        out[j] = 0.0;
    }
}


void unpack_karatsuba_overlap_add_vbmi(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val, int target_bits, double* ptr_A2, double* ptr_AB, double* ptr_B2) {
    const size_t n_fft = ctx.fft_len;
    const size_t split_chunks = n_fft / 2;

    size_t max_chunks = n_limbs * 8;
    size_t double_limbs = n_limbs * 2 + 2;

    thread_local std::vector<uint64_t> sq;
    thread_local std::vector<uint64_t> H_arr;
    thread_local std::vector<uint64_t> Q_arr;

    sq.assign(double_limbs, 0);
    H_arr.assign(n_limbs + 2, 0);
    Q_arr.assign(n_limbs + 2, 0);

    
    int64_t* raw_sums = (int64_t*)ctx.g_C_re;
    double global_max_err = 0.0;

#pragma omp parallel
    {
        __m512d local_max_err = _mm512_setzero_pd();

#pragma omp for schedule(static)
        for (size_t i = 0; i < max_chunks; i += 8) {
            
            __m512d v_A2 = (i < n_fft) ? _mm512_load_pd(&ptr_A2[i]) : _mm512_setzero_pd();
            __m512d v_AB = ((i >= split_chunks) && ((i - split_chunks) < n_fft))
                ? _mm512_load_pd(&ptr_AB[i - split_chunks])
                : _mm512_setzero_pd();
            __m512d v_B2 = ((i >= 2 * split_chunks) && ((i - 2 * split_chunks) < n_fft))
                ? _mm512_load_pd(&ptr_B2[i - 2 * split_chunks])
                : _mm512_setzero_pd();

            __m512d v_sum = _mm512_add_pd(_mm512_add_pd(v_A2, v_AB), v_B2);
            __m512i v_int = _mm512_cvtpd_epi64(_mm512_roundscale_pd(v_sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m512d v_rounded_back = _mm512_cvtepi64_pd(v_int);
            __m512d v_err = _mm512_sub_pd(v_sum, v_rounded_back);

            local_max_err = _mm512_max_pd(local_max_err, _mm512_abs_pd(v_err));
            _mm512_store_si512((__m512i*)&raw_sums[i], v_int);
        }

        alignas(64) double err_arr[8];
        _mm512_store_pd(err_arr, local_max_err);
        double thread_max = 0.0;
        for (int i = 0; i < 8; i++) {
            if (err_arr[i] > thread_max) thread_max = err_arr[i];
        }

#pragma omp critical
        {
            if (thread_max > global_max_err) global_max_err = thread_max;
        }
    }
    ctx.last_max_diff = global_max_err;

    int64_t carry = 0;
    for (size_t i = 0; i < max_chunks; i += 8) {
        size_t limb_base = i >> 2;
        uint64_t acc0 = 0;
        uint64_t acc1 = 0;

        for (size_t k = 0; k < 8; k++) {
            int64_t ival = raw_sums[i + k] + carry;
            carry = ival >> 16;
            uint64_t chunk = (uint64_t)(ival & 0xFFFF);
            if (k < 4) acc0 |= (chunk << (k << 4));
            else       acc1 |= (chunk << ((k - 4) << 4));
        }
        if (limb_base < double_limbs) sq[limb_base] |= acc0;
        if (limb_base + 1 < double_limbs) sq[limb_base + 1] |= acc1;
    }

    size_t carry_idx = max_chunks >> 2;
    while (carry > 0 && carry_idx < sq.size()) {
        unsigned __int128 sum = (unsigned __int128)sq[carry_idx] + carry;
        sq[carry_idx] = (uint64_t)sum;
        carry = (int64_t)(sum >> 64);
        carry_idx++;
    }

    size_t split_q = n_bits / 64;
    unsigned split_r = n_bits % 64;

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
        asm volatile ("divq %[k_val]" : "=a" (q_val), "=d" (r_val) : "d" (high), "a" (low), [k_val] "c" (k_val) : "cc");
        Q_arr[i] = q_val;
        rem = r_val;
    }

    std::memset(out, 0, n_limbs * sizeof(uint64_t));
    for (size_t i = 0; i < split_q && i < n_limbs; i++) out[i] = sq[i];
    if (split_r > 0 && split_q < n_limbs) out[split_q] = sq[split_q] & ((1ULL << split_r) - 1);

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
            if (i == split_q) val_to_add += (split_r == 0) ? k_val : (k_val << split_r);
            if (i == split_q + 1 && split_r > 0) val_to_add += (k_val >> (64 - split_r));
            unsigned __int128 sum = (unsigned __int128)out[i] + val_to_add;
            out[i] = (uint64_t)sum;
            local_carry += (uint64_t)(sum >> 64);
        }
    }
}