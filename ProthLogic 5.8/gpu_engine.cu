// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gpu_engine.h"
#include "log_config.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

// The check kernels: 2^-n / 2^n powering without multiplications, O(1) k membership test.
// LOTR_SIEVE_PROFILE=1 prints sieve stage timings.
static const bool g_sieve_prof = (std::getenv("LOTR_SIEVE_PROFILE") != nullptr);
static double g_t_gen = 0.0, g_t_extract = 0.0, g_t_check = 0.0;   // stage times [s]: window / prime extraction / checking
static std::vector<uint32_t> g_sparse_perm;                         // sort permutation of k in the sparse sieve
static uint64_t g_sparse_kmin = 0, g_sparse_kmax = 0;

// Window generation: small primes (< SIEVE_TILE_PRIME_LIMIT) sieve tiles in shared memory,
// large ones - one warp per prime (atomicAnd in global memory).
#define SIEVE_TILE_WORDS 8192                          // 32 KB of shared memory per tile
#define SIEVE_TILE_BITS (SIEVE_TILE_WORDS * 32)        // 262144 bits = 524288 numbers
#define SIEVE_TILE_PRIME_LIMIT 65536
static uint32_t g_num_tile_primes = 0;                 // how many leading (sorted) primes go through the tiles

struct SieveTimer {
    double& acc;
    std::chrono::high_resolution_clock::time_point t0;
    explicit SieveTimer(double& a) : acc(a), t0(std::chrono::high_resolution_clock::now()) {}
    ~SieveTimer() { acc += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count(); }
};

static void sieve_prof_print(const char* what) {
    if (!g_sieve_prof) return;
    std::printf("\n[SIEVE-PROFILE %s] window generation=%.1f s  prime extraction=%.1f s  checking=%.1f s\n",
        what, g_t_gen, g_t_extract, g_t_check);
    g_t_gen = g_t_extract = g_t_check = 0.0;
}
static uint32_t* d_vram_bit_mask = nullptr;      
static uint64_t* d_internal_prime_batch = nullptr; 
static int* d_internal_prime_count = nullptr;      
static uint64_t* d_small_primes_vram = nullptr;
static uint8_t* d_is_candidate = nullptr;
static size_t current_range_size = 0;

static uint8_t* d_sparse_is_active = nullptr;
static uint64_t* d_sparse_ks = nullptr;
static uint32_t current_num_ks = 0;
static uint64_t current_allocated_window_size = 0;

extern "C" void load_small_primes_to_gpu(const uint64_t* small_primes, uint64_t num_small_primes) {
    if (d_small_primes_vram != nullptr) {
        cudaFree(d_small_primes_vram);
    }
    cudaMalloc(&d_small_primes_vram, num_small_primes * sizeof(uint64_t));
    cudaMemcpy(d_small_primes_vram, small_primes, num_small_primes * sizeof(uint64_t), cudaMemcpyHostToDevice);
    g_num_tile_primes = 0;
    while (g_num_tile_primes < num_small_primes && small_primes[g_num_tile_primes] < SIEVE_TILE_PRIME_LIMIT) ++g_num_tile_primes;
}

// =========================================================================
// WINDOW GENERATION
// =========================================================================
// Small primes: a tile of SIEVE_TILE_BITS bits (odd numbers) in shared memory. Window bit i corresponds to
// the number window_start + 2*i + 1. Writes the WHOLE tile to global memory (no prior cudaMemset needed).
__global__ void mark_small_tiled_kernel(
    uint32_t* d_bitmask,
    uint64_t window_start,
    uint64_t window_size,
    const uint64_t* small_primes,
    uint32_t num_tile_primes)
{
    __shared__ uint32_t tile[SIEVE_TILE_WORDS];

    const uint64_t elements = window_size / 2;
    const uint64_t base_bit = (uint64_t)blockIdx.x * SIEVE_TILE_BITS;
    if (base_bit >= elements) return;
    const uint64_t rest = elements - base_bit;
    const uint32_t tile_bits = (uint32_t)(rest < (uint64_t)SIEVE_TILE_BITS ? rest : (uint64_t)SIEVE_TILE_BITS);
    const uint32_t tile_words = (tile_bits + 31) / 32;

    for (uint32_t i = threadIdx.x; i < SIEVE_TILE_WORDS; i += blockDim.x) tile[i] = 0xFFFFFFFFu;
    __syncthreads();

    const uint64_t X0 = window_start + 2 * base_bit + 1;      // number corresponding to bit 0 of the tile (odd)
    const uint64_t window_end = window_start + window_size;

    for (uint32_t pi = threadIdx.x; pi < num_tile_primes; pi += blockDim.x) {
        const uint64_t p = small_primes[pi];
        if (p < 3 || p * p >= window_end) continue;
        uint64_t m = ((X0 + p - 1) / p) * p;                   // first multiple >= X0
        if (m < p * p) m = p * p;
        if ((m & 1ULL) == 0) m += p;                           // odd multiples only
        for (uint64_t b = (m - X0) >> 1; b < tile_bits; b += p) {
            atomicAnd(&tile[b >> 5], ~(1u << (b & 31)));
        }
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < tile_words; i += blockDim.x) {
        d_bitmask[(base_bit >> 5) + i] = tile[i];
    }
}

// Large primes: one warp per prime (the start offset is computed once per warp, not per thread).
__global__ void mark_large_warp_kernel(
    uint32_t* d_bitmask,
    uint64_t window_start,
    uint64_t window_size,
    const uint64_t* primes,
    uint64_t num_primes)
{
    const uint64_t warp_global = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31;
    if (warp_global >= num_primes) return;

    const uint64_t p = primes[warp_global];
    if (p < 3) return;
    const uint64_t window_end = window_start + window_size;
    if (p * p >= window_end) return;

    uint64_t first_mult = ((window_start + p - 1) / p) * p;
    if (first_mult < p * p) first_mult = p * p;
    if ((first_mult & 1ULL) == 0) first_mult += p;
    if (first_mult >= window_end) return;

    const uint64_t first_bit = (first_mult - window_start) / 2;
    const uint64_t elements = window_size / 2;
    for (uint64_t i = first_bit + (uint64_t)lane * p; i < elements; i += 32ULL * p) {
        atomicAnd(&d_bitmask[i >> 5], ~(1u << (i & 31)));
    }
}

extern "C" void generate_sieve_window_on_gpu_vram(
    uint64_t window_start,
    uint64_t window_size,
    uint64_t num_small_primes)
{
    uint64_t elements = window_size / 2;
    uint32_t num_words = (elements + 31) / 32;
    size_t required_bytes = num_words * sizeof(uint32_t);
    SieveTimer _t_gen(g_t_gen);

    if (window_size != current_allocated_window_size) {
        if (d_vram_bit_mask != nullptr) {
            cudaFree(d_vram_bit_mask);
            d_vram_bit_mask = nullptr;
        }
        cudaError_t err = cudaMalloc(&d_vram_bit_mask, required_bytes);
        if (err != cudaSuccess) {
            printf("\n[FATAL] Failed to allocate %zu bytes of VRAM for bitmask! Reason: %s\n", required_bytes, cudaGetErrorString(err));
            exit(1);
        }
        current_allocated_window_size = window_size;
    }

    // 1) small primes: tiles in shared memory (they write the whole bitmask, no cudaMemset)
    uint32_t num_tile_primes = (g_num_tile_primes < num_small_primes) ? g_num_tile_primes : (uint32_t)num_small_primes;
    uint64_t num_tiles = (elements + SIEVE_TILE_BITS - 1) / SIEVE_TILE_BITS;
    if (num_tiles > 0) {
        mark_small_tiled_kernel<<<(unsigned)num_tiles, 256>>>(
            d_vram_bit_mask, window_start, window_size, d_small_primes_vram, num_tile_primes
        );
    }
    // 2) large primes: one warp per prime (4 warps in a block of 128 threads)
    uint64_t num_large = (num_small_primes > num_tile_primes) ? (num_small_primes - num_tile_primes) : 0;
    if (num_large > 0) {
        uint64_t blocks = (num_large + 3) / 4;
        mark_large_warp_kernel<<<(unsigned)blocks, 128>>>(
            d_vram_bit_mask, window_start, window_size, d_small_primes_vram + num_tile_primes, num_large
        );
    }

    cudaDeviceSynchronize();
}



__global__ void extract_primes_kernel(
    const uint32_t* d_window_bitmask, 
    uint64_t window_start, 
    uint64_t pmax, 
    uint64_t odd_elements_in_window, 
    uint64_t* d_prime_batch, 
    int* d_prime_count) 
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32; 
    
    int is_prime = 0;
    uint64_t p = 0;
    
   
    if (i < odd_elements_in_window) {
        if (d_window_bitmask[i >> 5] & (1U << (i & 31))) {
            p = window_start + 2 * i + 1;
            if (p <= pmax && p >= 3) {
                is_prime = 1;
            }
        }
    }
    
    
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, is_prime);
    
    
    if (active_mask != 0) {
        
        int leader = __ffs(active_mask) - 1; 
        int warp_offset = 0;
        
        
        if (lane_id == leader) {
            warp_offset = atomicAdd(d_prime_count, __popc(active_mask));
        }
        
        
        warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, leader);
        
        
        if (is_prime) {
            
            int my_offset = warp_offset + __popc(active_mask & ((1 << lane_id) - 1));
            d_prime_batch[my_offset] = p;
        }
    }
}

extern "C" int extract_primes_on_gpu(uint64_t window_start, uint64_t window_size, uint64_t pmax) {
    uint64_t odd_elements = window_size / 2;
    SieveTimer _t_ext(g_t_extract);
    int threads = 256;
    int blocks = (odd_elements + threads - 1) / threads;

    extract_primes_kernel<<<blocks, threads>>>(
        d_vram_bit_mask, window_start, pmax, odd_elements, d_internal_prime_batch, d_internal_prime_count
    );
    
    // HARD GUARD AGAINST A SILENT KERNEL FAILURE
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("\n[FATAL KERNEL ERROR] extract_primes_kernel crashed! Reason: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);
    return current_count;
}

extern "C" int extract_sparse_primes_on_gpu(uint64_t window_start, uint64_t window_size, uint64_t pmax) {
    return extract_primes_on_gpu(window_start, window_size, pmax);
}



__device__ __forceinline__ uint64_t calc_p_inv(uint64_t p) {
    uint64_t inv = p;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    return ~inv + 1ULL; 
}


__device__ __forceinline__ uint64_t monty_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t p_inv) {
    uint64_t T_lo = a * b;
    uint64_t T_hi = __umul64hi(a, b); // Native PTX: 64x64 -> upper 64 bits
    
    uint64_t m = T_lo * p_inv;
    
    uint64_t mp_lo = m * p;
    uint64_t mp_hi = __umul64hi(m, p);
    
    uint64_t carry = (T_lo + mp_lo < T_lo) ? 1ULL : 0ULL;
    uint64_t t = T_hi + mp_hi + carry;
    
    if (t >= p) t -= p;
    return t;
}

// =========================================================================
// FULL SIEVE
// =========================================================================
// -------------------------------------------------------------------------
// FAST POWERS OF TWO (odd p < 2^63), Montgomery, exponentiation from the most significant bit:
// square + (multiply by 2 or by 1/2) is just an add/shift, with no second modular multiplication
// and no R^2 mod p overhead (128-bit %). ~23 multiplications instead of ~70 + a 128-bit division per prime.
// -------------------------------------------------------------------------
// 2^(-n) mod p
__device__ __forceinline__ uint64_t inv_pow2_mod(uint32_t n, uint64_t p) {
    if (n == 0) return 1 % p;
    const uint64_t p_inv = calc_p_inv(p);
    uint64_t x = (0ULL - p) % p;   // R mod p (R = 2^64) = Montgomery representation of 1
    for (int b = 31 - __clz(n); b >= 0; --b) {
        x = monty_mul(x, x, p, p_inv);
        if ((n >> b) & 1u) {
            x = (x & 1ULL) ? ((x + p) >> 1) : (x >> 1);   // x/2 mod p (x < p < 2^63, so x + p < 2^64)
        }
    }
    return monty_mul(x, 1ULL, p, p_inv);
}

// 2^n mod p
__device__ __forceinline__ uint64_t pow2_mod(uint32_t n, uint64_t p) {
    if (n == 0) return 1 % p;
    const uint64_t p_inv = calc_p_inv(p);
    uint64_t x = (0ULL - p) % p;
    for (int b = 31 - __clz(n); b >= 0; --b) {
        x = monty_mul(x, x, p, p_inv);
        if ((n >> b) & 1u) {
            x <<= 1;
            if (x >= p) x -= p;
        }
    }
    return monty_mul(x, 1ULL, p, p_inv);
}

// Full sieve (range of k, fixed n): for p, candidate k is removed when k = -2^(-n) (mod p).
__global__ void check_proth_candidates_kernel_fast(
    uint8_t* is_candidate,
    const uint64_t* primes_batch,
    uint32_t n,
    uint64_t k_start,
    uint64_t k_end,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t p = primes_batch[idx];
    if (p < 3) return;

    uint64_t inv_2n = inv_pow2_mod(n, p);
    if (inv_2n == 0) return;

    uint64_t target_k_mod = p - inv_2n;   // inv_2n in [1, p-1]
    uint64_t rem = k_start % p;

    uint64_t step_to_first = (target_k_mod >= rem) ? (target_k_mod - rem) : (p - rem + target_k_mod);
    uint64_t first_k = k_start + step_to_first;
    if (first_k > k_end) return;

    for (uint64_t k = first_k; k <= k_end; k += p) {
        is_candidate[k - k_start] = 0;
    }
}

// Sparse sieve (list of k, fixed n): for p > k_max we have k mod p == k, so it is enough to compare the target t = -2^(-n) mod p
// with the list (one range check, binary search only on hits). For small p (rare) a full % as before.
__global__ void check_sparse_candidates_kernel_fast(
    uint8_t* __restrict__ is_active_k,
    const uint64_t* __restrict__ ks_sorted,
    uint32_t num_ks,
    uint64_t k_min,
    uint64_t k_max,
    const uint64_t* __restrict__ primes,
    uint64_t count,
    uint32_t n)
{
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t p = primes[idx];
    if (p < 3) return;

    uint64_t inv_2n = inv_pow2_mod(n, p);
    if (inv_2n == 0) return;
    uint64_t t = p - inv_2n;

    if (p > k_max) {
        if (t < k_min || t > k_max) return;
        uint32_t lo = 0, hi = num_ks;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (ks_sorted[mid] < t) lo = mid + 1; else hi = mid;
        }
        for (uint32_t j = lo; j < num_ks && ks_sorted[j] == t; ++j) is_active_k[j] = 0;   // duplicate k values too
    }
    else {
        for (uint32_t i = 0; i < num_ks; ++i) {
            if ((ks_sorted[i] % p) == t) is_active_k[i] = 0;
        }
    }
}

// Sieve over N (fixed k): k * 2^n = -1 (mod p). Instead of the inverse of k (Fermat, ~75 multiplications) compute c = k * 2^n_start mod p
// and loop c = 2c mod p; a hit when c == p - 1.
__global__ void check_proth_candidates_n_range_kernel_fast(
    uint8_t* is_candidate,
    const uint64_t* primes_batch,
    uint64_t k_val,
    uint32_t n_start,
    uint32_t n_end,
    uint64_t count)
{
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t p = primes_batch[idx];
    if (p < 3) return;

    uint64_t k_mod = k_val % p;
    if (k_mod == 0) return;

    uint64_t c = (uint64_t)(((unsigned __int128)k_mod * pow2_mod(n_start, p)) % p);
    for (uint32_t n_val = n_start; n_val <= n_end; ++n_val) {
        if (c == p - 1) {
            is_candidate[n_val - n_start] = 0;
        }
        c <<= 1;
        if (c >= p) c -= p;
    }
}

// -------------------------------------------------------------------------
// Sieve over N by baby-step giant-step: the plain kernel above walks the whole range for every prime
// (R doublings, nearly all of the sieve time at R = 200000), although a large prime almost never hits more than one n.
// Write n = n_start + g*m + j (0 <= j < m); with c0 = k * 2^n_start mod p:
//     k * 2^n = -1  <=>  c0 * 2^(g*m) = -2^(-j)   (mod p)
// baby steps: the m values -2^(-j) (halving only, no multiplication) go to a hash table in shared memory;
// giant steps: G = c0 * (2^m)^g, ceil(R/m) Montgomery multiplications, each looked up in the table -> n = n_start + g*m + j.
// No modular inverse is needed. Exact: if ord_p(2) >= m the baby values are distinct, every giant step matches at most
// one j, and two solutions inside one block of m would differ by a multiple of ord_p(2) >= m. If ord_p(2) < m a
// repeated baby value shows it and that prime falls back to the plain walk (rare: divisors of 2^o - 1, o < m).
// One warp per prime: the lanes share the baby steps (m/32 each) and the giant steps (ceil(R/m)/32 each).
// LOTR_NSIEVE_BSGS=0 -> the plain kernel (for comparisons).
// -------------------------------------------------------------------------
#define BSGS_M 256                 // baby steps per prime (j < 2^16: the index is a uint16_t)
#define BSGS_HBITS 9
#define BSGS_H (1 << BSGS_HBITS)   // hash slots per warp (load 50%)
#define BSGS_WARPS 4               // warps (= primes) per block of 128 threads; shared memory 4 x 5 KB
#define BSGS_MIN_RANGE 4096        // shorter N ranges: the plain walk

__device__ __forceinline__ uint32_t bsgs_hash(uint64_t v) {
    return ((uint32_t)(v ^ (v >> 29)) * 0x9E3779B1u) >> (32 - BSGS_HBITS);
}

__global__ void __launch_bounds__(BSGS_WARPS * 32) check_n_range_bsgs_kernel(
    uint8_t* is_candidate,
    const uint64_t* primes_batch,
    uint64_t k_val,
    uint32_t n_start,
    uint32_t range,
    uint64_t count)
{
    __shared__ unsigned long long s_val[BSGS_WARPS][BSGS_H];   // 0 = empty (a baby value is never 0)
    __shared__ uint16_t s_idx[BSGS_WARPS][BSGS_H];

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t w = threadIdx.x >> 5;
    const uint64_t prime_idx = (uint64_t)blockIdx.x * BSGS_WARPS + w;
    if (prime_idx >= count) return;                 // the whole warp: one prime per warp

    const uint64_t p = primes_batch[prime_idx];
    if (p < 3) return;
    const uint64_t k_mod = k_val % p;
    if (k_mod == 0) return;                         // p | k: p never divides k*2^n+1
    const uint64_t p_inv = calc_p_inv(p);

    unsigned long long* tab = s_val[w];
    uint16_t* idx = s_idx[w];
    for (uint32_t s = lane; s < BSGS_H; s += 32) tab[s] = 0ULL;
    __syncwarp();

    // baby steps: lane L stores j = L*(m/32) .. L*(m/32) + m/32 - 1, the value -2^(-j) mod p
    const uint32_t per_lane = BSGS_M / 32;
    uint64_t x = inv_pow2_mod(lane * per_lane, p);  // 2^(-j) for the lane's first j
    bool dup = false;
    for (uint32_t t = 0; t < per_lane; ++t) {
        const unsigned long long b = p - x;
        uint32_t s = bsgs_hash(b);
        while (true) {
            const unsigned long long old = atomicCAS(&tab[s], 0ULL, b);
            if (old == 0ULL) { idx[s] = (uint16_t)(lane * per_lane + t); break; }
            if (old == b) { dup = true; break; }    // the same value twice: ord_p(2) < m
            s = (s + 1) & (BSGS_H - 1);
        }
        x = (x & 1ULL) ? ((x + p) >> 1) : (x >> 1); // x/2 mod p
    }
    const bool fallback = __any_sync(0xFFFFFFFFu, dup);
    __syncwarp();

    if (fallback) {                                 // ord_p(2) < m: the plain walk, the range split between the lanes
        const uint32_t ws = (range + 31) / 32;
        const uint32_t a = lane * ws;
        if (a >= range) return;
        const uint32_t e = (a + ws < range) ? a + ws : range;
        uint64_t c = monty_mul(k_mod, pow2_mod(n_start + a + 64, p), p, p_inv);   // k * 2^(n_start + a)
        for (uint32_t i = a; i < e; ++i) {
            if (c == p - 1) is_candidate[i] = 0;
            c <<= 1;
            if (c >= p) c -= p;
        }
        return;
    }

    // giant steps: lane L takes g = g0 .. g1-1
    const uint32_t giants = (range + BSGS_M - 1) / BSGS_M;
    const uint32_t gs = (giants + 31) / 32;
    const uint32_t g0 = lane * gs;
    if (g0 >= giants) return;
    const uint32_t g1 = (g0 + gs < giants) ? g0 + gs : giants;
    const uint64_t m_mont = pow2_mod(BSGS_M + 64, p);                                 // 2^m in Montgomery form
    uint64_t G = monty_mul(k_mod, pow2_mod(n_start + g0 * BSGS_M + 64, p), p, p_inv);  // k * 2^(n_start + g0*m)
    for (uint32_t g = g0; g < g1; ++g) {
        uint32_t s = bsgs_hash(G);
        unsigned long long v;
        while ((v = tab[s]) != 0ULL) {
            if (v == G) {
                const uint32_t off = g * BSGS_M + idx[s];
                if (off < range) is_candidate[off] = 0;
                break;
            }
            s = (s + 1) & (BSGS_H - 1);
        }
        G = monty_mul(G, m_mont, p, p_inv);         // G * 2^m
    }
}

// -------------------------------------------------------------------------
// v2: v1 with a smaller operation count and more warps per SM:
//  - no 64-bit division: 2^64 mod p and k mod p through a double quotient (error <= 1, one correction); p < 2^32 or
//    k >= 2^53 (rare) the plain %;
//  - p_inv once; powers of two in Montgomery form straight from one = 2^64 mod p;
//  - a table entry is ONE 32-bit word: a 24-bit fingerprint of the value + j (8 bits) -> 2 KB per warp. A fingerprint match
//    is only a candidate: G * 2^j == -1 is checked exactly (j doublings) before n is removed, so a false match never
//    removes anything; a fingerprint repeat while inserting -> the plain walk (a real repeat = ord_p(2) < m; a false one
//    ~2e-5 of primes, only slower);
//  - baby steps interleaved (lane L: j = L + 32t): start 2^(-L) by L halvings, then * 2^(-32) (one multiplication);
//  - __launch_bounds__ with a minimum of blocks per SM (<= 64 registers).
// -------------------------------------------------------------------------
#define BSGS2_WARPS 8                                  // warps (= primes) per block of 256 threads; 8 x 2 KB
#define BSGS2_EMPTY 0xFFFFFFFFu

__device__ __forceinline__ uint64_t bsgs2_r64(uint64_t p) {        // 2^64 mod p (p odd)
    if (p < (1ULL << 32) || p >= (1ULL << 53)) return (0ULL - p) % p;
    const uint64_t q = (uint64_t)(18446744073709551616.0 / (double)p);   // < 2^32, off by at most 1
    int64_t r = (int64_t)(0ULL - q * p);                                  // 2^64 - q*p, in (-p, 2p)
    if (r < 0) r += (int64_t)p;
    else if (r >= (int64_t)p) r -= (int64_t)p;
    return (uint64_t)r;
}

__device__ __forceinline__ uint64_t bsgs2_mod(uint64_t k, uint64_t p) {     // k mod p
    if (k < p) return k;
    if (k >= (1ULL << 53)) return k % p;
    const uint64_t q = (uint64_t)((double)k / (double)p);                 // exact operands, off by at most 1
    int64_t r = (int64_t)(k - q * p);
    if (r < 0) r += (int64_t)p;
    else if (r >= (int64_t)p) r -= (int64_t)p;
    return (uint64_t)r;
}

// 2^e in Montgomery form (2^e * 2^64 mod p); one = 2^64 mod p
__device__ __forceinline__ uint64_t bsgs2_mpow2(uint32_t e, uint64_t one, uint64_t p, uint64_t p_inv) {
    uint64_t x = one;
    for (int b = 31 - __clz(e); b >= 0; --b) {     // e == 0: no step
        x = monty_mul(x, x, p, p_inv);
        if ((e >> b) & 1u) { x <<= 1; if (x >= p) x -= p; }
    }
    return x;
}

// slot (top BSGS_HBITS bits) and fingerprint (24 other bits, never 0xFFFFFF, so no entry equals BSGS2_EMPTY)
__device__ __forceinline__ void bsgs2_key(uint64_t v, uint32_t& slot, uint32_t& fp) {
    const uint64_t h = v * 0x9E3779B97F4A7C15ULL;
    slot = (uint32_t)(h >> (64 - BSGS_HBITS));
    fp = (uint32_t)(h >> 31) & 0xFFFFFFu;
    if (fp == 0xFFFFFFu) fp = 0xFFFFFEu;
}

__global__ void __launch_bounds__(BSGS2_WARPS * 32, 4) check_n_range_bsgs2_kernel(
    uint8_t* is_candidate,
    const uint64_t* primes_batch,
    uint64_t k_val,
    uint32_t n_start,
    uint32_t range,
    uint64_t count)
{
    __shared__ uint32_t s_tab[BSGS2_WARPS][BSGS_H];

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t w = threadIdx.x >> 5;
    const uint64_t prime_idx = (uint64_t)blockIdx.x * BSGS2_WARPS + w;
    if (prime_idx >= count) return;                 // the whole warp: one prime per warp

    const uint64_t p = primes_batch[prime_idx];
    if (p < 3) return;
    const uint64_t k_mod = bsgs2_mod(k_val, p);
    if (k_mod == 0) return;                         // p | k: p never divides k*2^n+1
    const uint64_t p_inv = calc_p_inv(p);
    const uint64_t one = bsgs2_r64(p);              // 1 in Montgomery form

    uint32_t* tab = s_tab[w];
    for (uint32_t s = lane; s < BSGS_H; s += 32) tab[s] = BSGS2_EMPTY;
    __syncwarp();

    // baby steps: lane L stores j = L + 32t (t = 0 .. m/32-1), the value -2^(-j) mod p
    uint64_t x = 1;                                 // 2^(-L): L halvings
    for (uint32_t t = 0; t < lane; ++t) x = (x & 1ULL) ? ((x + p) >> 1) : (x >> 1);
    const uint64_t step = (p > (1ULL << 32)) ? (1ULL << 32) : ((1ULL << 32) % p);   // 2^(-32) in Montgomery form = 2^32 mod p
    bool dup = false;
    for (uint32_t t = 0; t < BSGS_M / 32; ++t) {
        const uint64_t b = p - x;
        uint32_t slot, fp;
        bsgs2_key(b, slot, fp);
        const uint32_t entry = (fp << 8) | (lane + 32 * t);
        while (true) {
            const uint32_t old = atomicCAS(&tab[slot], BSGS2_EMPTY, entry);
            if (old == BSGS2_EMPTY) break;
            if ((old >> 8) == fp) { dup = true; break; }   // a repeated value (ord_p(2) < m) or a fingerprint clash
            slot = (slot + 1) & (BSGS_H - 1);
        }
        x = monty_mul(x, step, p, p_inv);           // * 2^(-32)
    }
    const bool fallback = __any_sync(0xFFFFFFFFu, dup);
    __syncwarp();

    if (fallback) {                                 // the plain walk, the range split between the lanes
        const uint32_t ws = (range + 31) / 32;
        const uint32_t a = lane * ws;
        if (a >= range) return;
        const uint32_t e = (a + ws < range) ? a + ws : range;
        uint64_t c = monty_mul(bsgs2_mpow2(n_start + a, one, p, p_inv), k_mod, p, p_inv);   // k * 2^(n_start + a)
        for (uint32_t i = a; i < e; ++i) {
            if (c == p - 1) is_candidate[i] = 0;
            c <<= 1;
            if (c >= p) c -= p;
        }
        return;
    }

    // giant steps: lane L takes g = g0 .. g1-1
    const uint32_t giants = (range + BSGS_M - 1) / BSGS_M;
    const uint32_t gs = (giants + 31) / 32;
    const uint32_t g0 = lane * gs;
    if (g0 >= giants) return;
    const uint32_t g1 = (g0 + gs < giants) ? g0 + gs : giants;
    const uint64_t m_mont = bsgs2_mpow2(BSGS_M, one, p, p_inv);                                   // 2^m, Montgomery form
    uint64_t G = monty_mul(bsgs2_mpow2(n_start + g0 * BSGS_M, one, p, p_inv), k_mod, p, p_inv);  // k * 2^(n_start + g0*m)
    for (uint32_t g = g0; g < g1; ++g) {
        uint32_t slot, fp;
        bsgs2_key(G, slot, fp);
        uint32_t e;
        while ((e = tab[slot]) != BSGS2_EMPTY) {
            if ((e >> 8) == fp) {                   // a candidate: check G * 2^j == -1 exactly
                const uint32_t j = e & 0xFFu;
                uint64_t c = G;
                for (uint32_t t = 0; t < j; ++t) { c <<= 1; if (c >= p) c -= p; }
                if (c == p - 1) {
                    const uint32_t off = g * BSGS_M + j;
                    if (off < range) is_candidate[off] = 0;
                    break;
                }
            }
            slot = (slot + 1) & (BSGS_H - 1);
        }
        G = monty_mul(G, m_mont, p, p_inv);         // G * 2^m
    }
}

// -------------------------------------------------------------------------
// v3: v2 with m = 2^MB chosen from the range length (the cost per lane ~ m/32 baby + R/m/32 giant steps plus a fixed
// setup; a baby step costs ~4 giant ones, so m ~ sqrt(R)/2: 256 for R = 200000, 512 for R = 1e6 - long ranges are
// much cheaper per n).
//  - a table entry: (fingerprint << MB) | j, fingerprint 32-MB bits; table H = 2m slots, warps per block so that a block
//    has <= 32 KB of shared memory;
//  - a fingerprint repeat while inserting is checked exactly (2^|j-j'| == 1 <=> a real repeat, ord_p(2) < m -> walk);
//    a clash only probes on, so short fingerprints never send a prime to the slow walk;
//  - baby start 2^(-L) = monty(2^32, 2^(32-L)) for p > 2^32 (one multiplication instead of up to 31 halvings).
// LOTR_NSIEVE_MB=8..11 forces m (tests); LOTR_NSIEVE_BSGS: 0 walk, 1 v1, 2 v2, otherwise v3.
// -------------------------------------------------------------------------
template <int MB> struct Bsgs3 {
    static constexpr int M = 1 << MB;
    static constexpr int HB = MB + 1;
    static constexpr int H = 1 << HB;
    static constexpr int FPB = 32 - MB;                                   // fingerprint bits
    static constexpr uint32_t FPMASK = (1u << FPB) - 1u;
    static constexpr int W = (H * 4 * 8 <= 32768) ? 8 : (32768 / (H * 4));  // warps per block
    static constexpr int MINB = 1024 / (W * 32);                          // -> <= 64 registers
};

template <int MB>
__global__ void __launch_bounds__(Bsgs3<MB>::W * 32, Bsgs3<MB>::MINB) check_n_range_bsgs3_kernel(
    uint8_t* is_candidate,
    const uint64_t* primes_batch,
    uint64_t k_val,
    uint32_t n_start,
    uint32_t range,
    uint64_t count)
{
    typedef Bsgs3<MB> C;
    __shared__ uint32_t s_tab[C::W][C::H];

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t w = threadIdx.x >> 5;
    const uint64_t prime_idx = (uint64_t)blockIdx.x * C::W + w;
    if (prime_idx >= count) return;                 // the whole warp: one prime per warp

    const uint64_t p = primes_batch[prime_idx];
    if (p < 3) return;
    const uint64_t k_mod = bsgs2_mod(k_val, p);
    if (k_mod == 0) return;                         // p | k: p never divides k*2^n+1
    const uint64_t p_inv = calc_p_inv(p);
    const uint64_t one = bsgs2_r64(p);              // 1 in Montgomery form
    const bool big = p > (1ULL << 32);

    uint32_t* tab = s_tab[w];
    for (uint32_t s = lane; s < (uint32_t)C::H; s += 32) tab[s] = BSGS2_EMPTY;
    __syncwarp();

    // baby steps: lane L stores j = L + 32t (t = 0 .. m/32-1), the value -2^(-j) mod p
    uint64_t x;
    if (big) {
        x = monty_mul(1ULL << 32, 1ULL << (32 - lane), p, p_inv);       // 2^(32 + 32 - L - 64) = 2^(-L)
    } else {
        x = 1;
        for (uint32_t t = 0; t < lane; ++t) x = (x & 1ULL) ? ((x + p) >> 1) : (x >> 1);
    }
    const uint64_t step = big ? (1ULL << 32) : ((1ULL << 32) % p);       // 2^(-32) in Montgomery form = 2^32 mod p
    bool dup = false;
    for (uint32_t t = 0; t < (uint32_t)C::M / 32; ++t) {
        const uint32_t j = lane + 32 * t;
        const uint64_t h = (p - x) * 0x9E3779B97F4A7C15ULL;
        uint32_t slot = (uint32_t)(h >> (64 - C::HB));
        uint32_t fp = (uint32_t)(h >> (64 - C::HB - C::FPB)) & C::FPMASK;
        if (fp == C::FPMASK) fp = C::FPMASK - 1;    // no entry equals BSGS2_EMPTY
        const uint32_t entry = (fp << MB) | j;
        while (true) {
            const uint32_t old = atomicCAS(&tab[slot], BSGS2_EMPTY, entry);
            if (old == BSGS2_EMPTY) break;
            if ((old >> MB) == fp) {                // the same fingerprint: the same value? <=> 2^|j - j'| == 1
                const uint32_t jo = old & (uint32_t)(C::M - 1);
                const uint32_t d = (j > jo) ? j - jo : jo - j;
                uint64_t y = 1;
                for (uint32_t i = 0; i < d; ++i) { y <<= 1; if (y >= p) y -= p; }
                if (y == 1) { dup = true; break; }  // ord_p(2) divides d < m
            }
            slot = (slot + 1) & (uint32_t)(C::H - 1);
        }
        x = monty_mul(x, step, p, p_inv);           // * 2^(-32)
    }
    const bool fallback = __any_sync(0xFFFFFFFFu, dup);
    __syncwarp();

    if (fallback) {                                 // ord_p(2) < m: the plain walk, the range split between the lanes
        const uint32_t ws = (range + 31) / 32;
        const uint32_t a = lane * ws;
        if (a >= range) return;
        const uint32_t e = (a + ws < range) ? a + ws : range;
        uint64_t c = monty_mul(bsgs2_mpow2(n_start + a, one, p, p_inv), k_mod, p, p_inv);   // k * 2^(n_start + a)
        for (uint32_t i = a; i < e; ++i) {
            if (c == p - 1) is_candidate[i] = 0;
            c <<= 1;
            if (c >= p) c -= p;
        }
        return;
    }

    // giant steps: lane L takes g = g0 .. g1-1
    const uint32_t giants = (range + C::M - 1) / C::M;
    const uint32_t gs = (giants + 31) / 32;
    const uint32_t g0 = lane * gs;
    if (g0 >= giants) return;
    const uint32_t g1 = (g0 + gs < giants) ? g0 + gs : giants;
    const uint64_t m_mont = bsgs2_mpow2(C::M, one, p, p_inv);                                   // 2^m, Montgomery form
    uint64_t G = monty_mul(bsgs2_mpow2(n_start + g0 * C::M, one, p, p_inv), k_mod, p, p_inv);  // k * 2^(n_start + g0*m)
    for (uint32_t g = g0; g < g1; ++g) {
        const uint64_t h = G * 0x9E3779B97F4A7C15ULL;
        uint32_t slot = (uint32_t)(h >> (64 - C::HB));
        uint32_t fp = (uint32_t)(h >> (64 - C::HB - C::FPB)) & C::FPMASK;
        if (fp == C::FPMASK) fp = C::FPMASK - 1;
        uint32_t e;
        while ((e = tab[slot]) != BSGS2_EMPTY) {
            if ((e >> MB) == fp) {                  // a candidate: check G * 2^j == -1 exactly
                const uint32_t j = e & (uint32_t)(C::M - 1);
                uint64_t c = G;
                for (uint32_t t = 0; t < j; ++t) { c <<= 1; if (c >= p) c -= p; }
                if (c == p - 1) {
                    const uint32_t off = g * C::M + j;
                    if (off < range) is_candidate[off] = 0;
                    break;
                }
            }
            slot = (slot + 1) & (uint32_t)(C::H - 1);
        }
        G = monty_mul(G, m_mont, p, p_inv);         // G * 2^m
    }
}

template <int MB>
static void launch_bsgs3(uint8_t* cand, const uint64_t* primes, uint64_t k, uint32_t n_start, uint32_t range, uint64_t count) {
    const uint64_t blocks = (count + Bsgs3<MB>::W - 1) / Bsgs3<MB>::W;
    check_n_range_bsgs3_kernel<MB><<<(unsigned)blocks, Bsgs3<MB>::W * 32>>>(cand, primes, k, n_start, range, count);
}

// m = 2^MB for a range: the smallest 4*m + ceil(R/m) over MB = 8..10 (LOTR_NSIEVE_MB=8..11 forces it). A baby step
// (atomicCAS into the table) costs ~4 giant steps (measured: R = 200k best at m=256, R = 1M at m=512) -> m ~ sqrt(R)/2
static int bsgs3_pick_mb(uint32_t range) {
    const char* e = std::getenv("LOTR_NSIEVE_MB");
    if (e && *e) { int v = std::atoi(e); if (v >= 8 && v <= 11) return v; }
    int best = 8; uint64_t best_cost = ~0ULL;
    for (int mb = 8; mb <= 10; ++mb) {
        const uint64_t m = 1ULL << mb, cost = 4 * m + (range + m - 1) / m;
        if (cost < best_cost) { best_cost = cost; best = mb; }
    }
    return best;
}

extern "C" void init_gpu_sieve(size_t range_size, uint64_t max_primes_per_batch) {
    current_range_size = range_size;
    
    // Guard against double allocation
    if (d_is_candidate) cudaFree(d_is_candidate);
    if (d_internal_prime_batch) cudaFree(d_internal_prime_batch);
    if (d_internal_prime_count) cudaFree(d_internal_prime_count);

    cudaError_t e1 = cudaMalloc(&d_is_candidate, range_size * sizeof(uint8_t));
    cudaError_t e2 = cudaMalloc(&d_internal_prime_batch, max_primes_per_batch * sizeof(uint64_t));
    cudaError_t e3 = cudaMalloc(&d_internal_prime_count, sizeof(int));
    
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        printf("\n[FATAL] init_gpu_sieve failed to allocate VRAM! Ensure no other process (like Jupyter/PyTorch) is holding the GPU memory.\n");
        exit(1);
    }

    cudaMemset(d_is_candidate, 1, range_size * sizeof(uint8_t)); 
    cudaMemset(d_internal_prime_count, 0, sizeof(int));
}
extern "C" void process_prime_batch_gpu_from_vram(uint32_t n, uint64_t k_start, uint64_t k_end) {
    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (current_count > 0) {
        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;
        
        SieveTimer _t_chk(g_t_check);
        check_proth_candidates_kernel_fast<<<blocks, threads>>>(d_is_candidate, d_internal_prime_batch, n, k_start, k_end, current_count);
        
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n[FATAL KERNEL ERROR] Card killed during Proth FFT! Reason: %s\n", cudaGetErrorString(err));
        }
        
        cudaMemset(d_internal_prime_count, 0, sizeof(int));
    }
}

extern "C" void flush_prime_batch_gpu_from_vram(uint32_t n, uint64_t k_start, uint64_t k_end) {
    process_prime_batch_gpu_from_vram(n, k_start, k_end);
}




extern "C" void init_sparse_gpu_sieve(const uint64_t* h_ks, uint32_t num_ks, uint64_t max_primes_per_batch) {
    current_num_ks = num_ks;

    // Sort k: the fast sieve checks membership with a binary search. The permutation is undone in finalize,
    // so for the caller the input and output order stays the same.
    g_sparse_perm.resize(num_ks);
    for (uint32_t i = 0; i < num_ks; ++i) g_sparse_perm[i] = i;
    std::sort(g_sparse_perm.begin(), g_sparse_perm.end(), [&](uint32_t a, uint32_t b) { return h_ks[a] < h_ks[b]; });
    std::vector<uint64_t> sorted_ks(num_ks);
    for (uint32_t i = 0; i < num_ks; ++i) sorted_ks[i] = h_ks[g_sparse_perm[i]];
    g_sparse_kmin = num_ks ? sorted_ks.front() : 0;
    g_sparse_kmax = num_ks ? sorted_ks.back() : 0;

    cudaMalloc(&d_sparse_is_active, num_ks * sizeof(uint8_t));
    cudaMalloc(&d_sparse_ks, num_ks * sizeof(uint64_t));

    cudaMemset(d_sparse_is_active, 1, num_ks * sizeof(uint8_t));
    cudaMemcpy(d_sparse_ks, sorted_ks.data(), num_ks * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_internal_prime_batch, max_primes_per_batch * sizeof(uint64_t));
    cudaMalloc(&d_internal_prime_count, sizeof(int));
    cudaMemset(d_internal_prime_count, 0, sizeof(int));
    g_t_gen = g_t_extract = g_t_check = 0.0;
}

extern "C" void process_sparse_prime_batch_gpu_from_vram(uint32_t n) {
    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (current_count > 0) {
        SieveTimer _t(g_t_check);
        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;

        check_sparse_candidates_kernel_fast<<<blocks, threads>>>(
            d_sparse_is_active, d_sparse_ks, current_num_ks, g_sparse_kmin, g_sparse_kmax,
            d_internal_prime_batch, current_count, n
        );
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n[FATAL KERNEL ERROR] Sparse sieve kernel error: %s\n", cudaGetErrorString(err));
        }
        cudaMemset(d_internal_prime_count, 0, sizeof(int));
    }
}

extern "C" void flush_sparse_prime_batch_gpu_from_vram(uint32_t n) {
    process_sparse_prime_batch_gpu_from_vram(n);
}

extern "C" void finalize_sparse_gpu_sieve(uint8_t* h_is_active) {
    std::vector<uint8_t> tmp(current_num_ks);
    cudaMemcpy(tmp.data(), d_sparse_is_active, current_num_ks * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < current_num_ks; ++i) h_is_active[g_sparse_perm[i]] = tmp[i];
    sieve_prof_print("SPARSE");

    cudaFree(d_sparse_is_active);
    cudaFree(d_sparse_ks);

    if (d_internal_prime_batch != nullptr) { cudaFree(d_internal_prime_batch); d_internal_prime_batch = nullptr; }
    if (d_internal_prime_count != nullptr) { cudaFree(d_internal_prime_count); d_internal_prime_count = nullptr; }
    if (d_vram_bit_mask != nullptr) { cudaFree(d_vram_bit_mask); d_vram_bit_mask = nullptr; }
}

extern "C" void finalize_gpu_sieve(uint8_t* h_is_candidate) {
    cudaMemcpy(h_is_candidate, d_is_candidate, current_range_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    sieve_prof_print("FULL/N-RANGE");

    if (d_is_candidate != nullptr) { cudaFree(d_is_candidate); d_is_candidate = nullptr; }
    if (d_internal_prime_batch != nullptr) { cudaFree(d_internal_prime_batch); d_internal_prime_batch = nullptr; }
    if (d_internal_prime_count != nullptr) { cudaFree(d_internal_prime_count); d_internal_prime_count = nullptr; }
    if (d_vram_bit_mask != nullptr) { cudaFree(d_vram_bit_mask); d_vram_bit_mask = nullptr; }
    if (d_small_primes_vram != nullptr) { cudaFree(d_small_primes_vram); d_small_primes_vram = nullptr; }
    
    // CRITICAL: reset the state for further tests in the same process
    current_allocated_window_size = 0; 
}

// =========================================================================
// SIEVE FOR VARIABLE N (FIXED K)
// =========================================================================
// Wrapper called from C++
extern "C" void process_prime_batch_gpu_for_n_range(uint64_t k_val, uint32_t n_start, uint32_t n_end) {
    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (current_count > 0) {
        SieveTimer _t_chk(g_t_check);
        const uint32_t range = n_end - n_start + 1;
        // LOTR_NSIEVE_BSGS: 0 = the plain walk, 1 = baby-step giant-step v1, 2 = v2, unset/other = v3
        static const int bsgs_mode = [] {
            const char* e = std::getenv("LOTR_NSIEVE_BSGS");
            return (e && *e == '0') ? 0 : (e && *e == '1') ? 1 : (e && *e == '2') ? 2 : 3;
        }();
        if (bsgs_mode != 0 && range >= BSGS_MIN_RANGE) {
            const int mb = (bsgs_mode == 3) ? bsgs3_pick_mb(range) : 8;
            static bool told = false;                // --detail only
            if (g_detail && !told) { std::printf("\n[SIEVE] N-range check: baby-step giant-step v%d, table %d per prime\n", bsgs_mode, 1 << mb); told = true; }
            if (bsgs_mode == 1) {
                const uint64_t blocks = ((uint64_t)current_count + BSGS_WARPS - 1) / BSGS_WARPS;
                check_n_range_bsgs_kernel<<<(unsigned)blocks, BSGS_WARPS * 32>>>(d_is_candidate, d_internal_prime_batch, k_val, n_start, range, current_count);
            } else if (bsgs_mode == 2) {
                const uint64_t blocks = ((uint64_t)current_count + BSGS2_WARPS - 1) / BSGS2_WARPS;
                check_n_range_bsgs2_kernel<<<(unsigned)blocks, BSGS2_WARPS * 32>>>(d_is_candidate, d_internal_prime_batch, k_val, n_start, range, current_count);
            } else {
                const uint64_t cnt = (uint64_t)current_count;
                switch (mb) {
                case 8:  launch_bsgs3<8>(d_is_candidate, d_internal_prime_batch, k_val, n_start, range, cnt); break;
                case 9:  launch_bsgs3<9>(d_is_candidate, d_internal_prime_batch, k_val, n_start, range, cnt); break;
                case 10: launch_bsgs3<10>(d_is_candidate, d_internal_prime_batch, k_val, n_start, range, cnt); break;
                default: launch_bsgs3<11>(d_is_candidate, d_internal_prime_batch, k_val, n_start, range, cnt); break;
                }
            }
        } else {
            int threads = 256;
            int blocks = (current_count + threads - 1) / threads;
            check_proth_candidates_n_range_kernel_fast<<<blocks, threads>>>(d_is_candidate, d_internal_prime_batch, k_val, n_start, n_end, current_count);
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n[FATAL KERNEL ERROR] N-Range Kernel Error: %s\n", cudaGetErrorString(err));
        }
        
        cudaMemset(d_internal_prime_count, 0, sizeof(int));
    }
}

extern "C" void flush_prime_batch_gpu_for_n_range(uint64_t k_val, uint32_t n_start, uint32_t n_end) {
    process_prime_batch_gpu_for_n_range(k_val, n_start, n_end);
}

extern "C" void set_gpu_sieve_mask(const uint8_t* h_mask, size_t size) {
    if (d_is_candidate) {
        cudaMemcpy(d_is_candidate, h_mask, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
}

