// Copyright (C) Damian Koba.

#include "gpu_engine.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

// The fast check kernels (2^-n / 2^n powering without multiplications, O(1) k membership test) are the default.
// LOTR_SIEVE_OLD=1 enables the old kernels (to compare results); LOTR_SIEVE_PROFILE=1 prints sieve stage timings.
static const bool g_sieve_old = (std::getenv("LOTR_SIEVE_OLD") != nullptr);
static const bool g_sieve_prof = (std::getenv("LOTR_SIEVE_PROFILE") != nullptr);
static double g_t_gen = 0.0, g_t_extract = 0.0, g_t_check = 0.0;   // stage times [s]: window / prime extraction / checking
static std::vector<uint32_t> g_sparse_perm;                         // sort permutation of k in the sparse sieve
static uint64_t g_sparse_kmin = 0, g_sparse_kmax = 0;

// Window generation: small primes (< SIEVE_TILE_PRIME_LIMIT) sieve tiles in shared memory,
// large ones - one warp per prime (atomicAnd in global memory). LOTR_SIEVE_OLD=1 = old kernel (one block per prime).
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
    std::printf("\n[SIEVE-PROFILE %s] window generation=%.1f s  prime extraction=%.1f s  checking=%.1f s  (%s kernels)\n",
        what, g_t_gen, g_t_extract, g_t_check, g_sieve_old ? "OLD" : "fast");
    g_t_gen = g_t_extract = g_t_check = 0.0;
}
__constant__ uint32_t d_delta_n[8192]; 

// 2. CUDA HELPER FUNCTIONS
// Safe 128-bit modular multiplication on the GPU cores
__device__ __forceinline__ uint64_t mul_mod_gpu(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((unsigned __int128)a * b) % m);
}

// Fast modular exponentiation
__device__ __forceinline__ uint64_t mod_exp_gpu(uint64_t base, uint32_t exp, uint64_t mod) {
    uint64_t res = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul_mod_gpu(res, base, mod);
        base = mul_mod_gpu(base, base, mod);
        exp /= 2;
    }
    return res;
}

// 3. NEW DELTA-SIEVE KERNEL
__global__ void sparse_n_delta_sieve_kernel(uint64_t k_val, uint64_t p_start, uint32_t num_candidates, uint8_t* d_survivors) {
    // Compute this thread's prime candidate P
    uint64_t p = p_start + (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // Guard against out-of-range / invalid P
    if (p <= 2) return;

    // The first candidate is computed the classic way (from scratch)
    uint64_t current_power = mod_exp_gpu(2, d_delta_n[0], p);
    
    // Condition for k * 2^N + 1 == 0 mod P  --->  k * 2^N == P - 1 mod P
    if (mul_mod_gpu(k_val, current_power, p) == p - 1) {
        d_survivors[0] = 0; 
    }

    // For all remaining candidates (deltas) take a fast step
    for (int i = 1; i < num_candidates; i++) {
        uint32_t delta = d_delta_n[i];
        uint64_t step;
        
        // Bit shift for small deltas (the biggest time saving)
        if (delta < 63) {
            step = (1ULL << delta) % p; 
        } else {
            step = mod_exp_gpu(2, delta, p);
        }
        
        // Multiply the power by the step
        current_power = mul_mod_gpu(current_power, step, p);
        
        // Divisibility test for the candidate
        if (mul_mod_gpu(k_val, current_power, p) == p - 1) {
            d_survivors[i] = 0; 
        }
    }
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
//static uint8_t* d_vram_byte_mask = nullptr;
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
// MASS GRID-STRIDE
// =========================================================================
__global__ void mark_window_kernel_batched(uint8_t* window_byte_mask, const uint64_t* d_primes, uint64_t num_primes, uint64_t window_start, uint64_t window_size) {
   // Each block on the graphics card gets its own unique prime number
    uint64_t prime_idx = blockIdx.x;
    if (prime_idx >= num_primes) return;

    uint64_t p = d_primes[prime_idx];
    if (p < 3) return;

    uint64_t window_end = window_start + window_size;
    if (p * p >= window_end) return; 

    uint64_t first_mult = ((window_start + p - 1) / p) * p;
    if (first_mult < p * p) first_mult = p * p;
    if (first_mult % 2 == 0) first_mult += p; 

    
    uint64_t current = first_mult + (threadIdx.x * 2 * p);
    uint64_t step = blockDim.x * 2 * p;

    while (current < window_end) {
        uint64_t local_idx = (current - window_start) / 2;
        window_byte_mask[local_idx] = 0; 
        current += step;
    }
}

__global__ void compress_to_bits_kernel(const uint8_t* byte_mask, uint32_t* bit_mask, uint64_t num_odd_elements) {
    uint64_t word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t max_words = (num_odd_elements + 31) / 32;

    if (word_idx >= max_words) return;

    uint32_t word = 0;
    for (int i = 0; i < 32; i++) {
        uint64_t local_bit_idx = word_idx * 32 + i;
        if (local_bit_idx < num_odd_elements) {
            if (byte_mask[local_bit_idx] == 1) {
                word |= (1U << i);
            }
        }
    }
    bit_mask[word_idx] = word;
}



// =========================================================================
// Block-Stride Safe Sieve
// =========================================================================
__global__ void mark_window_kernel_optimized(
    uint32_t* d_bitmask,
    uint64_t window_start,
    uint64_t window_size,
    const uint64_t* small_primes,
    uint32_t num_small_primes) 
{
    uint64_t p_idx = blockIdx.x;
    if (p_idx >= num_small_primes) return;

    uint64_t p = small_primes[p_idx];
    if (p < 3) return;

    uint64_t window_end = window_start + window_size;
    if (p * p >= window_end) return; 

    
    uint64_t first_mult = ((window_start + p - 1) / p) * p;
    if (first_mult < p * p) first_mult = p * p;
    if (first_mult % 2 == 0) first_mult += p; 

    if (first_mult >= window_end) return;

    uint64_t first_bit = (first_mult - window_start) / 2;
    uint64_t elements = window_size / 2;
    uint64_t step = p; 

    
    for (uint64_t i = first_bit + threadIdx.x * step; i < elements; i += blockDim.x * step) {
        atomicAnd(&d_bitmask[i >> 5], ~(1U << (i & 31)));
    }
}

// =========================================================================
// LAUNCHER C++
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

    if (g_sieve_old) {
        // Old generator: one block per small prime, atomicAnd in global memory.
        cudaMemset(d_vram_bit_mask, 0xFF, required_bytes);
        int threads = 256;
        int blocks = num_small_primes;
        mark_window_kernel_optimized<<<blocks, threads>>>(
            d_vram_bit_mask, window_start, window_size, d_small_primes_vram, num_small_primes
        );
    }
    else {
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

__device__ uint64_t gpu_pow_mod(uint64_t base, uint64_t exp, uint64_t p) {
    if (base == 0) return 0;
    
    uint64_t p_inv = calc_p_inv(p);
    
    
    uint64_t R = (0ULL - p) % p; 
    uint64_t R2 = (uint64_t)((((unsigned __int128)R) * R) % p);
    
    
    uint64_t base_m = monty_mul(base, R2, p, p_inv);
    uint64_t res_m = monty_mul(1ULL, R2, p, p_inv);
    
    while (exp > 0) {
        if (exp % 2 == 1) res_m = monty_mul(res_m, base_m, p, p_inv);
        base_m = monty_mul(base_m, base_m, p, p_inv);
        exp /= 2;
    }
    
    
    return monty_mul(res_m, 1ULL, p, p_inv);
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

__global__ void check_proth_candidates_kernel(
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

    uint64_t inv2 = (p + 1) / 2;
    uint64_t inv_2n = gpu_pow_mod(inv2, n, p);
    if (inv_2n == 0) return; 
    
    uint64_t target_k_mod = (p - inv_2n) % p;
    uint64_t rem = k_start % p;
    
    uint64_t step_to_first = (target_k_mod >= rem) ? (target_k_mod - rem) : (p - rem + target_k_mod);
    uint64_t first_k = k_start + step_to_first;

    if (first_k > k_end) return; 

    for (uint64_t k = first_k; k <= k_end; k += p) {
        uint64_t k_idx = k - k_start;
        is_candidate[k_idx] = 0; 
    }
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
        if (g_sieve_old) {
            check_proth_candidates_kernel<<<blocks, threads>>>(d_is_candidate, d_internal_prime_batch, n, k_start, k_end, current_count);
        }
        else {
            check_proth_candidates_kernel_fast<<<blocks, threads>>>(d_is_candidate, d_internal_prime_batch, n, k_start, k_end, current_count);
        }
        
        
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




// =========================================================================
//  SIEVE - SHARED MEMORY + COALESCED)
// =========================================================================
__global__ void check_sparse_candidates_kernel_shared(
    uint8_t* __restrict__ is_active_k,       
    const uint64_t* __restrict__ ks_array,   
    uint32_t num_ks,            
    const uint64_t* __restrict__ primes,     
    uint64_t count,             
    uint32_t n)                 
{
    
    extern __shared__ uint64_t shared_ks[];

    
    for (uint32_t i = threadIdx.x; i < num_ks; i += blockDim.x) {
        shared_ks[i] = ks_array[i];
    }
    __syncthreads(); 

    
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t p = primes[idx];
    if (p < 3) return;

    uint64_t inv2 = (p + 1) / 2;
    uint64_t inv_2n = gpu_pow_mod(inv2, n, p);
    if (inv_2n == 0) return;
    
    uint64_t target_k_mod = (p - inv_2n) % p;

    
    for (uint32_t i = 0; i < num_ks; ++i) {
        if (is_active_k[i]) { 
            
            if ((shared_ks[i] % p) == target_k_mod) {
                is_active_k[i] = 0; 
            }
        }
    }
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

        if (g_sieve_old) {
            size_t shared_mem_size = current_num_ks * sizeof(uint64_t);
            check_sparse_candidates_kernel_shared<<<blocks, threads, shared_mem_size>>>(
                d_sparse_is_active, d_sparse_ks, current_num_ks, d_internal_prime_batch, current_count, n
            );
        }
        else {
            check_sparse_candidates_kernel_fast<<<blocks, threads>>>(
                d_sparse_is_active, d_sparse_ks, current_num_ks, g_sparse_kmin, g_sparse_kmax,
                d_internal_prime_batch, current_count, n
            );
        }
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
__global__ void check_proth_candidates_n_range_kernel(
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

    // We need k^-1 mod p. If p divides k, k_mod == 0 (then k*2^N+1 == 1 mod p, so p never divides it)
    uint64_t k_mod = k_val % p;
    if (k_mod == 0) return; 
    
    // From k * 2^N + 1 = 0 (mod p) we get: 2^N = -k^-1 (mod p)
    uint64_t k_inv = gpu_pow_mod(k_mod, p - 2, p); // by Fermat's little theorem
    uint64_t target_mod = p - k_inv; // this is -k^-1 mod p

    // Compute 2^{n_start} mod p
    uint64_t current_2n = gpu_pow_mod(2ULL, n_start, p);

    // Iterate over exponents from n_start to n_end
    for (uint32_t n_val = n_start; n_val <= n_end; ++n_val) {
        if (current_2n == target_mod) {
            is_candidate[n_val - n_start] = 0; // reject this N
        }
        // Going to 2^{N+1} mod p is just (current_2n * 2) % p
        current_2n <<= 1;
        if (current_2n >= p) {
            current_2n -= p;
        }
    }
}

// Wrapper called from C++
extern "C" void process_prime_batch_gpu_for_n_range(uint64_t k_val, uint32_t n_start, uint32_t n_end) {
    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (current_count > 0) {
        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;
        
        SieveTimer _t_chk(g_t_check);
        if (g_sieve_old) {
            check_proth_candidates_n_range_kernel<<<blocks, threads>>>(d_is_candidate, d_internal_prime_batch, k_val, n_start, n_end, current_count);
        }
        else {
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


// =========================================================================
// SPARSE N-LIST SIEVE (ABSOLUTE N WITH GLOBAL SKIP)
// =========================================================================

static uint8_t* d_sparse_n_active = nullptr;
static uint32_t* d_sparse_n_array = nullptr;
static uint32_t current_sparse_num_ns = 0;

__global__ void check_sparse_n_candidates_kernel(
    uint8_t* __restrict__ is_active_n, 
    const uint32_t* __restrict__ n_array,
    uint32_t num_ns,
    const uint64_t* __restrict__ primes_batch,
    uint64_t k_val,
    uint64_t count)
{
    extern __shared__ uint32_t shared_ns[];

    for (uint32_t i = threadIdx.x; i < num_ns; i += blockDim.x) {
        shared_ns[i] = n_array[i];
    }
    __syncthreads();

    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t p = primes_batch[idx];
    if (p < 3) return;

    // Fast reduction of k (instead of modulo for small k)
    uint64_t k_mod = (k_val >= p) ? k_val % p : k_val;
    if (k_mod == 0) return; 

    uint64_t p_inv = calc_p_inv(p);
    
    uint64_t R = (0ULL - p) % p; 
    uint64_t R2 = (uint64_t)((((unsigned __int128)R) * R) % p);
    
    uint64_t k_mod_m = monty_mul(k_mod, R2, p, p_inv);
    uint64_t res_m = monty_mul(1ULL, R2, p, p_inv); 
    uint64_t R_mod_p = res_m;
    
    uint64_t exp = p - 2;
    uint64_t b_m = k_mod_m;
    while(exp > 0) {
        if (exp & 1) res_m = monty_mul(res_m, b_m, p, p_inv);
        b_m = monty_mul(b_m, b_m, p, p_inv);
        exp >>= 1;
    }
    
    uint64_t k_inv = monty_mul(res_m, 1ULL, p, p_inv);
    uint64_t target_mod = p - k_inv;
    uint64_t target_m = monty_mul(target_mod, R2, p, p_inv);
    uint64_t two_m = monty_mul(2ULL, R2, p, p_inv);

    // WARP-DIVERGENCE OPTIMIZATION: record max_n for the block
    // so that powering proceeds in lockstep without deep loops in a single thread.
    for (uint32_t i = 0; i < num_ns; ++i) {
        if (is_active_n[i]) {
            uint32_t n_val = shared_ns[i];
            uint64_t curr_res_m = R_mod_p;
            uint64_t curr_b_m = two_m;
            
            while (n_val > 0) {
                if (n_val & 1) curr_res_m = monty_mul(curr_res_m, curr_b_m, p, p_inv);
                curr_b_m = monty_mul(curr_b_m, curr_b_m, p, p_inv);
                n_val >>= 1;
            }
            
            if (curr_res_m == target_m) {
                is_active_n[i] = 0; 
            }
        }
    }
}
// -------------------------------------------------------------
// LAUNCHERS (called from presieve_core.cpp)
// -------------------------------------------------------------
extern "C" void init_sparse_n_gpu_sieve(const uint32_t* h_ns, uint32_t num_ns, uint64_t max_primes_per_batch) {
    current_sparse_num_ns = num_ns;
    
    cudaMalloc(&d_sparse_n_active, num_ns * sizeof(uint8_t));
    cudaMemset(d_sparse_n_active, 1, num_ns * sizeof(uint8_t));
    
    cudaMalloc(&d_sparse_n_array, num_ns * sizeof(uint32_t));
    cudaMemcpy(d_sparse_n_array, h_ns, num_ns * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_internal_prime_batch, max_primes_per_batch * sizeof(uint64_t));
    cudaMalloc(&d_internal_prime_count, sizeof(int));
    cudaMemset(d_internal_prime_count, 0, sizeof(int));
}

extern "C" void process_sparse_n_prime_batch_gpu_from_vram(uint64_t k_val) {
    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (current_count > 0) {
        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;
        
        // Compute the shared memory size dynamically from the loaded list (4 bytes per N value)
        size_t shared_mem_size = current_sparse_num_ns * sizeof(uint32_t);

        check_sparse_n_candidates_kernel<<<blocks, threads, shared_mem_size>>>(
            d_sparse_n_active, d_sparse_n_array, current_sparse_num_ns, d_internal_prime_batch, k_val, current_count
        );
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n[FATAL KERNEL ERROR] Sparse N-Range Kernel Error: %s\n", cudaGetErrorString(err));
        }
        
        cudaMemset(d_internal_prime_count, 0, sizeof(int));
    }
}

extern "C" void flush_sparse_n_prime_batch_gpu_from_vram(uint64_t k_val) {
    process_sparse_n_prime_batch_gpu_from_vram(k_val);
}

extern "C" void finalize_sparse_n_gpu_sieve(uint8_t* h_is_active) {
    cudaMemcpy(h_is_active, d_sparse_n_active, current_sparse_num_ns * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    // Free the buffers
    cudaFree(d_sparse_n_active);
    cudaFree(d_sparse_n_array);
    
    if (d_internal_prime_batch != nullptr) { cudaFree(d_internal_prime_batch); d_internal_prime_batch = nullptr; }
    if (d_internal_prime_count != nullptr) { cudaFree(d_internal_prime_count); d_internal_prime_count = nullptr; }
    if (d_vram_bit_mask != nullptr) { cudaFree(d_vram_bit_mask); d_vram_bit_mask = nullptr; }
}