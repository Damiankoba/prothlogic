// Copyright (C) Damian Koba.

#pragma once
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

    
    void init_gpu_sieve(size_t range_size, uint64_t max_primes_per_batch);
    void finalize_gpu_sieve(uint8_t* h_is_candidate);
    void init_sparse_gpu_sieve(const uint64_t* h_ks, uint32_t num_ks, uint64_t max_primes_per_batch);
    void finalize_sparse_gpu_sieve(uint8_t* h_is_active);   

   void load_small_primes_to_gpu(const uint64_t* small_primes, uint64_t num_small_primes);
   void generate_sieve_window_on_gpu_vram(
        uint64_t window_start,
        uint64_t window_size,
        uint64_t num_small_primes
    );    
    int extract_primes_on_gpu(uint64_t window_start, uint64_t window_size, uint64_t pmax);
    int extract_sparse_primes_on_gpu(uint64_t window_start, uint64_t window_size, uint64_t pmax);    

    // FULL SIEVE (k range)
    void process_prime_batch_gpu_from_vram(uint32_t n, uint64_t k_start, uint64_t k_end);    
    void flush_prime_batch_gpu_from_vram(uint32_t n, uint64_t k_start, uint64_t k_end);

    // SPARSE SIEVE (list of k)
    void process_sparse_prime_batch_gpu_from_vram(uint32_t n);    
    void flush_sparse_prime_batch_gpu_from_vram(uint32_t n);

// SIEVE FOR A RANGE OF N (fixed k)
    void process_prime_batch_gpu_for_n_range(uint64_t k_val, uint32_t n_start, uint32_t n_end);    
    void flush_prime_batch_gpu_for_n_range(uint64_t k_val, uint32_t n_start, uint32_t n_end);
    void set_gpu_sieve_mask(const uint8_t* h_mask, size_t size);
#ifdef __cplusplus
}
#endif