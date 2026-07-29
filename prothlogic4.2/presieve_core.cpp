// Copyright (C) Damian Koba.

#include "presieve_core.h"
#include "gpu_engine.h"
#include <cmath>
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>

// =========================================================================
// PEŁNE SITO (ZERO-COPY OFFLOAD)
// =========================================================================
std::vector<uint64_t> execute_full_streaming_sieve(
    uint64_t k_start,
    uint64_t k_end_inclusive,
    unsigned n_pow2,
    uint64_t pmax)
{
    if (k_end_inclusive < k_start) return {};
    size_t range_size = k_end_inclusive - k_start + 1;

    // --- 1. Mini-sito na CPU (Baza do wysłania na GPU) ---
    uint64_t sqrt_pmax = (uint64_t)std::sqrt(pmax);
    std::vector<bool> is_prime_small(sqrt_pmax + 1, true);
    is_prime_small[0] = is_prime_small[1] = false;
    for (uint64_t i = 2; i * i <= sqrt_pmax; ++i) {
        if (is_prime_small[i]) {
            for (uint64_t j = i * i; j <= sqrt_pmax; j += i) is_prime_small[j] = false;
        }
    }
    std::vector<uint64_t> small_primes;
    for (uint64_t i = 2; i <= sqrt_pmax; ++i) {
        if (is_prime_small[i]) small_primes.push_back(i);
    }

    // --- 2. Konfiguracja ruchomego okna Zero-Copy ---
    uint64_t WINDOW_SIZE = 512000000ULL;    
    uint64_t MAX_BATCH = 100000000;

    init_gpu_sieve(range_size, MAX_BATCH);
    load_small_primes_to_gpu(small_primes.data(), small_primes.size());
    std::cout << "[GPU] Starting the ZERO-COPY sieve (window " << (WINDOW_SIZE / 1000000)
        << "M) for Pmax=" << pmax << "..." << std::endl;
    

    for (uint64_t window_start = 0; window_start <= pmax; window_start += WINDOW_SIZE) {

        uint64_t current_window_size = WINDOW_SIZE;
        if (window_start + WINDOW_SIZE > pmax) {
            current_window_size = pmax - window_start + 1;
        }

        generate_sieve_window_on_gpu_vram(
            window_start,
            current_window_size,
            small_primes.size() 
        );

        int current_batch_size = extract_primes_on_gpu(window_start, current_window_size, pmax);       
        if (current_batch_size >= 60000000) {
            process_prime_batch_gpu_from_vram(n_pow2, k_start, k_end_inclusive);
            std::cout << "\r[STREAM] Processed to P = " << window_start + current_window_size << "..." << std::flush;
        }
    }

    // KROK D: Wymuszenie przetworzenia ewentualnych "resztek" uwięzionych w VRAM
    flush_prime_batch_gpu_from_vram(n_pow2, k_start, k_end_inclusive);

    std::cout << "\n[STREAM] Candidate screening has been completed!" << std::endl;

    // --- 4. Zbieranie wyników ---
    std::vector<uint8_t> h_is_candidate(range_size);
    finalize_gpu_sieve(h_is_candidate.data());

    std::vector<uint64_t> ks;
    ks.reserve(1000);
    for (size_t i = 0; i < range_size; ++i) {
        uint64_t k = k_start + i;
        if ((k & 1ull) != 0 && h_is_candidate[i] == 1) {
            ks.push_back(k);
        }
    }

    std::cout << "[DONE] Candidates after sive " << ks.size() << " of " << range_size << std::endl;
    return ks;
}

// =========================================================================
// RZADKIE SITO (ZERO-COPY OFFLOAD)
// =========================================================================
std::vector<uint64_t> execute_sparse_streaming_sieve(
    const std::vector<uint64_t>& input_ks,
    unsigned n_pow2,
    uint64_t pstart,
    uint64_t pmax)
{
    if (input_ks.empty()) return {};
    uint32_t num_ks = input_ks.size();

    // 1. Mini-sito na CPU do małych liczb pierwszych
    uint64_t sqrt_pmax = (uint64_t)std::sqrt(pmax);
    std::vector<bool> is_prime_small(sqrt_pmax + 1, true);
    is_prime_small[0] = is_prime_small[1] = false;
    for (uint64_t i = 2; i * i <= sqrt_pmax; ++i) {
        if (is_prime_small[i]) {
            for (uint64_t j = i * i; j <= sqrt_pmax; j += i) is_prime_small[j] = false;
        }
    }
    std::vector<uint64_t> small_primes;
    for (uint64_t i = 2; i <= sqrt_pmax; ++i) {
        if (is_prime_small[i]) small_primes.push_back(i);
    }

    load_small_primes_to_gpu(small_primes.data(), small_primes.size());
    
    std::cout << "\n[SPARSE GPU] Starting a sieve for " << num_ks << " candidates.\n";

    uint64_t MAX_BATCH = 75000000;
    init_sparse_gpu_sieve(input_ks.data(), num_ks, MAX_BATCH);

    // KRYTYCZNA ZMIANA: Brak deklaracji new uint32_t[max_words_per_window] na procesorze!
    uint64_t WINDOW_SIZE = 1000000000ULL;

    std::cout << "[SPARSE STREAM ZERO-COPY] Sieving, from P=" << pstart << " to P=" << pmax << "..." << std::endl;

    uint64_t current_pstart = (pstart % 2 == 0) ? pstart : pstart - 1;

    for (uint64_t win_start = current_pstart; win_start < pmax; win_start += WINDOW_SIZE) {
        uint64_t current_window = std::min(WINDOW_SIZE, pmax - win_start);

        // 1. GPU wykreśla liczby i zapisuje maskę u siebie
        generate_sieve_window_on_gpu_vram(
            win_start,
            current_window,
            small_primes.size() // Podajemy tylko wielkość bazy
        );

        // 2. GPU odczytuje maskę bitową i pakuje ocalałych do swojego bufora w VRAM
        int current_batch_size = extract_sparse_primes_on_gpu(win_start, current_window, pmax);

        // 3. Sprawdzamy czy bufor VRAM zbliża się do pełna (60% MAX_BATCH)
        if (current_batch_size >= (MAX_BATCH * 0.6)) {
            process_sparse_prime_batch_gpu_from_vram(n_pow2);
            std::cout << "\r[SPARSE STREAM] Processed to P = " << win_start + current_window << "..." << std::flush;
        }
    }

    // 4. Uderzamy resztkami, które nie przekroczyły progu 60%
    flush_sparse_prime_batch_gpu_from_vram(n_pow2);

    std::cout << "\n[SPARSE STREAM] The list check has been completed!" << std::endl;

    // Pobranie wynikow
    std::vector<uint8_t> h_is_active(num_ks);
    finalize_sparse_gpu_sieve(h_is_active.data());

    std::vector<uint64_t> surviving_ks;
    for (uint32_t i = 0; i < num_ks; ++i) {
        if (h_is_active[i] == 1) {
            surviving_ks.push_back(input_ks[i]);
        }
    }

    std::cout << "[SPARSE DONE] Candidates after sive: " << surviving_ks.size() << " z " << num_ks << std::endl;
    return surviving_ks;
}