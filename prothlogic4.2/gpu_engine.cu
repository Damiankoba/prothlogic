// Copyright (C) Damian Koba.

#include "gpu_engine.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <iostream>

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
}

// =========================================================================
// CZĘŚĆ 1: KERNEL GENEROWANIA MASKI SITA (MASOWY GRID-STRIDE)
// =========================================================================
__global__ void mark_window_kernel_batched(uint8_t* window_byte_mask, const uint64_t* d_primes, uint64_t num_primes, uint64_t window_start, uint64_t window_size) {
    // Każdy blok na karcie graficznej otrzymuje własną, unikalną liczbę pierwszą
    uint64_t prime_idx = blockIdx.x;
    if (prime_idx >= num_primes) return;

    uint64_t p = d_primes[prime_idx];
    if (p < 3) return;

    uint64_t window_end = window_start + window_size;
    if (p * p >= window_end) return; // Karta sama ucina zbędne obliczenia

    uint64_t first_mult = ((window_start + p - 1) / p) * p;
    if (first_mult < p * p) first_mult = p * p;
    if (first_mult % 2 == 0) first_mult += p; 

    // Wątki wewnątrz bloku skaczą po oknie wykreślając wielokrotności
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
// CZĘŚĆ 2: WYDAJNE SITO BLOKOWE (Block-Stride Safe Sieve)
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

    // Precyzyjne wyznaczenie pierwszej wielokrotności w oknie (zgodne z oryginalnym sitem)
    uint64_t first_mult = ((window_start + p - 1) / p) * p;
    if (first_mult < p * p) first_mult = p * p;
    if (first_mult % 2 == 0) first_mult += p; 

    if (first_mult >= window_end) return;

    uint64_t first_bit = (first_mult - window_start) / 2;
    uint64_t elements = window_size / 2;
    uint64_t step = p; // Krok o p dla operacji na bitach (co druga liczba to nieparzysta)

    // Grid-stride wewnątrz bloku (256 wątków wspólnie wykreśla wielokrotności)
    for (uint64_t i = first_bit + threadIdx.x * step; i < elements; i += blockDim.x * step) {
        atomicAnd(&d_bitmask[i >> 5], ~(1U << (i & 31)));
    }
}

// =========================================================================
// ZAKTUALIZOWANY LAUNCHER C++
// =========================================================================
extern "C" void generate_sieve_window_on_gpu_vram(
    uint64_t window_start,
    uint64_t window_size,
    uint64_t num_small_primes)
{
    uint64_t elements = window_size / 2;
    uint32_t num_words = (elements + 31) / 32;
    size_t required_bytes = num_words * sizeof(uint32_t);

    // Jeśli okno zmieniło rozmiar lub pamięć nie była jeszcze alokowana
    if (window_size != current_allocated_window_size) {
        if (d_vram_bit_mask != nullptr) {
            cudaFree(d_vram_bit_mask);
        }
        cudaError_t err = cudaMalloc(&d_vram_bit_mask, required_bytes);
        if (err != cudaSuccess) {
            std::cerr << "[FATAL] Nie udalo się zaalokować VRAM na maskę bitową!" << std::endl;
            return;
        }
        current_allocated_window_size = window_size;
    }

    // Reset maski (wypełnienie jedynkami)
    cudaMemset(d_vram_bit_mask, 0xFF, required_bytes);

    int threads = 256;
    int blocks = (num_small_primes + threads - 1) / threads;

    // Uruchomienie stabilnego kernela
    mark_window_kernel_optimized<<<blocks, threads>>>(
        d_vram_bit_mask, window_start, window_size, d_small_primes_vram, num_small_primes
    );
    
    cudaDeviceSynchronize();
}


// =========================================================================
// CZĘŚĆ 3: EKSTRAKCJA BEZPOŚREDNIO NA KARTĘ GRAFICZNĄ (WARP-AGGREGATED)
// =========================================================================
__global__ void extract_primes_kernel(
    const uint32_t* d_window_bitmask, 
    uint64_t window_start, 
    uint64_t pmax, 
    uint64_t odd_elements_in_window, 
    uint64_t* d_prime_batch, 
    int* d_prime_count) 
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32; // Identyfikator wątku wewnątrz 32-osobowej grupy (Warp)
    
    int is_prime = 0;
    uint64_t p = 0;
    
    // 1. Każdy wątek cicho sprawdza swój bit, nie dotykając biletomatu
    if (i < odd_elements_in_window) {
        if (d_window_bitmask[i >> 5] & (1U << (i & 31))) {
            p = window_start + 2 * i + 1;
            if (p <= pmax && p >= 3) {
                is_prime = 1;
            }
        }
    }
    
    // 2. MAGIA REJESTRÓW: Wszystkie 32 wątki głosują w 1 cyklu zegara.
    // Tworzy się 32-bitowa maska, gdzie "1" oznacza, że dany wątek ma ocalałą liczbę.
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, is_prime);
    
    // Jeśli ktokolwiek w warpie znalazł liczbę pierwszą
    if (active_mask != 0) {
        // Liderem zostaje pierwszy wątek (od najmłodszego bitu), który ma liczbę pierwszą
        int leader = __ffs(active_mask) - 1; 
        int warp_offset = 0;
        
        // 3. TYLKO LIDER uderza do powolnej pamięci VRAM.
        // Rezerwuje miejsce dla wszystkich wątków w grupie naraz (__popc zlicza jedynki w masce)
        if (lane_id == leader) {
            warp_offset = atomicAdd(d_prime_count, __popc(active_mask));
        }
        
        // 4. Lider krzyczy do reszty grupy (rozgłasza pobrany z VRAM indeks startowy)
        warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, leader);
        
        // 5. Każdy aktywny wątek oblicza swój prywatny, bezkonfliktowy indeks i zapisuje liczbę
        if (is_prime) {
            // Przesunięcie bitowe oblicza, ilu "kolegów" przed danym wątkiem miało liczbę
            int my_offset = warp_offset + __popc(active_mask & ((1 << lane_id) - 1));
            d_prime_batch[my_offset] = p;
        }
    }
}

extern "C" int extract_primes_on_gpu(uint64_t window_start, uint64_t window_size, uint64_t pmax) {
    uint64_t odd_elements = window_size / 2;
    int threads = 256;
    int blocks = (odd_elements + threads - 1) / threads;

    extract_primes_kernel<<<blocks, threads>>>(
        d_vram_bit_mask, window_start, pmax, odd_elements, d_internal_prime_batch, d_internal_prime_count
    );
    cudaDeviceSynchronize();

    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);
    return current_count;
}

extern "C" int extract_sparse_primes_on_gpu(uint64_t window_start, uint64_t window_size, uint64_t pmax) {
    return extract_primes_on_gpu(window_start, window_size, pmax);
}


// =========================================================================
// CZĘŚĆ 4: SZYBKA MATEMATYKA KARTY (REDUKCJA MONTGOMERY'EGO)
// =========================================================================

// Oblicza magiczną stałą -p^(-1) mod 2^64 dla algorytmu Montgomery'ego
__device__ __forceinline__ uint64_t calc_p_inv(uint64_t p) {
    uint64_t inv = p;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    inv *= 2ULL - p * inv;
    return ~inv + 1ULL; // Zwraca -inv (U2)
}

// Błyskawiczne mnożenie modulo używające natywnych instrukcji rdzeni CUDA
__device__ __forceinline__ uint64_t monty_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t p_inv) {
    uint64_t T_lo = a * b;
    uint64_t T_hi = __umul64hi(a, b); // Natywne PTX: 64x64 -> górne 64 bity
    
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
    
    // POPRAWKA: Bezpieczne, 64-bitowe obliczanie wartości R.
    // Dzięki underflow w typach bez znaku, 0ULL - p odpowiada 2^64 - p.
    uint64_t R = (0ULL - p) % p; 
    uint64_t R2 = (uint64_t)((((unsigned __int128)R) * R) % p);
    
    // Wejście w formę Montgomery'ego
    uint64_t base_m = monty_mul(base, R2, p, p_inv);
    uint64_t res_m = monty_mul(1ULL, R2, p, p_inv);
    
    while (exp > 0) {
        if (exp % 2 == 1) res_m = monty_mul(res_m, base_m, p, p_inv);
        base_m = monty_mul(base_m, base_m, p, p_inv);
        exp /= 2;
    }
    
    // Wyjście z formy Montgomery'ego
    return monty_mul(res_m, 1ULL, p, p_inv);
}


// =========================================================================
// CZĘŚĆ 5: UDERZENIE PROTHA BEZPOŚREDNIO Z VRAM (PEŁNE SITO)
// =========================================================================
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
    cudaMalloc(&d_is_candidate, range_size * sizeof(uint8_t));
    cudaMemset(d_is_candidate, 1, range_size * sizeof(uint8_t)); 
    
    // Inicjalizacja komponentów Zero-Copy
    cudaMalloc(&d_internal_prime_batch, max_primes_per_batch * sizeof(uint64_t));
    cudaMalloc(&d_internal_prime_count, sizeof(int));
    cudaMemset(d_internal_prime_count, 0, sizeof(int));
}

extern "C" void process_prime_batch_gpu_from_vram(uint32_t n, uint64_t k_start, uint64_t k_end) {
    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (current_count > 0) {
        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;
        
        check_proth_candidates_kernel<<<blocks, threads>>>(d_is_candidate, d_internal_prime_batch, n, k_start, k_end, current_count);
        
        // --- RADAR BŁĘDÓW KERNELA ---
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("\n[FATAL KERNEL ERROR] Karta zabita podczas Proth FFT! Przyczyna: %s\n", cudaGetErrorString(err));
        }
        
        cudaMemset(d_internal_prime_count, 0, sizeof(int));
    }
}

extern "C" void flush_prime_batch_gpu_from_vram(uint32_t n, uint64_t k_start, uint64_t k_end) {
    process_prime_batch_gpu_from_vram(n, k_start, k_end);
}




// =========================================================================
// CZĘŚĆ 6: UDERZENIE PROTHA Z VRAM (RZADKIE SITO - SHARED MEMORY + COALESCED)
// =========================================================================
__global__ void check_sparse_candidates_kernel_shared(
    uint8_t* __restrict__ is_active_k,       
    const uint64_t* __restrict__ ks_array,   
    uint32_t num_ks,            
    const uint64_t* __restrict__ primes,     
    uint64_t count,             
    uint32_t n)                 
{
    // Alokacja dynamicznej pamięci współdzielonej (najszybsza pamięć wewnątrz bloku SM na karcie)
    extern __shared__ uint64_t shared_ks[];

    // 1. Wątki w bloku współpracują, aby spójnie (coalesced) załadować kandydatów z VRAM do Shared Memory
    for (uint32_t i = threadIdx.x; i < num_ks; i += blockDim.x) {
        shared_ks[i] = ks_array[i];
    }
    __syncthreads(); // Wątki czekają, aż cała tablica bezpiecznie wyląduje w pamięci współdzielonej

    // 2. Klasyczny rozkład obciążenia: tysiące wątków = tysiące liczb pierwszych (pełna moc GPU)
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t p = primes[idx];
    if (p < 3) return;

    uint64_t inv2 = (p + 1) / 2;
    uint64_t inv_2n = gpu_pow_mod(inv2, n, p);
    if (inv_2n == 0) return;
    
    uint64_t target_k_mod = (p - inv_2n) % p;

    // 3. Iteracja po ultraszybkiej pamięci współdzielonej (brak obciążania szyny VRAM)
    for (uint32_t i = 0; i < num_ks; ++i) {
        // Zczytujemy z VRAM tylko 1-bajtowe flagi (ekstremalnie dobrze cachowane przez L1/L2)
        if (is_active_k[i]) { 
            if (shared_ks[i] == target_k_mod) {
                is_active_k[i] = 0; // Ustrzelony!
            }
        }
    }
}

extern "C" void init_sparse_gpu_sieve(const uint64_t* h_ks, uint32_t num_ks, uint64_t max_primes_per_batch) {
    current_num_ks = num_ks;
    
    cudaMalloc(&d_sparse_is_active, num_ks * sizeof(uint8_t));
    cudaMalloc(&d_sparse_ks, num_ks * sizeof(uint64_t));
    
    cudaMemset(d_sparse_is_active, 1, num_ks * sizeof(uint8_t));
    cudaMemcpy(d_sparse_ks, h_ks, num_ks * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Inicjalizacja komponentów Zero-Copy
    cudaMalloc(&d_internal_prime_batch, max_primes_per_batch * sizeof(uint64_t));
    cudaMalloc(&d_internal_prime_count, sizeof(int));
    cudaMemset(d_internal_prime_count, 0, sizeof(int));
}

extern "C" void process_sparse_prime_batch_gpu_from_vram(uint32_t n) {
    int current_count = 0;
    cudaMemcpy(&current_count, d_internal_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (current_count > 0) {
        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;
        
        // Obliczamy rozmiar pamięci współdzielonej (1660 kandydatów * 8 bajtów = ~13 KB)
        // Karty NVIDIA gwarantują min. 48 KB na blok, więc mieścimy się z ogromnym zapasem
        size_t shared_mem_size = current_num_ks * sizeof(uint64_t);

        check_sparse_candidates_kernel_shared<<<blocks, threads, shared_mem_size>>>(
            d_sparse_is_active, d_sparse_ks, current_num_ks, d_internal_prime_batch, current_count, n
        );
        cudaDeviceSynchronize();
        cudaMemset(d_internal_prime_count, 0, sizeof(int));
    }
}

extern "C" void flush_sparse_prime_batch_gpu_from_vram(uint32_t n) {
    process_sparse_prime_batch_gpu_from_vram(n);
}

extern "C" void finalize_sparse_gpu_sieve(uint8_t* h_is_active) {
    cudaMemcpy(h_is_active, d_sparse_is_active, current_num_ks * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_sparse_is_active);
    cudaFree(d_sparse_ks);
    
    if (d_internal_prime_batch != nullptr) { cudaFree(d_internal_prime_batch); d_internal_prime_batch = nullptr; }
    if (d_internal_prime_count != nullptr) { cudaFree(d_internal_prime_count); d_internal_prime_count = nullptr; }
    if (d_vram_bit_mask != nullptr) { cudaFree(d_vram_bit_mask); d_vram_bit_mask = nullptr; }
}

extern "C" void finalize_gpu_sieve(uint8_t* h_is_candidate) {
    cudaMemcpy(h_is_candidate, d_is_candidate, current_range_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_is_candidate);
    cudaFree(d_internal_prime_batch);
    cudaFree(d_internal_prime_count);
    
    if (d_vram_bit_mask != nullptr) { cudaFree(d_vram_bit_mask); d_vram_bit_mask = nullptr; }
    if (d_small_primes_vram != nullptr) { cudaFree(d_small_primes_vram); d_small_primes_vram = nullptr; }
}