// Copyright (C) Damian Koba.


#include <cmath>
#include <cstring>
#include "proth_logic.h"
#include "checkpoint_manager.h" 
#include "fft_engine.h"
#include "storage_core.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <fstream>
#include <chrono>

extern std::mutex g_io_mutex;
thread_local uint64_t dbg_hn1 = 0;
thread_local uint64_t dbg_hn2 = 0;
thread_local uint64_t dbg_hn3p = 0;
thread_local uint64_t dbg_ret_nohigh = 0;
thread_local uint64_t dbg_ret_false = 0;
thread_local uint64_t dbg_ret_true_end = 0;
thread_local uint64_t dbg_borrow_fix = 0;

// --- POMOCNICZE FUNKCJE DLA BRAMKI ---
static inline uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)(((__uint128_t)a * b) % m);
}

static uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m; a %= m;
    while (e) {
        if (e & 1) r = mulmod_u64(r, a, m);
        a = mulmod_u64(a, a, m);
        e >>= 1;
    }
    return r;
}

static bool is_divisible_by_gate(uint64_t k, unsigned n, uint64_t p) {
    if (p % 2 == 0) return false;
    uint64_t km = k % p;
    if (km == 0) return false;
    uint64_t invk = powmod_u64(km, p - 2, p);
    uint64_t target = (p - 1);
    target = mulmod_u64(target, invk, p);
    uint64_t two_n = powmod_u64(2, n, p);
    return (two_n == target);
}

static inline void proth_clear_above_n(uint64_t* limbs, size_t n_limbs, unsigned n_bits)
{
    size_t cut_limb = n_bits >> 6;
    unsigned cut_shift = n_bits & 63u;

    if (cut_limb >= n_limbs) return;

    if (cut_shift == 0) {
        for (size_t i = cut_limb; i < n_limbs; ++i) limbs[i] = 0;
    }
    else {
        uint64_t mask = (1ULL << cut_shift) - 1ULL;
        limbs[cut_limb] &= mask;
        for (size_t i = cut_limb + 1; i < n_limbs; ++i) limbs[i] = 0;
    }
}

static inline void add_u64_shifted(uint64_t* limbs, size_t n_limbs, uint64_t val, unsigned bitpos)
{
    if (val == 0) return;
    size_t limb = bitpos >> 6;
    unsigned shift = bitpos & 63u;
    if (limb >= n_limbs) return;

    __uint128_t acc = (__uint128_t)limbs[limb] + ((__uint128_t)val << shift);
    limbs[limb] = (uint64_t)acc;    
    uint64_t carry = (uint64_t)(acc >> 64);
    size_t i = limb + 1;
    
    while (carry && i < n_limbs) {
        __uint128_t acc3 = (__uint128_t)limbs[i] + carry;
        limbs[i] = (uint64_t)acc3;
        carry = (uint64_t)(acc3 >> 64);
        ++i;
    }
}

int cmp_proth_modulus(const uint64_t* limbs, size_t n_limbs, size_t n_bits, uint64_t k) {
    size_t k_limb = n_bits / 64;
    size_t k_shift = n_bits % 64;
    uint64_t mod_low = k << k_shift;
    uint64_t mod_high = (k_shift == 0) ? 0 : (k >> (64 - k_shift));   
    for (size_t i = n_limbs - 1; i > k_limb + 1; --i) {
        if (limbs[i] > 0) return 1;
    }    
    if (k_limb + 1 < n_limbs) {
        if (limbs[k_limb + 1] > mod_high) return 1;
        if (limbs[k_limb + 1] < mod_high) return -1;
    }    
    if (k_limb < n_limbs) {
        if (limbs[k_limb] > mod_low) return 1;
        if (limbs[k_limb] < mod_low) return -1;
    }    
    if (k_limb > 0) {
        for (size_t i = k_limb - 1; i >= 1; --i) {
            if (limbs[i] > 0) return 1;
        }
    }    
    if (limbs[0] > 1) return 1;
    if (limbs[0] < 1) return -1;        
    return 0;
}




void add_proth_modulus_once(uint64_t* limbs, size_t n_limbs, size_t n_bits, uint64_t k) {
    
    uint64_t carry = 1;
    for (size_t i = 0; i < n_limbs && carry; ++i) {
        uint64_t old = limbs[i];
        limbs[i] += carry;
        carry = (limbs[i] < old);
    }    
    size_t k_limb = n_bits / 64;
    size_t k_shift = n_bits % 64;    
    uint64_t low = k << k_shift;
    uint64_t high = (k_shift == 0) ? 0 : (k >> (64 - k_shift));    
    carry = 0;        
    if (k_limb < n_limbs) {
        uint64_t old = limbs[k_limb];
        limbs[k_limb] += low;
        carry = (limbs[k_limb] < old);
    }
    
    // 4. Dodajemy górną część K wraz z przeniesieniem
    if (k_limb + 1 < n_limbs) {
        uint64_t old = limbs[k_limb + 1];
        limbs[k_limb + 1] += high + carry;
        carry = (limbs[k_limb + 1] < old);
    }
    
    // 5. Roznosimy resztę carry aż do samego szczytu tablicy!
    for (size_t i = k_limb + 2; i < n_limbs && carry; ++i) {
        uint64_t old = limbs[i];
        limbs[i] += carry;
        carry = (limbs[i] < old);
    }
}

static inline void sub_proth_modulus_once(uint64_t* limbs, size_t n_limbs, unsigned n_bits, uint64_t k_val)
{
    uint64_t borrow = 0;
    size_t cut_limb = n_bits >> 6;
    unsigned cut_shift = n_bits & 63u;

    for (size_t idx = 0; idx < n_limbs; ++idx) {
        __uint128_t modv = 0;

        if (idx == 0) modv += 1;

        if (idx == cut_limb) {
            modv += ((__uint128_t)k_val << cut_shift);
        }
        if (cut_shift != 0 && idx == cut_limb + 1) {
            modv += (k_val >> (64 - cut_shift));
        }

        uint64_t sub = (uint64_t)modv;

        uint64_t oldv = limbs[idx];
        uint64_t tmp = oldv - sub;
        uint64_t b1 = (oldv < sub) ? 1 : 0;

        uint64_t tmp2 = tmp - borrow;
        uint64_t b2 = (tmp < borrow) ? 1 : 0;

        limbs[idx] = tmp2;
        borrow = (b1 | b2);
    }
}

static inline size_t extract_high_shifted(const uint64_t* limbs, size_t n_limbs,
    unsigned n_bits, uint64_t* high)
{
    size_t cut_limb = n_bits >> 6;
    unsigned cut_shift = n_bits & 63u;

    std::memset(high, 0, n_limbs * sizeof(uint64_t));

    if (cut_limb >= n_limbs) return 0;

    if (cut_shift == 0) {
        for (size_t i = cut_limb; i < n_limbs; ++i) {
            high[i - cut_limb] = limbs[i];
        }
    }
    else {
        for (size_t i = cut_limb; i < n_limbs; ++i) {
            uint64_t v = limbs[i] >> cut_shift;
            if (i + 1 < n_limbs) v |= (limbs[i + 1] << (64 - cut_shift));
            high[i - cut_limb] = v;
        }
    }

    size_t hn = n_limbs;
    while (hn > 0 && high[hn - 1] == 0) --hn;
    return hn;
}

static inline uint64_t div_big_by_u64(uint64_t* a, size_t n, uint64_t d)
{
    __uint128_t rem = 0;
    for (size_t i = n; i-- > 0;) {
        __uint128_t cur = (rem << 64) | a[i];
        a[i] = (uint64_t)(cur / d);
        rem = cur % d;
    }
    return (uint64_t)rem;
}

static inline bool sub_big_inplace(uint64_t* dst, size_t dst_n, const uint64_t* sub, size_t sub_n)
{
    uint64_t borrow = 0;
    size_t i = 0;

    for (; i < sub_n; ++i) {
        uint64_t s = sub[i];
        uint64_t oldv = dst[i];
        uint64_t tmp = oldv - s;
        uint64_t b1 = (oldv < s) ? 1 : 0;

        uint64_t tmp2 = tmp - borrow;
        uint64_t b2 = (tmp < borrow) ? 1 : 0;

        dst[i] = tmp2;
        borrow = (b1 | b2);
    }

    while (borrow && i < dst_n) {
        uint64_t oldv = dst[i];
        dst[i] -= 1;
        borrow = (oldv == 0) ? 1 : 0;
        ++i;
    }

    return (borrow != 0);
}
static bool proth_reduce_stage1_rec(uint64_t* limbs, size_t n_limbs, unsigned n_bits, uint64_t k_val, int depth)
{
    if (depth > 32) {
        
        dbg_ret_false++;
        return false;
    }

    std::vector<uint64_t> high(n_limbs, 0);

    // ZWIĘKSZONY LIMIT RUND DO 10
    for (int round = 0; round < 10; ++round) {
std::fill(high.begin(), high.end(), 0);

        size_t hn = extract_high_shifted(limbs, n_limbs, n_bits, high.data());

        bool fully_reduced = false;

        if (hn == 0) {
            fully_reduced = true;
        } else if (hn == 1) {
            if (high[0] < k_val) {
                fully_reduced = true;
            } else if (high[0] == k_val) {
                if (cmp_proth_modulus(limbs, n_limbs, n_bits, k_val) < 0) {
                    fully_reduced = true;
                }
            }
        }

        if (fully_reduced) {
            dbg_ret_nohigh++;
            for (int norm = 0; norm < 5; ++norm) {
                if (cmp_proth_modulus(limbs, n_limbs, n_bits, k_val) >= 0) {
                    sub_proth_modulus_once(limbs, n_limbs, n_bits, k_val);
                } else {
                    break;
                }
            }
            dbg_ret_true_end++;
            return true;
        }

        if (hn == 1) dbg_hn1++;
        else if (hn == 2) dbg_hn2++;
        else dbg_hn3p++;

        proth_clear_above_n(limbs, n_limbs, n_bits);

        uint64_t r = div_big_by_u64(high.data(), hn, k_val);

        if (!proth_reduce_stage1_rec(high.data(), n_limbs, n_bits, k_val, depth + 1)) {
            fprintf(stderr, "\n[DEBUG] RECURSION FAILED AT DEPTH %d!\n", depth);
            dbg_ret_false++;
            return false;
        }

        size_t qn = n_limbs;
        while (qn > 0 && high[qn - 1] == 0) --qn;

        add_u64_shifted(limbs, n_limbs, r, n_bits);

        bool borrow = sub_big_inplace(limbs, n_limbs, high.data(), qn);
        if (borrow) {
            dbg_borrow_fix++;
            add_proth_modulus_once(limbs, n_limbs, n_bits, k_val);

            if ((int64_t)limbs[n_limbs - 1] < 0) {
                add_proth_modulus_once(limbs, n_limbs, n_bits, k_val);
            }
                        
            proth_clear_above_n(limbs, n_limbs, n_bits + 64);
        }

    } 
        
    dbg_ret_false++;
    return false;
}


static bool proth_reduce_stage1(uint64_t* limbs, size_t n_limbs, unsigned n_bits, uint64_t k_val)
{
    return proth_reduce_stage1_rec(limbs, n_limbs, n_bits, k_val, 0);
}

static bool is_nm1_limbs(const uint64_t* limbs, size_t n_limbs, unsigned n_bits, uint64_t k_val)
{
    std::vector<uint64_t> target(n_limbs, 0);

    size_t limb = n_bits >> 6;
    unsigned shift = n_bits & 63u;

    if (limb < n_limbs) {
        __uint128_t acc = (__uint128_t)k_val << shift;
        target[limb] = (uint64_t)acc;
        if (shift != 0 && limb + 1 < n_limbs) {
            target[limb + 1] = (uint64_t)(acc >> 64);
        }
    }

    for (size_t i = 0; i < n_limbs; ++i) {
        if (limbs[i] != target[i]) return false;
    }
    return true;
}

static void debug_nm1_diff(const uint64_t* limbs, size_t n_limbs, unsigned n_bits, uint64_t k_val)
{
    std::vector<uint64_t> target(n_limbs, 0);

    size_t limb = n_bits >> 6;
    unsigned shift = n_bits & 63u;

    if (limb < n_limbs) {
        __uint128_t acc = (__uint128_t)k_val << shift;
        target[limb] = (uint64_t)acc;
        if (shift != 0 && limb + 1 < n_limbs) {
            target[limb + 1] = (uint64_t)(acc >> 64);
        }
    }

    
}

static size_t highest_nonzero_limb(const uint64_t* limbs, size_t n_limbs)
{
    for (size_t i = n_limbs; i-- > 0;) {
        if (limbs[i] != 0) return i;
    }
    return 0;
}

// --- GŁÓWNY TEST PROTHA ---
bool execute_proth_test_fft(uint64_t k, unsigned n, uint64_t* limbs, FFTContext& ctx, int report_step) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    double prof_init_powm = 0.0;
    double prof_loop_fft = 0.0;
    double prof_loop_rest = 0.0;
    double prof_final_verify = 0.0;
    double prof_red_stage1 = 0.0;
    uint64_t prof_red_stage1_ok = 0;
    uint64_t prof_red_stage1_fallback = 0;

    dbg_hn1 = 0;
    dbg_hn2 = 0;
    dbg_hn3p = 0;
    dbg_ret_nohigh = 0;
    dbg_ret_false = 0;
    dbg_ret_true_end = 0;
    dbg_borrow_fix = 0;

    // KROK 1: SZYBKA BRAMKA
    static const uint64_t extra_primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67 };
    for (uint64_t p : extra_primes) {
        if (is_divisible_by_gate(k, n, p)) return false;
    }

    // KROK 2: PRZYGOTOWANIE STRUKTUR GMP
    static thread_local mpz_t g_res, g_nm1, g_N, g_val;
    static thread_local bool init = false;
    if (!init) {
        mpz_inits(g_res, g_nm1, g_N, g_val, nullptr);
        init = true;
    }

    mpz_set_ui(g_val, k);
    mpz_mul_2exp(g_N, g_val, n);
    mpz_add_ui(g_N, g_N, 1);
    mpz_sub_ui(g_nm1, g_N, 1);

    // KROK 3: WYBÓR BAZY 
    uint64_t a = 0;
    static const uint64_t preferred_bases[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, };
    for (uint64_t b : preferred_bases) {
        mpz_set_ui(g_val, b);
        if (mpz_jacobi(g_val, g_N) == -1) { a = b; break; }
    }
    if (a == 0) {
        for (uint64_t b = 2; b < 100; ++b) {
            mpz_set_ui(g_val, b);
            if (mpz_jacobi(g_val, g_N) == -1) { a = b; break; }
        }
    }

    // KROK 4: INICJALIZACJA ŚWIADKA: g_res = a^k mod N
    mpz_set_ui(g_res, a);
    mpz_set_ui(g_val, k);
    mpz_powm(g_res, g_res, g_val, g_N);

    const size_t n_limbs_safe = ctx.fft_len / 4;
    std::memset(limbs, 0, n_limbs_safe * sizeof(uint64_t));

    size_t exported_count = 0;
    mpz_export(limbs, &exported_count, -1, 8, 0, 0, g_res);

    // --- CICHE WCZYTYWANIE CHECKPOINTU (NADPISUJE INICJALIZACJĘ JEŚLI PLIK ISTNIEJE) ---
    std::string db_file = "check_k" + std::to_string(k) + "_n" + std::to_string(n) + ".bin";
    uint64_t recovered_i = 0;
    load_checkpoint(recovered_i, n_limbs_safe, limbs, db_file);
    // ---------------------------------------------------------------------------------

    // KROK 5: GŁÓWNA PĘTLA KWADRATOWAŃ
    ctx.last_max_diff = 0.0;
    ctx.prof_pack = 0.0;
    ctx.prof_fwd = 0.0;
    ctx.prof_square = 0.0;
    ctx.prof_inv = 0.0;
    ctx.prof_unpack = 0.0;
    ctx.prof_calls = 0;

    int last_reported_pct = -1;
    const unsigned reduce_every = 1;

    size_t last_hi = (n + 64) / 64 + 10;
    size_t temp_len = ctx.fft_len;
    unsigned p = 0;
    while (temp_len > 1) { temp_len >>= 1; p++; }

    std::string gear;
    if (p == 18) gear = "Radix-4 (Optimal Bypass)";
    else if (p % 3 == 0) gear = "Radix-8 (Max Speed)";
    else if (p % 2 == 0) gear = "Radix-4 (Optimal)";
    else gear = "Radix-8 (Mixed)";

    static bool gear_reported = false;
    if (!gear_reported) {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "  [HYBRID ENGINE] Selected: " << gear << " (FFT_Len = 2^" << p << ")" << std::endl;
        gear_reported = true;
    }

    // --- GŁÓWNA PĘTLA (START OD RECOVERED_I) ---
    for (unsigned i = (unsigned)recovered_i; i < (n - 1); ++i)
    {
        // 1. Kwadratowanie w FFT
        fft_square_karatsuba(ctx, limbs, n_limbs_safe, n, k);

        size_t max_sq_limbs = (2 * n) / 64 + 4;
        if (max_sq_limbs < n_limbs_safe) {
            std::memset(limbs + max_sq_limbs, 0, (n_limbs_safe - max_sq_limbs) * sizeof(uint64_t));
            for (size_t j = max_sq_limbs; j-- > (2 * n) / 64 + 1;) {
                if (limbs[j] == 0xFFFFFFFFFFFFFFFFULL) limbs[j] = 0;
            }
        }

        size_t cut_limb = n >> 6;
        size_t search_start = (last_hi * 2) + 5;
        if (search_start > n_limbs_safe) search_start = n_limbs_safe;

        size_t current_hi = highest_nonzero_limb(limbs, search_start);
        last_hi = current_hi;

        // 2. REDUKCJA
        if (current_hi >= cut_limb) {
            if ((i % reduce_every) == 0 || i == (n - 2)) {
                std::vector<uint64_t> backup_limbs(n_limbs_safe);
                std::memcpy(backup_limbs.data(), limbs, n_limbs_safe * sizeof(uint64_t));

                bool reduced = proth_reduce_stage1(limbs, n_limbs_safe, n, k);

                if (reduced) {
                    prof_red_stage1_ok++;
                }
                else {
                    prof_red_stage1_fallback++;
                    mpz_t safe_N, safe_res;
                    mpz_init(safe_N); mpz_init(safe_res);
                    mpz_set_ui(safe_N, 1);
                    mpz_mul_2exp(safe_N, safe_N, n);
                    mpz_mul_ui(safe_N, safe_N, k);
                    mpz_add_ui(safe_N, safe_N, 1);

                    mpz_import(safe_res, n_limbs_safe, -1, 8, 0, 0, backup_limbs.data());
                    mpz_mod(safe_res, safe_res, safe_N);
                    std::memset(limbs, 0, n_limbs_safe * sizeof(uint64_t));
                    mpz_export(limbs, nullptr, -1, 8, 0, 0, safe_res);

                    mpz_clear(safe_N); mpz_clear(safe_res);
                }
            }
        }

        // --- OKRESOWY CICHE ZAPISYWANIE (CHECKPOINT) ---
        if (i > 0 && i % 10000 == 0) {
            save_checkpoint((uint64_t)i, n_limbs_safe, limbs, db_file);
        }

        // 3. MONITOROWANIE
        int current_pct = (int)((double)(i + 1) * 100.0 / (double)(n - 1));
        if (report_step > 0 && current_pct != last_reported_pct && current_pct > 0 && current_pct % report_step == 0) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cout << "    [STEP] k=" << k << " -> " << current_pct << "% "
                << "| Max Err: " << std::scientific << std::setprecision(2)
                << ctx.last_max_diff << std::endl;
            last_reported_pct = current_pct;
        }
    }

    // KROK 6: WERYFIKACJA KOŃCOWA
    bool success = is_nm1_limbs(limbs, n_limbs_safe, n, k);

    if (!success) {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        debug_nm1_diff(limbs, n_limbs_safe, n, k);
    }

    if (success) {
        {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cout << "\n[!] FOUND PRIME: " << k << " * 2^" << n << " + 1 (a=" << a << ")" << std::endl;
            std::cout << "\a";
        }
        std::string line = std::to_string(k) + " * 2^" + std::to_string(n) + " + 1 (a=" + std::to_string(a) + ")";
        append_line_txt("primes.txt", line);

        // SPRZĄTANIE: Usuwamy checkpoint po znalezieniu liczby pierwszej
        std::remove(db_file.c_str());
    }
    else {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        std::cout << "[+] " << k << " * 2^" << n << " + 1 is COMPOSITE" << std::endl;

        // SPRZĄTANIE: Usuwamy checkpoint po zakończeniu testu (composite)
        std::remove(db_file.c_str());
    }

    return success;
}

