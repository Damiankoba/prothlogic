// Copyright (C) Damian Koba.


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "proth_logic.h"
#include "checkpoint_manager.h" 
#include "fft_engine.h"
#include "storage_core.h"
#include "log_config.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <atomic>

extern std::mutex g_io_mutex;

// Progress line refreshed in place (log_config.h: g_progress_tty). It must be cleared before any other message
// so that the text does not overlap it. Call with g_io_mutex held.
static bool g_live_line_active = false;
static void clear_live_line() {
    if (g_live_line_active) {
        std::cout << "\r\033[K" << std::flush;
        g_live_line_active = false;
    }
}

static std::string fmt_duration(double sec);

// =====================================================================
// BATCH PROGRESS LINE (g_batch_agg): shared by all workers, refreshed at most once per ~1 s
// (from the squaring loop of any worker and after every finished candidate). Outside a terminal: a line every 5 % of candidates.
// =====================================================================
static std::atomic<unsigned long long> g_b_done{ 0 };
static std::atomic<unsigned long long> g_b_primes{ 0 };
static std::atomic<long long> g_b_last_ms{ -1000000 };
static std::chrono::steady_clock::time_point g_b_t0;
static unsigned long long g_b_next_line = 0;   // next threshold (number of candidates) for a line outside a terminal

void batch_progress_start() {
    g_b_t0 = std::chrono::steady_clock::now();
    g_b_done = 0;
    g_b_primes = 0;
    g_b_next_line = 0;
}

static void batch_print(bool force) {
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_b_t0).count();
    const long long ms = (long long)(el * 1000.0);
    const unsigned long long done = g_b_done.load();
    const unsigned long long total = g_batch_total;
    if (!g_stdout_tty) {
        // outside a terminal: a line every 5 % of candidates (and at the end)
        std::lock_guard<std::mutex> lock(g_io_mutex);
        if (done < g_b_next_line && done < total) return;
        const unsigned long long step = std::max<unsigned long long>(1, total / 20);
        while (g_b_next_line <= done) g_b_next_line += step;
    }
    else if (!force) {
        long long last = g_b_last_ms.load();
        if (ms - last < 1000) return;
        if (!g_b_last_ms.compare_exchange_strong(last, ms)) return;   // another worker is printing right now
    }
    const double pct = total ? 100.0 * (double)done / (double)total : 0.0;
    const double eta = done ? el / (double)done * (double)(total - done) : 0.0;
    std::ostringstream ps;
    ps << std::fixed << std::setprecision(1) << "Batch: " << done << " / " << total << " (" << pct << "%) | primes: "
       << g_b_primes.load() << " | elapsed " << fmt_duration(el) << " | ETA " << (done ? fmt_duration(eta) : std::string("?"));
    std::lock_guard<std::mutex> lock(g_io_mutex);
    if (g_stdout_tty) {
        std::cout << "\r\033[K  " << ps.str() << std::flush;
        g_live_line_active = true;
        if (done >= total) { std::cout << std::endl; g_live_line_active = false; }
    }
    else {
        std::cout << "  " << ps.str() << std::endl;
    }
}

void batch_note_done(bool prime) {
    if (!g_batch_agg) return;
    if (prime) ++g_b_primes;
    ++g_b_done;
    batch_print(true);
}

// Human-readable duration: "2d 03h", "1h 23m", "12m 05s", "45s"
static std::string fmt_duration(double sec) {
    if (sec < 0) sec = 0;
    const unsigned long long s = (unsigned long long)(sec + 0.5);
    char buf[48];
    if (s >= 86400ULL) std::snprintf(buf, sizeof(buf), "%llud %02lluh", s / 86400ULL, (s % 86400ULL) / 3600ULL);
    else if (s >= 3600ULL) std::snprintf(buf, sizeof(buf), "%lluh %02llum", s / 3600ULL, (s % 3600ULL) / 60ULL);
    else if (s >= 60ULL) std::snprintf(buf, sizeof(buf), "%llum %02llus", s / 60ULL, s % 60ULL);
    else std::snprintf(buf, sizeof(buf), "%llus", s);
    return buf;
}


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
    
    
    if (k_limb + 1 < n_limbs) {
        uint64_t old = limbs[k_limb + 1];
        limbs[k_limb + 1] += high + carry;
        carry = (limbs[k_limb + 1] < old);
    }
    
    
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
        return false;
    }

    // --- MEMORY OPTIMIZATION: allocated once, thread-safe, flat buffer for all recursion levels ---
    static thread_local std::vector<uint64_t> tl_high_buffers;
    if (tl_high_buffers.size() < (size_t)(32 * n_limbs)) {
        tl_high_buffers.resize(32 * n_limbs, 0);
    }
    // Pointer to the buffer segment owned by the current recursion depth
    uint64_t* high = tl_high_buffers.data() + (depth * n_limbs);

    for (int round = 0; round < 10; ++round) {
        std::memset(high, 0, n_limbs * sizeof(uint64_t)); // fast C-style clear instead of a slower std::fill

        size_t hn = extract_high_shifted(limbs, n_limbs, n_bits, high);

        bool fully_reduced = false;

        if (hn == 0) {
            fully_reduced = true;
        }
        else if (hn == 1) {
            if (high[0] < k_val) {
                fully_reduced = true;
            }
            else if (high[0] == k_val) {
                if (cmp_proth_modulus(limbs, n_limbs, n_bits, k_val) < 0) {
                    fully_reduced = true;
                }
            }
        }

        if (fully_reduced) {
            for (int norm = 0; norm < 5; ++norm) {
                if (cmp_proth_modulus(limbs, n_limbs, n_bits, k_val) >= 0) {
                    sub_proth_modulus_once(limbs, n_limbs, n_bits, k_val);
                }
                else {
                    break;
                }
            }
            return true;
        }

        proth_clear_above_n(limbs, n_limbs, n_bits);

        uint64_t r = div_big_by_u64(high, hn, k_val);

        if (!proth_reduce_stage1_rec(high, n_limbs, n_bits, k_val, depth + 1)) {
            fprintf(stderr, "\n[DEBUG] RECURSION FAILED AT DEPTH %d!\n", depth);
            return false;
        }

        size_t qn = n_limbs;
        while (qn > 0 && high[qn - 1] == 0) --qn;

        add_u64_shifted(limbs, n_limbs, r, n_bits);

        bool borrow = sub_big_inplace(limbs, n_limbs, high, qn);
        if (borrow) {
            add_proth_modulus_once(limbs, n_limbs, n_bits, k_val);

            if ((int64_t)limbs[n_limbs - 1] < 0) {
                add_proth_modulus_once(limbs, n_limbs, n_bits, k_val);
            }

            proth_clear_above_n(limbs, n_limbs, n_bits + 64);
        }

    }

    return false;
}


static bool proth_reduce_stage1(uint64_t* limbs, size_t n_limbs, unsigned n_bits, uint64_t k_val)
{
    return proth_reduce_stage1_rec(limbs, n_limbs, n_bits, k_val, 0);
}

// Quick check (no large buffers): is the value already fully reduced, i.e. would the full reduction
// (proth_reduce_stage1) change nothing? The conditions are exactly those of the "fully_reduced" path:
// the part above n bits is zero or a single limb <= k (for == k additionally value < N),
// and the first cmp of the normalization loop gives < 0 (value < k*2^n+1), so nothing is subtracted.
// hi_hint = highest nonzero limb (from highest_nonzero_limb): everything above is zero, so we do not scan it
// again (for an FFT much larger than the number, e.g. 2^19 at N=4.3M, that is about 64k zero limbs).
// LOTR_VERIFY_UNPACK=1 compares this shortcut against the full reduction.
static inline int proth_cmp_hint(const uint64_t* limbs, size_t n_limbs, size_t n_bits, uint64_t k, size_t hi_hint)
{
    const size_t k_limb = n_bits / 64;
    const size_t k_shift = n_bits % 64;
    const uint64_t mod_low = k << k_shift;
    const uint64_t mod_high = (k_shift == 0) ? 0 : (k >> (64 - k_shift));
    const size_t top = (hi_hint < n_limbs) ? hi_hint : (n_limbs - 1);
    for (size_t i = top; i > k_limb + 1; --i) {
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

static inline bool proth_needs_no_reduction(const uint64_t* limbs, size_t n_limbs, unsigned n_bits, uint64_t k_val, size_t hi_hint)
{
    const size_t cut_limb = n_bits >> 6;
    const unsigned cut_shift = n_bits & 63u;
    size_t hn = 0;
    uint64_t h0 = 0;
    if (cut_limb < n_limbs) {
        auto hval = [&](size_t i) -> uint64_t {
            if (cut_shift == 0) return limbs[i];
            uint64_t v = limbs[i] >> cut_shift;
            if (i + 1 < n_limbs) v |= (limbs[i + 1] << (64 - cut_shift));
            return v;
        };
        const size_t scan_top = (hi_hint + 2 < n_limbs) ? (hi_hint + 2) : n_limbs;
        for (size_t i = scan_top; i-- > cut_limb;) {
            if (hval(i) != 0) { hn = i - cut_limb + 1; break; }
        }
        if (hn == 1) h0 = hval(cut_limb);
    }
    if (hn > 1) return false;
    if (hn == 1 && h0 > k_val) return false;
    return proth_cmp_hint(limbs, n_limbs, n_bits, k_val, hi_hint) < 0;
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

static size_t highest_nonzero_limb(const uint64_t* limbs, size_t n_limbs)
{
    // First skip zero blocks of 8 limbs (no branch per limb), then scan exactly from the top.
    size_t i = n_limbs;
    while (i >= 8) {
        const uint64_t any = limbs[i - 1] | limbs[i - 2] | limbs[i - 3] | limbs[i - 4] |
                             limbs[i - 5] | limbs[i - 6] | limbs[i - 7] | limbs[i - 8];
        if (any != 0) break;
        i -= 8;
    }
    for (; i-- > 0;) {
        if (limbs[i] != 0) return i;
    }
    return 0;
}
std::string get_fft_engine_name(const FFTContext& ctx);
// --- PROTH TEST ---
bool execute_proth_test_fft(uint64_t k, unsigned n, uint64_t* limbs, FFTContext& ctx, int report_step) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    // STEP 1: QUICK GATE
    static const uint64_t extra_primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67 };
    for (uint64_t p : extra_primes) {
        if (is_divisible_by_gate(k, n, p)) return false;
    }

    // STEP 2: PREPARING GMP STRUCTURES
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

    // STEP 3: SELECTING THE BASE 
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

    // STEP 4: WITNESS INITIALIZATION: g_res = a^k mod N
    mpz_set_ui(g_res, a);
    mpz_set_ui(g_val, k);
    mpz_powm(g_res, g_res, g_val, g_N);

    const size_t n_limbs_safe = ctx.fft_len / 4;
    std::memset(limbs, 0, n_limbs_safe * sizeof(uint64_t));

    size_t exported_count = 0;
    mpz_export(limbs, &exported_count, -1, 8, 0, 0, g_res);

    // --- CHECKPOINT ---
    std::string db_file = "check_k" + std::to_string(k) + "_n" + std::to_string(n) + ".bin";
    uint64_t recovered_i = 0;
    load_checkpoint(recovered_i, n_limbs_safe, limbs, db_file);
    // ---------------------------------------------------------------------------------

   // STEP 5: MAIN SQUARING LOOP
    ctx.last_max_diff = 0.0;
    ctx.prof_pack = 0.0;
    ctx.prof_fwd = 0.0;
    ctx.prof_square = 0.0;
    ctx.prof_inv = 0.0;
    ctx.prof_unpack = 0.0;
    ctx.prof_calls = 0;

    size_t last_hi = (n + 64) / 64 + 10;

    static bool gear_reported = !(g_detail || !g_progress);   // selected engine: with --detail or --report
    if (!gear_reported) {
#pragma omp critical
        {
            if (!gear_reported) {
                std::cout << "  [HYBRID ENGINE] Selected: " << get_fft_engine_name(ctx) << "\n" << std::flush;
                gear_reported = true;
            }
        }
    }

    // --- MEMORY OPTIMIZATION: persistent GMP fallback objects (prevents leaks) ---
    static thread_local mpz_t safe_N, safe_res;
    static thread_local bool gmp_fallback_init = false;
    if (!gmp_fallback_init) {
        mpz_init(safe_N);
        mpz_init(safe_res);
        gmp_fallback_init = true;
    }

    // --- GERBICZ-LI: computation correctness check (segment rollback on error) ---
    // u_i = a^(k*2^i) mod N. Segment = M intervals of L squarings: v_j = u_{i0+j*L}, v_{j+1} = v_j^(2^L).
    // Accumulator d = v_0*v_1*...*v_{M-1} (multiplied after every L squarings). Then d^(2^L) = v_1*...*v_M = d*v_M/v_0,
    // so the check is: v_0 * d^(2^L) == d * v_M (mod N). An error in any squaring of the segment (FFT, RAM, cache)
    // breaks this equality with probability close to 1. Cost: (M+1) multiplications + L squarings per segment of L*M iterations.
    // LOTR_GEC_L=<L> changes L (default 256, segment of 65536 iterations); LOTR_GEC_OFF=1 disables the check (timing comparisons only).
    unsigned gec_L_main = 256;
    if (const char* e = std::getenv("LOTR_GEC_L")) {
        const unsigned v = (unsigned)std::strtoul(e, nullptr, 10);
        if (v >= 1) gec_L_main = v;
    }
    const bool gec_off = (std::getenv("LOTR_GEC_OFF") != nullptr);
    // Check test: LOTR_GEC_INJECT=<iteration> flips one state bit (once) after that iteration; the check must detect it and roll back the segment.
    unsigned long gec_inject_at = 0;
    bool gec_injected = false;
    if (const char* e = std::getenv("LOTR_GEC_INJECT")) gec_inject_at = std::strtoul(e, nullptr, 10);

    const size_t seg_bytes = n_limbs_safe * sizeof(uint64_t);
    std::vector<uint64_t> backup_limbs(n_limbs_safe, 0);       // v_0: verified state at the start of the segment
    std::vector<uint64_t> gec_d(n_limbs_safe, 0);              // accumulator d
    std::vector<uint64_t> gec_w(n_limbs_safe, 0);              // d^(2^L) * v_0
    std::vector<uint64_t> test_mul_result(n_limbs_safe, 0);
    std::vector<uint64_t> fallback_backup(n_limbs_safe, 0);    // copy taken before the full reduction (slow path only)

    const size_t cut_limb = n >> 6;
    const size_t cmp_limbs = std::min(n_limbs_safe, (size_t)(n / 64 + 2));   // N < 2^(n+64): this many limbs are enough for the comparison
    const size_t hi_init = (n + 64) / 64 + 10;
    double seg_max_err = 0.0;
#ifdef LOTR_PROFILE   // measurements enabled with the -DLOTR_PROFILE build flag
    double blk_reduce_sec = 0.0;
#endif

    // Reduce the value in buf to [0, N): fast path, full reduction as a fallback, and GMP as the last resort.
    auto reduce_value = [&](uint64_t* buf, size_t cur_hi, unsigned i_dbg) {
        if (cur_hi < cut_limb) return;
        bool reduced = true;
        if (!proth_needs_no_reduction(buf, n_limbs_safe, n, k, cur_hi)) {
            std::memcpy(fallback_backup.data(), buf, seg_bytes);
            reduced = proth_reduce_stage1(buf, n_limbs_safe, n, k);
        }
        else {
            // Check (LOTR_VERIFY_UNPACK=1): a full reduction of a copy must not change anything.
            static const bool verify_reduce = (std::getenv("LOTR_VERIFY_UNPACK") != nullptr);
            if (verify_reduce) {
                std::vector<uint64_t> tmp(buf, buf + n_limbs_safe);
                proth_reduce_stage1(tmp.data(), n_limbs_safe, n, k);
                if (std::memcmp(tmp.data(), buf, seg_bytes) != 0) {
                    std::fprintf(stderr, "[VERIFY_REDUCE] MISMATCH: fast reduction path != full reduction (iteration %u)\n", i_dbg);
                    std::memcpy(buf, tmp.data(), seg_bytes);
                }
            }
        }

        if (!reduced) {
            // Use the preallocated GMP objects
            mpz_set_ui(safe_N, 1);
            mpz_mul_2exp(safe_N, safe_N, n);
            mpz_mul_ui(safe_N, safe_N, k);
            mpz_add_ui(safe_N, safe_N, 1);

            mpz_import(safe_res, n_limbs_safe, -1, 8, 0, 0, fallback_backup.data());
            mpz_mod(safe_res, safe_res, safe_N);
            std::memset(buf, 0, seg_bytes);
            mpz_export(buf, nullptr, -1, 8, 0, 0, safe_res);
        }
    };

    // Cleanup after the unpack (square or product) and reduction. hi_state = tracked highest nonzero limb.
    auto finish_value = [&](uint64_t* buf, size_t& hi_state, unsigned i_dbg) {
        size_t max_sq_limbs = (2 * n) / 64 + 4;
        if (max_sq_limbs < n_limbs_safe) {
            std::memset(buf + max_sq_limbs, 0, (n_limbs_safe - max_sq_limbs) * sizeof(uint64_t));
            for (size_t j = max_sq_limbs; j-- > (2 * n) / 64 + 1;) {
                if (buf[j] == 0xFFFFFFFFFFFFFFFFULL) buf[j] = 0;
            }
        }

        size_t search_start = (hi_state * 2) + 5;
        if (search_start > n_limbs_safe) search_start = n_limbs_safe;

        size_t current_hi = highest_nonzero_limb(buf, search_start);
        hi_state = current_hi;

        reduce_value(buf, current_hi, i_dbg);
    };

    // buf = buf^2 mod N
    auto square_step = [&](uint64_t* buf, size_t& hi_state, unsigned i_dbg) {
        fft_square_karatsuba(ctx, buf, n_limbs_safe, n, k);
        seg_max_err = std::max(seg_max_err, ctx.last_max_diff);
#ifdef LOTR_PROFILE
        auto t_post0 = clock::now();
#endif
        finish_value(buf, hi_state, i_dbg);
#ifdef LOTR_PROFILE
        blk_reduce_sec += secd(clock::now() - t_post0).count();
#endif
    };

    // dst = x*y mod N (x, y in [0, N); dst may alias one of the operands)
    auto mul_values = [&](uint64_t* dst, const uint64_t* x, const uint64_t* y) {
        fft_mul_karatsuba(ctx, test_mul_result.data(), x, y, n_limbs_safe, n, k);
        seg_max_err = std::max(seg_max_err, ctx.last_max_diff);
        size_t hi_state = n_limbs_safe;   // full scan
        finish_value(test_mul_result.data(), hi_state, 0);
        std::memcpy(dst, test_mul_result.data(), seg_bytes);
    };

    unsigned current_i = (unsigned)recovered_i;
    int next_report_pct = report_step;
    bool first_block_measured = false;
    unsigned consecutive_failures = 0;

    auto test_start_time = clock::now();

    // Test progress (when --report is not given): in a terminal a single line refreshed about every 1 s,
    // outside a terminal or with several workers at once a plain line every 10 %.
    auto t_prog_last = test_start_time;
    int next_prog_pct = 10;
    auto show_progress = [&](unsigned i_now) {
        const double el = secd(clock::now() - test_start_time).count();
        const double done_here = (double)i_now - (double)recovered_i;   // iterations in this run (after a resume)
        const double ms_it = (done_here > 0) ? el * 1000.0 / done_here : 0.0;
        const double eta = ms_it * ((double)(n - 1) - (double)i_now) / 1000.0;
        std::ostringstream ps;
        ps << std::fixed << std::setprecision(2) << (100.0 * (double)i_now / (double)(n - 1)) << "% | iter "
           << i_now << " / " << (n - 1) << " | " << ms_it << " ms/iter | ETA " << fmt_duration(eta);
        std::lock_guard<std::mutex> lock(g_io_mutex);
        if (g_progress_tty) {
            std::cout << "\r\033[K  " << ps.str() << std::flush;
            g_live_line_active = true;
        }
        else {
            std::cout << "    " << k << "*2^" << n << "+1: " << ps.str() << std::endl;
        }
    };
    auto progress_tick = [&](unsigned i_now) {
        if (g_batch_agg) {   // batch mode: only the shared batch line (at most once per second)
            if (g_stdout_tty) batch_print(false);
            return;
        }
        if (g_progress_tty) {
            const auto now = clock::now();
            if (secd(now - t_prog_last).count() >= 1.0) {
                show_progress(i_now);
                t_prog_last = now;
            }
        }
        else {
            const int pct = (int)((double)i_now * 100.0 / (double)(n - 1));
            if (pct >= next_prog_pct) {
                show_progress(i_now);
                while (next_prog_pct <= pct) next_prog_pct += 10;
            }
        }
    };

    while (current_i < (n - 1)) {
        // Choose segment L and M. The first segment is short (quick speed estimate); the tail is shortened using a square root.
        const unsigned remaining = (n - 1) - current_i;
        unsigned L = (!first_block_measured && gec_L_main > 64) ? 64u : gec_L_main;
        unsigned M = L;
        if ((uint64_t)L * M > remaining) {
            L = std::max(1u, (unsigned)std::sqrt((double)remaining));
            M = remaining / L;
        }
        const unsigned steps_to_do = L * M;

        // [ROLLBACK POINT] v_0 = verified state at the start of the segment
        std::memcpy(backup_limbs.data(), limbs, seg_bytes);
        if (!gec_off) std::memcpy(gec_d.data(), limbs, seg_bytes);
        ctx.last_max_diff = 0.0;
        seg_max_err = 0.0;
        auto block_start_time = clock::now();
#ifdef LOTR_PROFILE
        // Per-segment profile (ctx.prof_* is filled by fft_square_karatsuba)
        ctx.prof_pack = 0.0; ctx.prof_fwd = 0.0; ctx.prof_square = 0.0;
        ctx.prof_inv = 0.0; ctx.prof_unpack = 0.0; ctx.prof_calls = 0;
        ctx.prof_unpack_clear = 0.0; ctx.prof_unpack_scalar = 0.0; ctx.prof_unpack_finalfix = 0.0;
        for (int pi = 0; pi < 8; ++pi) ctx.prof_pass[pi] = 0.0;
        blk_reduce_sec = 0.0;
#endif
        // 1. SEGMENT SQUARINGS (after every L: d *= v_j)
        for (unsigned j = 0; j < M; ++j) {
            for (unsigned s = 0; s < L; ++s) {
                square_step(limbs, last_hi, current_i + j * L + s);
                if (g_progress && (s & 63) == 0) progress_tick(current_i + j * L + s + 1);
                if (gec_inject_at != 0 && !gec_injected && (unsigned long)(current_i + j * L + s) == gec_inject_at) {
                    limbs[1] ^= 0x10ULL;   // simulated memory/FFT error
                    gec_injected = true;
                    std::cerr << "\n[GEC-TEST] injected error after iteration" << gec_inject_at << std::endl;
                }
            }
            if (!gec_off && j + 1 < M) mul_values(gec_d.data(), gec_d.data(), limbs);
        }
#ifdef LOTR_PROFILE
        auto t_loop_end = clock::now();
        // PROFILE: ms/iter split into stages (without checkpoint and Gerbicz check; "other" includes the d multiplications)
        std::string prof_line;
        {
            const double per = 1000.0 / (double)steps_to_do;
            double pack_ms = ctx.prof_pack * per, fwd_ms = ctx.prof_fwd * per;
            double sq_ms = ctx.prof_square * per, inv_ms = ctx.prof_inv * per;
            double unp_ms = ctx.prof_unpack * per, red_ms = blk_reduce_sec * per;
            double loop_ms = secd(t_loop_end - block_start_time).count() * per;
            double other_ms = loop_ms - (pack_ms + fwd_ms + sq_ms + inv_ms + unp_ms + red_ms);
            std::ostringstream ps;
            ps << std::fixed << std::setprecision(3)
               << "    [PROFILE] ms/iter: pack=" << pack_ms << " fwd=" << fwd_ms
               << " square=" << sq_ms << " inv=" << inv_ms << " unpack=" << unp_ms
               << " reduce=" << red_ms << " other=" << other_ms << " | loop=" << loop_ms
               << " | unpack: conv+carry=" << ctx.prof_unpack_clear * per
               << " div=" << ctx.prof_unpack_scalar * per
               << " out=" << ctx.prof_unpack_finalfix * per;
            ps << "\n    [PASSES] ms/iter (sums over the FFTs of one iteration: 2 = real FFT, 3 = LOTR_SQ_KARATSUBA) Bailey 5-pass: p0=transp p1=rowFFT p2=transp+tw p3=rowFFT p4=transp; two-pass: p0, p1 | Stockham: p0..p6=passes:";
            for (int pi = 0; pi < 8; ++pi) ps << " p" << pi << "=" << std::fixed << std::setprecision(3) << ctx.prof_pass[pi] * per;
            prof_line = ps.str();
        }
#endif

        // =========================================================
        // 2. SEGMENT VERIFICATION (GERBICZ-LI + FFT round-off error check)
        // =========================================================
        bool block_ok = true;

        // A. FFT round-off error
        if (seg_max_err > 0.45) {
            block_ok = false;
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            std::cerr << "\n[!] AVX-512 HARDWARE ERROR: Max Err = " << seg_max_err << " > 0.45!" << std::endl;
        }

        // B. Gerbicz-Li: v_0 * d^(2^L) == d * v_M (mod N)
#ifdef LOTR_PROFILE
        auto t_check0 = clock::now();
#endif
        if (block_ok && !gec_off) {
            std::memcpy(gec_w.data(), gec_d.data(), seg_bytes);
            size_t hi_w = hi_init;
            for (unsigned s = 0; s < L; ++s) square_step(gec_w.data(), hi_w, 0);
            mul_values(gec_w.data(), gec_w.data(), backup_limbs.data());   // left side: v_0 * d^(2^L)
            mul_values(gec_d.data(), gec_d.data(), limbs);                 // right side: d * v_M

            if (seg_max_err > 0.45) {
                block_ok = false;
                std::lock_guard<std::mutex> lock(g_io_mutex);
                clear_live_line();
                std::cerr << "\n[!] AVX-512 HARDWARE ERROR (in Gerbicz check): Max Err = " << seg_max_err << " > 0.45!" << std::endl;
            }
            else if (std::memcmp(gec_w.data(), gec_d.data(), cmp_limbs * sizeof(uint64_t)) != 0) {
                block_ok = false;
                std::lock_guard<std::mutex> lock(g_io_mutex);
                clear_live_line();
                std::cerr << "\n[!] GERBICZ-LI CHECK FAILED: computation error in segment of iterations"
                          << current_i << ".." << (current_i + steps_to_do) << std::endl;
            }
        }
#ifdef LOTR_PROFILE
        const double blk_check_ms = secd(clock::now() - t_check0).count() * 1000.0;
#endif

        // =========================================================
        // 3. DECISION: ROLLBACK OR CHECKPOINT?
        // =========================================================
        if (!block_ok) {
            ++consecutive_failures;
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            if (consecutive_failures >= 5) {
                std::cerr << "\n[FATAL] k=" << k << " n=" << n << ": " << consecutive_failures
                          << " failed verifications in a row from iteration " << current_i
                          << " - aborting test (checkpoint kept, result NOT written)." << std::endl;
                return false;
            }
            std::cout << "    -> [ROLLBACK] Reverting to iteration " << current_i << " to save progress..." << std::endl;
            // Restore the verified state and repeat the segment without advancing current_i
            std::memcpy(limbs, backup_limbs.data(), seg_bytes);
            last_hi = hi_init;
            continue;
        }
        consecutive_failures = 0;

        // =========================================================
        // 4. END OF SEGMENT (SUCCESS)
        // =========================================================
        current_i += steps_to_do;
        save_checkpoint((uint64_t)current_i, n_limbs_safe, limbs, db_file);

        auto block_end_time = clock::now();
        double block_duration_sec = secd(block_end_time - block_start_time).count();
        double iter_ms = (block_duration_sec / steps_to_do) * 1000.0;   // including the Gerbicz check cost

#ifdef LOTR_PROFILE
        {
            std::ostringstream cs;
            cs << std::fixed << std::setprecision(1) << "\n    [GEC] L=" << L << " M=" << M << " check(per segment)=" << blk_check_ms << " ms";
            prof_line += cs.str();
        }
#endif

        // Estimate after the first block: with --detail or --report (measurements)
        if (!first_block_measured && (g_detail || !g_progress)) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            double est_1pct_sec = ((n - 1) * 0.01) * (iter_ms / 1000.0);
            std::cout << "    [ESTIMATE] Initial block (" << steps_to_do << " iter) finished!\n"
                << "    [ESTIMATE] Speed: " << std::fixed << std::setprecision(2) << iter_ms << " ms/iter.\n"
                << "    [ESTIMATE] Expected time for next 1%: ~" << est_1pct_sec << " seconds."
#ifdef LOTR_PROFILE
                << "\n" << prof_line
#endif
                << std::endl;
        }
        first_block_measured = true;

        // REPORTING
        int current_pct = (int)((double)(current_i) * 100.0 / (double)(n - 1));
        if (report_step > 0 && current_pct >= next_report_pct) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            std::cout << "    [STEP] k=" << k << " -> " << current_pct << "% "
                << "| Max Err: " << std::scientific << std::setprecision(2)
                << seg_max_err
                << std::fixed << std::setprecision(2) << " | " << iter_ms << " ms/iter" << std::endl;
#ifdef LOTR_PROFILE
            std::cout << prof_line << std::endl;
#endif

            while (next_report_pct <= current_pct) next_report_pct += report_step;
        }
    }

    // STEP 6: FINAL VERIFICATION

    auto test_end_time = clock::now();
    double elapsed_seconds = secd(test_end_time - test_start_time).count(); 

   
    unsigned long long decimal_digits = static_cast<unsigned long long>(n * 0.30102999566 + std::log10(k)) + 1;

    bool success = is_nm1_limbs(limbs, n_limbs_safe, n, k);

    if (success) {
        {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            std::cout << "\n[!] FOUND PRIME: " << k << "*2^" << n << "+1 is prime! ("
                << decimal_digits << " decimal digits) Time : " << std::fixed << std::setprecision(3)
                << elapsed_seconds << " sec. (a=" << a << ")" << std::endl;
            std::cout << "\a";
        }

        std::string line = std::to_string(k) + "*2^" + std::to_string(n) + "+1 is prime! (" +
            std::to_string(decimal_digits) + " decimal digits) Time : " +
            std::to_string(elapsed_seconds) + " sec. (a=" + std::to_string(a) + ")";
        append_line_txt("primes.txt", line);

        // CLEANUP: Remove checkpoint after finding a prime
        std::remove(db_file.c_str());
    }
    else {
        // COMMUNITY STANDARD: native LSB read (no GMP in between)
        // proth_reduce_stage1 left the final, fully reduced result in limbs[0].
        // This avoids a +1/-1 drift caused by different 'mp_limb_t' vs 'uint64_t' representations.
        uint64_t res64 = limbs[0] + 1;

        if (!g_batch_agg || g_detail) {   // in batch mode "not prime" results only with --detail (they are in composite.txt)
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            std::cout << "[+] " << k << "*2^" << n << "+1 is not prime. Proth RES64: 0x"
                << std::hex << std::uppercase << std::setfill('0') << std::setw(16) << res64
                << std::dec << " Time : " << std::fixed << std::setprecision(3) << elapsed_seconds << " sec. (a=" << a << ")" << std::endl;
        }

        std::ostringstream comp_line;
        comp_line << k << "*2^" << n << "+1 is not prime. Proth RES64: 0x"
            << std::hex << std::uppercase << std::setfill('0') << std::setw(16) << res64
            << std::dec << " Time : " << std::fixed << std::setprecision(3) << elapsed_seconds << " sec. (a=" << a << ")";

        append_line_txt("composite.txt", comp_line.str());

        // CLEANUP: Remove checkpoint after composite result
        std::remove(db_file.c_str());
    }

    return success;
       

}