// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "proth_logic.h"
#include "checkpoint_manager.h" 
#include "fft_engine.h"
#include "storage_core.h"
#include "log_config.h"
#include "gpu_proth_engine.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <atomic>
#include <map>
#include <utility>
#include <immintrin.h>
#include <sys/ioctl.h>
#include <unistd.h>

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
// Clear the live line before load_checkpoint only when there is a checkpoint (it may print a line): cleared at every
// candidate's start, the Batch line would vanish right after each refresh with dozens of short candidates per second.
// Call with g_io_mutex held.
static void clear_live_line_for_checkpoint(const std::string& db_file) {
    if (!g_live_line_active) return;
    if (FILE* f = fopen(db_file.c_str(), "rb")) {
        fclose(f);
        clear_live_line();
    }
}
// Writes the live line, cut to the terminal width: "\r\033[K" clears only the last row of a wrapped line, so a line
// wider than the terminal (e.g. on a phone) left a new row behind on every refresh. Call with g_io_mutex held.
static void print_live_line(const std::string& s) {
    std::string out = "  " + s;
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 20 && out.size() >= ws.ws_col) out.resize(ws.ws_col - 1);
    std::cout << "\r\033[K" << out << std::flush;
    g_live_line_active = true;
}
// Several candidates one at a time in a terminal: one live line ([current/total] k*2^n+1 ... total ETA), replaced by
// the next candidate; "not prime" results only with --detail (they are in composite.txt), as in batch mode.
static bool one_by_one() { return g_progress && g_progress_tty && !g_batch_agg && g_batch_total > 1; }

static std::string fmt_duration(double sec);

// =====================================================================
// BATCH PROGRESS LINE (g_batch_agg): shared by all workers, refreshed at most once per ~1 s
// (from the squaring loop of any worker and after every finished candidate). Outside a terminal: a line every 5 % of candidates.
// =====================================================================
static std::atomic<unsigned long long> g_b_done{ 0 };
static std::atomic<unsigned long long> g_b_primes{ 0 };
// Squarings done by all workers in this run, the tests still running included: an ETA from finished candidates
// alone jumps when workers start and end together. ETA = elapsed * (all squarings - done) / done
// (g_batch_total_iters, cli.cpp).
static std::atomic<unsigned long long> g_b_iters{ 0 };
static std::atomic<unsigned long long> g_b_fin_iters{ 0 };   // n - 1 of every finished candidate (one_by_one total ETA)
static std::atomic<long long> g_b_last_ms{ -1000000 };
static std::chrono::steady_clock::time_point g_b_t0;
static unsigned long long g_b_next_line = 0;   // next threshold (number of candidates) for a line outside a terminal

void batch_progress_start() {
    g_b_t0 = std::chrono::steady_clock::now();
    g_b_done = 0;
    g_b_primes = 0;
    g_b_iters = 0;
    g_b_fin_iters = 0;
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
    const unsigned long long it = g_b_iters.load(), tot_it = g_batch_total_iters;
    double eta = 0.0;
    bool have_eta = false;
    // "?" for the first minute: the speed is not known yet (start-up, card set-up)
    const bool warm = (el >= 60.0 || done > 0);
    if (warm && it > 0 && tot_it > 0) {   // from the squarings, the running tests included
        eta = el / (double)it * (double)(tot_it > it ? tot_it - it : 0);
        have_eta = true;
    }
    else if (done) {
        eta = el / (double)done * (double)(total - done);
        have_eta = true;
    }
    std::ostringstream ps;
    ps << std::fixed << std::setprecision(1) << "Batch: " << done << " / " << total << " (" << pct << "%) | primes: "
       << g_b_primes.load() << " | elapsed " << fmt_duration(el);
    if (done < total) ps << " | ETA " << (have_eta ? fmt_duration(eta) : std::string("?"));   // finished: no ETA
    std::lock_guard<std::mutex> lock(g_io_mutex);
    if (g_stdout_tty) {
        print_live_line(ps.str());
        if (done >= total) { std::cout << std::endl; g_live_line_active = false; }
    }
    else {
        std::cout << "  " << ps.str() << std::endl;
    }
}

void batch_note_done(bool prime, unsigned n) {
    if (prime) ++g_b_primes;
    g_b_fin_iters += (n > 0 ? n - 1 : 0);
    ++g_b_done;   // also counted one-at-a-time: the progress line shows [current/total]
    // one at a time: the batch summary once, after the last candidate (it replaces the live line). Batch mode: at most
    // once per ~1 s like the squaring loop, forced only for the last candidate (forcing it after every one made the line
    // flicker with dozens of short candidates per second)
    const bool last = g_b_done.load() >= g_batch_total;
    if (g_batch_agg) batch_print(last);
    else if (one_by_one() && last) batch_print(true);
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
// Shared by the CPU test and the GPU test (--gpu): the same gate, base and result lines.
// Quick gate: N divisible by a small prime -> not prime. A console line, shown as the "not prime" lines; no line in
// composite.txt (no RES64; sieved files never get here).
static bool quick_gate(uint64_t k, unsigned n) {
    static const uint64_t extra_primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67 };
    for (uint64_t p : extra_primes) {
        if (is_divisible_by_gate(k, n, p)) {
            if (!(g_batch_agg || one_by_one()) || g_detail) {
                std::lock_guard<std::mutex> lock(g_io_mutex);
                clear_live_line();
                std::cout << "[+] " << k << "*2^" << n << "+1 is not prime: divisible by " << p << " (trial division)"
                          << std::endl;
            }
            return true;
        }
    }
    return false;
}

// The base a: the first preferred base with Jacobi (a/N) = -1, then 2..99 (0 = none). tmp = scratch.
static uint64_t choose_base(const mpz_t N, mpz_t tmp) {
    static const uint64_t preferred_bases[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, };
    for (uint64_t b : preferred_bases) {
        mpz_set_ui(tmp, b);
        if (mpz_jacobi(tmp, N) == -1) return b;
    }
    for (uint64_t b = 2; b < 100; ++b) {
        mpz_set_ui(tmp, b);
        if (mpz_jacobi(tmp, N) == -1) return b;
    }
    return 0;
}

// Result line (console + the primes file / composite.txt) and removal of the checkpoint. low_limb = the lowest 64 bits
// of the final plain X (fully reduced): RES64 = low_limb + 1.
static void report_result(uint64_t k, unsigned n, bool success, uint64_t low_limb, double elapsed_seconds, uint64_t a,
                          const std::string& db_file) {
    unsigned long long decimal_digits = static_cast<unsigned long long>(n * 0.30102999566 + std::log10(k)) + 1;

    if (success) {
        // Proth's theorem proves primality only for k < 2^n; above that the same test gives a probable prime
        const bool proof = n >= 64 || (k >> n) == 0;
        const char* what = proof ? "is prime!" : "is a probable prime! (k >= 2^n: no Proth proof)";
        {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            std::cout << "\n[!] FOUND " << (proof ? "PRIME: " : "PROBABLE PRIME: ") << k << "*2^" << n << "+1 " << what << " ("
                << decimal_digits << " decimal digits) Time : " << std::fixed << std::setprecision(3)
                << elapsed_seconds << " sec. (a=" << a << ")" << std::endl;
            std::cout << "\a";
        }

        std::string line = std::to_string(k) + "*2^" + std::to_string(n) + "+1 " + what + " (" +
            std::to_string(decimal_digits) + " decimal digits) Time : " +
            std::to_string(elapsed_seconds) + " sec. (a=" + std::to_string(a) + ")";
        append_line_txt(g_primes_file, line);

        // CLEANUP: Remove checkpoint after finding a prime
        std::remove(db_file.c_str());
        std::remove((db_file + ".time").c_str());
    }
    else {
        // COMMUNITY STANDARD: native LSB read (no GMP in between)
        // proth_reduce_stage1 left the final, fully reduced result in limbs[0].
        // This avoids a +1/-1 drift caused by different 'mp_limb_t' vs 'uint64_t' representations.
        uint64_t res64 = low_limb + 1;

        // in batch mode and one at a time in a terminal "not prime" results only with --detail (they are in composite.txt)
        if (!(g_batch_agg || one_by_one()) || g_detail) {
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
        std::remove((db_file + ".time").c_str());
    }
}

// --- PROTH TEST ON THE CARD (--gpu) ---
// The whole test of one candidate on the card (gpu_proth_engine.cu), as execute_proth_test_fft: the same quick gate,
// base a, checkpoint file (the plain X: a test started on the CPU continues on the card and back), Gerbicz-Li segments
// (64 first, then L = M = 256, the tail by a square root; LOTR_GEC_L / LOTR_GEC_OFF / LOTR_GEC_INJECT as on the CPU)
// and result lines / files (report_result). The NTT is exact: no Max Err, no switching of the FFT length.
// Returns 1 = prime, 0 = not prime, -1 = the card failed (a CUDA error or 5 failed checks in a row; no result written,
// the checkpoint kept): the caller hands the candidate to a CPU worker and stops the card; -2 = no NTT plan for this
// candidate (e.g. a very small n): the candidate goes to the CPU, the card takes the next one.
int execute_proth_test_gpu(uint64_t k, unsigned n, int device, int report_step) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    if (quick_gate(k, n)) return 0;

    struct Mpz {   // cleared on every return
        mpz_t N, Y, V, W, t;
        Mpz() { mpz_inits(N, Y, V, W, t, nullptr); }
        ~Mpz() { mpz_clears(N, Y, V, W, t, nullptr); }
    } z;
    mpz_set_ui(z.N, (unsigned long)k);
    mpz_mul_2exp(z.N, z.N, n);
    mpz_add_ui(z.N, z.N, 1);
    const uint64_t a = choose_base(z.N, z.t);

    // start: X = a^k mod N, or the checkpoint (plain X, any stored limb count)
    mpz_set_ui(z.Y, (unsigned long)a);
    mpz_set_ui(z.t, (unsigned long)k);
    mpz_powm(z.Y, z.Y, z.t, z.N);
    const std::string db_file = "check_k" + std::to_string(k) + "_n" + std::to_string(n) + ".bin";
    const size_t n_limbs = (size_t)n / 64 + 2;   // N < 2^(n+64)
    std::vector<uint64_t> buf(n_limbs, 0);
    mpz_export(buf.data(), nullptr, -1, 8, 0, 0, z.Y);
    uint64_t recovered_i = 0;
    bool ck_loaded;
    {   // the "[CHECKPOINT] ... converted" line must not land on the live line
        std::lock_guard<std::mutex> lock(g_io_mutex);
        clear_live_line_for_checkpoint(db_file);
        ck_loaded = load_checkpoint(recovered_i, n_limbs, buf.data(), db_file);
    }
    if (ck_loaded) {
        mpz_import(z.Y, n_limbs, -1, 8, 0, 0, buf.data());
        mpz_mod(z.Y, z.Y, z.N);
    }
    else recovered_i = 0;
    if (recovered_i > n - 1) recovered_i = 0;
    // test time of the earlier runs (the result shows the total time, not the time since the resume)
    const double prev_seconds = (recovered_i > 0) ? checkpoint_seconds(db_file) : 0.0;
    // scaled form Y = X*2^n mod N (as the CPU classic path)
    mpz_mul_2exp(z.Y, z.Y, n);
    mpz_mod(z.Y, z.Y, z.N);
    auto plain_to_buf = [&](mpz_t y) {   // buf <- X = -k*Y mod N
        mpz_mul_ui(z.t, y, (unsigned long)k);
        mpz_neg(z.t, z.t);
        mpz_mod(z.t, z.t, z.N);
        std::fill(buf.begin(), buf.end(), 0);
        mpz_export(buf.data(), nullptr, -1, 8, 0, 0, z.t);
    };

    unsigned gec_L_main = 256;
    if (const char* e = std::getenv("LOTR_GEC_L")) {
        const unsigned v = (unsigned)std::strtoul(e, nullptr, 10);
        if (v >= 1) gec_L_main = v;
    }
    const bool gec_off = (std::getenv("LOTR_GEC_OFF") != nullptr);
    unsigned long gec_inject_at = 0;
    bool gec_injected = false;
    if (const char* e = std::getenv("LOTR_GEC_INJECT")) gec_inject_at = std::strtoul(e, nullptr, 10);

    GpuEngine* g = nullptr;
    try {
        std::string info;
        g = gpu_create(n, k, device, info);
        if (!g) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            std::cout << "  [GPU] " << k << "*2^" << n << "+1: " << info << " - not tested on the card" << std::endl;
            return -2;
        }
        static std::string last_info;   // the plan once (and again when it changes)
        {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            if (info != last_info || g_detail) {
                clear_live_line();
                // batch mode: the plan without k and n - several candidates at once on the card (a batch may also have many n),
                // so a single k above the batch line would be misleading. A candidate with another plan prints the new one.
                if (g_batch_agg && !g_detail) std::cout << "  [GPU] plan: " << info << std::endl;
                else std::cout << "  [GPU] " << k << "*2^" << n << "+1: " << info << std::endl;
                last_info = info;
            }
        }
        gpu_set(g, GPU_X, z.Y);

        unsigned current_i = (unsigned)recovered_i;
        bool first_block = true;
        unsigned consecutive_failures = 0;
        int next_report_pct = report_step;
        const auto test_start_time = clock::now();
        auto t_prog_last = test_start_time;
        int next_prog_pct = 10;
        unsigned long long b_last_i = current_i;   // this test's squarings already added to g_b_iters
        auto progress_tick = [&](unsigned i_now) {
            if (g_batch_agg) {   // batch mode: only the shared batch line
                if (i_now > b_last_i) { g_b_iters.fetch_add(i_now - b_last_i); b_last_i = i_now; }
                if (g_stdout_tty) batch_print(false);
                return;
            }
            if (!g_progress) return;
            const auto now = clock::now();
            const int pct = (int)((double)i_now * 100.0 / (double)(n - 1));
            if (g_progress_tty ? secd(now - t_prog_last).count() < 1.0 : pct < next_prog_pct) return;
            t_prog_last = now;
            while (next_prog_pct <= pct) next_prog_pct += 10;
            const double el = secd(now - test_start_time).count();
            const double done_here = (double)i_now - (double)recovered_i;
            const double ms_it = (done_here > 0) ? el * 1000.0 / done_here : 0.0;
            const double eta = ms_it * ((double)(n - 1) - (double)i_now) / 1000.0;
            const bool multi = (g_batch_total > 1);
            std::ostringstream ps;
            if (multi) ps << "[" << (g_b_done.load() + 1) << "/" << g_batch_total << "] ";
            if (multi || !g_progress_tty) ps << k << "*2^" << n << "+1 | ";
            ps << "GPU | " << std::fixed << std::setprecision(2) << (100.0 * (double)i_now / (double)(n - 1)) << "% | iter "
               << i_now << " / " << (n - 1) << " | " << std::setprecision(4) << ms_it << " ms/iter | ETA " << fmt_duration(eta);
            std::lock_guard<std::mutex> lock(g_io_mutex);
            if (g_progress_tty) print_live_line(ps.str());
            else std::cout << "    " << ps.str() << std::endl;
        };

        // LOTR_GPU_SEGTIME=1: per Gerbicz segment the time of the squarings (+ the d products), of the check
        // (W = d^(2^L) v_0, d v_M, both read back and compared on the host) and of the checkpoint (X read back, plain,
        // written) - shows how the run time differs from the bare square time of the benchmarks
        static const bool seg_time = std::getenv("LOTR_GPU_SEGTIME") != nullptr;
        while (current_i < n - 1) {
            const unsigned remaining = (n - 1) - current_i;
            unsigned L = (first_block && gec_L_main > 64) ? 64u : gec_L_main;
            unsigned M = L;
            if ((uint64_t)L * M > remaining) {
                L = std::max(1u, (unsigned)std::sqrt((double)remaining));
                M = remaining / L;
            }
            const unsigned steps_to_do = L * M;
            const auto block_start_time = clock::now();

            // [ROLLBACK POINT] v_0 = verified state at the start of the segment, d = v_0
            gpu_copy(g, GPU_V0, GPU_X);
            if (!gec_off) gpu_copy(g, GPU_D, GPU_X);
            for (unsigned j = 0; j < M; ++j) {
                const unsigned long it0 = current_i + (unsigned long)j * L;
                if (gec_inject_at != 0 && !gec_injected && gec_inject_at >= it0 && gec_inject_at < it0 + L) {
                    const unsigned c1 = (unsigned)(gec_inject_at - it0 + 1);
                    gpu_square(g, GPU_X, c1);
                    gpu_poke(g, GPU_X);   // simulated memory/computation error
                    gpu_square(g, GPU_X, L - c1);
                    gec_injected = true;
                    std::lock_guard<std::mutex> lock(g_io_mutex);
                    clear_live_line();
                    std::cerr << "\n[GEC-TEST] injected error after iteration" << gec_inject_at << " (GPU)" << std::endl;
                }
                else gpu_square(g, GPU_X, L);
                if (!gec_off && j + 1 < M) gpu_mul(g, GPU_D, GPU_D, GPU_X);
                gpu_sync(g);   // the thread sleeps here; it also paces the queue of the card
                progress_tick((unsigned)(it0 + L));
            }

            // Gerbicz-Li: v_0 * d^(2^L) == d * v_M (mod N); values compared after the reduction mod N on the host
            const auto t_seg_sq = clock::now();
            double t_gpu_chk = 0;
            bool block_ok = true;
            bool bounds_fail = false;
            if (!gec_off) {
                gpu_copy(g, GPU_W, GPU_D);
                gpu_square(g, GPU_W, L);
                gpu_mul(g, GPU_W, GPU_W, GPU_V0);   // left side: v_0 * d^(2^L)
                gpu_mul(g, GPU_D, GPU_D, GPU_X);    // right side: d * v_M
                if (seg_time) { gpu_sync(g); t_gpu_chk = secd(clock::now() - t_seg_sq).count(); }
                const bool okw = gpu_get(g, GPU_W, z.W), okd = gpu_get(g, GPU_D, z.V);
                bounds_fail = !okw || !okd;
                block_ok = !bounds_fail && mpz_cmp(z.W, z.V) == 0;
            }
            const auto t_seg_chk = clock::now();
            if (!block_ok) {
                ++consecutive_failures;
                std::lock_guard<std::mutex> lock(g_io_mutex);
                clear_live_line();
                std::cerr << "\n[!] GERBICZ-LI CHECK FAILED (GPU" << (bounds_fail ? ", digits out of bounds" : "")
                          << "): computation error in segment of iterations" << current_i << ".." << (current_i + steps_to_do)
                          << std::endl;
                if (consecutive_failures >= 5) {
                    std::cerr << "[FATAL] k=" << k << " n=" << n << ": " << consecutive_failures
                              << " failed verifications in a row on the GPU from iteration " << current_i
                              << " - the card hands the candidate back (checkpoint kept)." << std::endl;
                    gpu_destroy(g);
                    return -1;
                }
                std::cout << "    -> [ROLLBACK] Reverting to iteration " << current_i << " (GPU)" << std::endl;
                gpu_copy(g, GPU_X, GPU_V0);
                continue;
            }
            consecutive_failures = 0;

            // END OF SEGMENT (SUCCESS): checkpoint (plain X)
            current_i += steps_to_do;
            first_block = false;
            gpu_get(g, GPU_X, z.Y);
            const auto t_seg_get = clock::now();
            plain_to_buf(z.Y);
            const auto t_seg_plain = clock::now();
            if (save_checkpoint((uint64_t)current_i, n_limbs, buf.data(), db_file))
                save_checkpoint_seconds(db_file, prev_seconds + secd(clock::now() - test_start_time).count());
            if (seg_time) {
                const auto t_end = clock::now();
                const double ts = secd(t_seg_sq - block_start_time).count(), tc = secd(t_seg_chk - t_seg_sq).count();
                const double tg = secd(t_seg_get - t_seg_chk).count(), tp = secd(t_seg_plain - t_seg_get).count();
                const double tw = secd(t_end - t_seg_plain).count(), all = secd(t_end - block_start_time).count();
                std::lock_guard<std::mutex> lock(g_io_mutex);
                clear_live_line();
                std::cout << "    [SEG] " << L << "x" << M << " = " << steps_to_do << " squarings: " << std::fixed << std::setprecision(3)
                          << ts << " s (" << std::setprecision(4) << ts * 1000.0 / steps_to_do << " ms/iter with the d products)"
                          << std::setprecision(3) << " | check " << tc << " s (card " << t_gpu_chk << ", host get + compare "
                          << tc - t_gpu_chk << ") | checkpoint: get X " << tg << " s, plain " << tp << " s, write " << tw
                          << " s | segment " << all << " s = " << std::setprecision(4) << all * 1000.0 / steps_to_do
                          << " ms/iter" << std::endl;
            }

            const int current_pct = (int)((double)current_i * 100.0 / (double)(n - 1));
            if (report_step > 0 && current_pct >= next_report_pct) {
                const double iter_ms = secd(clock::now() - block_start_time).count() / steps_to_do * 1000.0;
                std::lock_guard<std::mutex> lock(g_io_mutex);
                clear_live_line();
                std::cout << "    [STEP] k=" << k << " -> " << current_pct << "% | GPU | " << std::fixed << std::setprecision(4)
                          << iter_ms << " ms/iter" << std::endl;
                while (next_report_pct <= current_pct) next_report_pct += report_step;
            }
        }
        const double elapsed_seconds = prev_seconds + secd(clock::now() - test_start_time).count();

        // back to the plain X = -k*Y mod N: prime <=> X == N - 1
        gpu_get(g, GPU_X, z.Y);
        gpu_destroy(g);
        g = nullptr;
        mpz_mul_ui(z.Y, z.Y, (unsigned long)k);
        mpz_neg(z.Y, z.Y);
        mpz_mod(z.Y, z.Y, z.N);
        mpz_sub_ui(z.t, z.N, 1);
        const bool success = (mpz_cmp(z.Y, z.t) == 0);
        const uint64_t low = (mpz_sgn(z.Y) == 0) ? 0 : (uint64_t)mpz_getlimbn(z.Y, 0);
        report_result(k, n, success, low, elapsed_seconds, a, db_file);
        return success ? 1 : 0;
    }
    catch (const std::exception& e) {
        if (g) {
            try { gpu_destroy(g); } catch (...) {}
        }
        std::lock_guard<std::mutex> lock(g_io_mutex);
        clear_live_line();
        std::cerr << "\n[GPU] " << e.what() << " - the card hands " << k << "*2^" << n << "+1 back (checkpoint kept)"
                  << std::endl;
        return -1;
    }
}

// --- HANDOVER TO THE CARD (proth_logic.h) ---
// The running CPU tests: the last checkpoint and whether the card asked for the candidate. An entry lives while the
// test runs (CpuRunGuard); a request that arrives after the end leaves one behind, dispatch drops it (cpu_run_clear).
struct CpuRun { unsigned long long iter = 0; bool yield = false; };
static std::mutex g_cpu_run_mtx;
static std::map<std::pair<uint64_t, unsigned>, CpuRun> g_cpu_runs;
static thread_local bool t_handed_over = false;
unsigned long long cpu_run_iter(uint64_t k, unsigned n) {
    std::lock_guard<std::mutex> lk(g_cpu_run_mtx);
    const auto it = g_cpu_runs.find({ k, n });
    return it == g_cpu_runs.end() ? 0 : it->second.iter;
}
void cpu_run_request(uint64_t k, unsigned n) {
    std::lock_guard<std::mutex> lk(g_cpu_run_mtx);
    g_cpu_runs[{ k, n }].yield = true;
}
void cpu_run_clear(uint64_t k, unsigned n) {
    std::lock_guard<std::mutex> lk(g_cpu_run_mtx);
    g_cpu_runs.erase({ k, n });
}
bool cpu_handed_over() { return t_handed_over; }
static bool cpu_run_note(uint64_t k, unsigned n, unsigned long long iter) {   // the progress; true = hand over now
    std::lock_guard<std::mutex> lk(g_cpu_run_mtx);
    CpuRun& r = g_cpu_runs[{ k, n }];
    r.iter = iter;
    return r.yield;
}
struct CpuRunGuard {
    uint64_t k; unsigned n;
    ~CpuRunGuard() { cpu_run_clear(k, n); }
};

std::string get_fft_engine_name(const FFTContext& ctx);
// --- PROTH TEST ---
bool execute_proth_test_fft(uint64_t k, unsigned n, uint64_t* limbs, FFTContext& ctx, int report_step) {
    using clock = std::chrono::high_resolution_clock;
    using secd = std::chrono::duration<double>;

    t_handed_over = false;
    // STEP 1: QUICK GATE
    if (quick_gate(k, n)) return false;

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
    const uint64_t a = choose_base(g_N, g_val);

    // STEP 4: WITNESS INITIALIZATION: g_res = a^k mod N
    mpz_set_ui(g_res, a);
    mpz_set_ui(g_val, k);
    mpz_powm(g_res, g_res, g_val, g_N);

    // Limbs of the working value = digit capacity of the lower half of the FFT buffer (16-bit digits: fft_len/4;
    // fft_len = 2^p or 3*2^p, a multiple of 64)
    const size_t n_limbs_safe = ctx.fft_len * ctx.digit_bits / 64;
    // The working value (< N, n + bits of k) must fit; otherwise every square is garbage (Max Err 1e30) and the
    // exports below write past the buffers. dispatch calibrates for the largest N - this only stops the test cleanly
    // if it ever happens.
    if (n_limbs_safe * 64 < (size_t)mpz_sizeinbase(g_N, 2) + 1) {
        std::lock_guard<std::mutex> lock(g_io_mutex);
        clear_live_line();
        std::cerr << "\n[ERROR] " << k << "*2^" << n << "+1 does not fit the calibrated FFT " << ctx.fft_len << " ("
                  << n_limbs_safe * 64 << " bits) - test skipped, run it on its own (--n " << n << " --k " << k << ")"
                  << std::endl;
        return false;
    }
    std::memset(limbs, 0, n_limbs_safe * sizeof(uint64_t));

    size_t exported_count = 0;
    mpz_export(limbs, &exported_count, -1, 8, 0, 0, g_res);

    // --- CHECKPOINT ---
    std::string db_file = "check_k" + std::to_string(k) + "_n" + std::to_string(n) + ".bin";
    uint64_t recovered_i = 0;
    {   // the "[CHECKPOINT] ... converted" line must not land on the live line
        std::lock_guard<std::mutex> lock(g_io_mutex);
        clear_live_line_for_checkpoint(db_file);
        load_checkpoint(recovered_i, n_limbs_safe, limbs, db_file);
    }
    // test time of the earlier runs (the result shows the total time, not the time since the resume)
    const double prev_seconds = (recovered_i > 0) ? checkpoint_seconds(db_file) : 0.0;
    // the card may take this candidate over (dispatch, --gpu): asked already before the first segment -> at once
    CpuRunGuard run_guard{ k, n };
    if (cpu_run_note(k, n, recovered_i)) {
        t_handed_over = true;
        return false;
    }
    // ---------------------------------------------------------------------------------

    // SCALED FORM (fft_uses_scaled_form): the FFT squaring returns -k*S mod N (no division by k), so the working value
    // is Y = X*2^n mod N; squarings and Gerbicz-Li products keep this form. Conversions only here (start), when saving
    // a checkpoint (files keep the plain X, compatible with older versions) and before the final check.
    const bool scaled = fft_uses_scaled_form();
    auto to_scaled = [&](uint64_t* buf) {   // Y = X*2^n mod N
        mpz_import(g_val, n_limbs_safe, -1, 8, 0, 0, buf);
        mpz_mul_2exp(g_val, g_val, n);
        mpz_mod(g_val, g_val, g_N);
        std::memset(buf, 0, n_limbs_safe * sizeof(uint64_t));
        mpz_export(buf, nullptr, -1, 8, 0, 0, g_val);
    };
    auto from_scaled = [&](const uint64_t* src, uint64_t* dst) {   // X = Y*2^-n = -k*Y mod N
        mpz_import(g_val, n_limbs_safe, -1, 8, 0, 0, src);
        mpz_mul_ui(g_val, g_val, (unsigned long)k);
        mpz_neg(g_val, g_val);
        mpz_mod(g_val, g_val, g_N);
        std::memset(dst, 0, n_limbs_safe * sizeof(uint64_t));
        mpz_export(dst, nullptr, -1, 8, 0, 0, g_val);
    };
    if (scaled) to_scaled(limbs);
    std::vector<uint64_t> ckpt_plain(scaled ? n_limbs_safe : 0);

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

    // Wide digits or a 3*2^p FFT length (fft_real_only): the FFT has only a squaring, so the Gerbicz-Li multiplications
    // use x*y = ((x+y)^2 - (x-y)^2) / 4 (mod N): two squarings plus additions and halvings mod N on limbs.
    const bool wide_digits = fft_real_only(ctx);
    std::vector<uint64_t> N_limbs, gec_t2;
    if (wide_digits) {
        N_limbs.assign(n_limbs_safe, 0);
        N_limbs[0] = 1;
        const unsigned sh = n & 63u;
        const size_t li = n >> 6;
        N_limbs[li] |= k << sh;
        if (sh != 0 && li + 1 < n_limbs_safe) N_limbs[li + 1] |= k >> (64 - sh);
        gec_t2.assign(n_limbs_safe, 0);
    }
    // r = a + b mod N (a, b in [0, N); 2N fits in n_limbs_safe limbs)
    auto add_mod = [&](uint64_t* r, const uint64_t* a, const uint64_t* b) {
        unsigned char c = 0;
        for (size_t i = 0; i < n_limbs_safe; ++i) {
            unsigned long long s;
            c = _addcarry_u64(c, a[i], b[i], &s);
            r[i] = s;
        }
        bool ge = true;   // r >= N ?
        for (size_t i = n_limbs_safe; i-- > 0;) {
            if (r[i] != N_limbs[i]) { ge = (r[i] > N_limbs[i]); break; }
        }
        if (ge) {
            unsigned char br = 0;
            for (size_t i = 0; i < n_limbs_safe; ++i) {
                unsigned long long d;
                br = _subborrow_u64(br, r[i], N_limbs[i], &d);
                r[i] = d;
            }
        }
    };
    // r = a - b mod N (a, b in [0, N))
    auto sub_mod = [&](uint64_t* r, const uint64_t* a, const uint64_t* b) {
        unsigned char br = 0;
        for (size_t i = 0; i < n_limbs_safe; ++i) {
            unsigned long long d;
            br = _subborrow_u64(br, a[i], b[i], &d);
            r[i] = d;
        }
        if (br) {
            unsigned char c = 0;
            for (size_t i = 0; i < n_limbs_safe; ++i) {
                unsigned long long s;
                c = _addcarry_u64(c, r[i], N_limbs[i], &s);
                r[i] = s;
            }
        }
    };
    // r = r / 2 mod N (N odd: an odd r becomes even after adding N)
    auto half_mod = [&](uint64_t* r) {
        unsigned char c = 0;
        if (r[0] & 1) {
            for (size_t i = 0; i < n_limbs_safe; ++i) {
                unsigned long long s;
                c = _addcarry_u64(c, r[i], N_limbs[i], &s);
                r[i] = s;
            }
        }
        for (size_t i = 0; i + 1 < n_limbs_safe; ++i) r[i] = (r[i] >> 1) | (r[i + 1] << 63);
        r[n_limbs_safe - 1] = (r[n_limbs_safe - 1] >> 1) | ((uint64_t)c << 63);
    };

    // dst = x*y mod N (x, y in [0, N); dst may alias one of the operands)
    auto mul_values = [&](uint64_t* dst, const uint64_t* x, const uint64_t* y) {
        if (wide_digits) {
            uint64_t* const S = test_mul_result.data();
            uint64_t* const D = gec_t2.data();
            add_mod(S, x, y);
            sub_mod(D, x, y);
            size_t hs = n_limbs_safe, hd = n_limbs_safe;   // full scan
            square_step(S, hs, 0);
            square_step(D, hd, 0);
            sub_mod(dst, S, D);
            half_mod(dst);
            half_mod(dst);
            return;
        }
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
        // Several candidates one at a time: show which one is running ([current/total] k*2^n+1)
        const bool multi = (g_batch_total > 1);
        std::ostringstream ps;
        if (multi) ps << "[" << (g_b_done.load() + 1) << "/" << g_batch_total << "] ";
        if (multi || !g_progress_tty) ps << k << "*2^" << n << "+1 | ";   // one candidate in a terminal: the header already names it
        ps << std::fixed << std::setprecision(2) << (100.0 * (double)i_now / (double)(n - 1)) << "%";
        if (one_by_one()) {
            // the whole queue: the squarings left (this test and the waiting candidates) at the current speed
            const unsigned long long fin = g_b_fin_iters.load(), tot = g_batch_full_iters;
            const double left = (double)(tot > fin + i_now ? tot - fin - i_now : 0);
            ps << " | ETA " << fmt_duration(eta) << " | primes: " << g_b_primes.load() << " | total ETA "
               << fmt_duration(ms_it * left / 1000.0) << " | " << ms_it << " ms/iter";
        }
        else {
            ps << " | iter " << i_now;
            if (!multi) ps << " / " << (n - 1);   // shorter line with the candidate prefix (fits the terminal width)
            ps << " | " << ms_it << " ms/iter | ETA " << fmt_duration(eta);
        }
        std::lock_guard<std::mutex> lock(g_io_mutex);
        if (g_progress_tty) {
            print_live_line(ps.str());
        }
        else {
            std::cout << "    " << ps.str() << std::endl;
        }
    };
    unsigned long long b_last_i = current_i;   // this test's squarings already added to g_b_iters
    auto progress_tick = [&](unsigned i_now) {
        if (g_batch_agg) {   // batch mode: only the shared batch line (at most once per second)
            if (i_now > b_last_i) {   // (after a Gerbicz-Li rollback i_now goes back: nothing is added twice)
                g_b_iters.fetch_add(i_now - b_last_i);
                b_last_i = i_now;
            }
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
        for (int pi = 0; pi < 16; ++pi) ctx.prof_sub[pi] = 0.0;
        blk_reduce_sec = 0.0;
#endif
        // 1. SEGMENT SQUARINGS (after every L: d *= v_j)
        for (unsigned j = 0; j < M; ++j) {
            if (ctx.ibdwt) {
                // IBDWT: runs of up to ib_run squarings with the digits kept inside the FFT context (limbs only at the end
                // of a run; the outputs are already reduced, no finish_value). A run ends right after the GEC test injection.
                // ib_run: 256 (with 64 the pack + full unpack at the ends of a run cost a lot on large lengths);
                // LOTR_IB_RUN=<n> changes it.
                static const unsigned ib_run = []() {
                    const char* e = std::getenv("LOTR_IB_RUN");
                    const unsigned v = e ? (unsigned)std::strtoul(e, nullptr, 10) : 0u;
                    return v >= 1 ? v : 256u;
                }();
                for (unsigned s = 0; s < L;) {
                    const unsigned long it0 = current_i + j * L + s;
                    unsigned chunk = std::min(ib_run, L - s);
                    const bool inj = gec_inject_at != 0 && !gec_injected && gec_inject_at >= it0 && gec_inject_at < it0 + chunk;
                    if (inj) chunk = (unsigned)(gec_inject_at - it0 + 1);
                    fft_square_repeat(ctx, limbs, n_limbs_safe, n, k, chunk);
                    seg_max_err = std::max(seg_max_err, ctx.last_max_diff);
                    s += chunk;
                    if (g_progress) progress_tick(current_i + j * L + s);
                    if (inj) {
                        limbs[1] ^= 0x10ULL;   // simulated memory/FFT error
                        gec_injected = true;
                        std::cerr << "\n[GEC-TEST] injected error after iteration" << gec_inject_at << std::endl;
                    }
                }
            }
            else {
                for (unsigned s = 0; s < L; ++s) {
                    square_step(limbs, last_hi, current_i + j * L + s);
                    if (g_progress && (s & 63) == 0) progress_tick(current_i + j * L + s + 1);
                    if (gec_inject_at != 0 && !gec_injected && (unsigned long)(current_i + j * L + s) == gec_inject_at) {
                        limbs[1] ^= 0x10ULL;   // simulated memory/FFT error
                        gec_injected = true;
                        std::cerr << "\n[GEC-TEST] injected error after iteration" << gec_inject_at << std::endl;
                    }
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
            ps << "\n    [PASSES] ms/iter (sums over the FFTs of one iteration: 2 = real FFT, 3 = the 3-FFT fallback) Bailey two-pass: p0, p1 | Stockham: p0..p6=passes:";
            for (int pi = 0; pi < 8; ++pi) ps << " p" << pi << "=" << std::fixed << std::setprecision(3) << ctx.prof_pass[pi] * per;
            // parts of the fused-square passes (average thread; max = busiest thread, compare with the pass time above)
            const double* sb = ctx.prof_sub;
            ps << std::fixed << std::setprecision(4)
               << "\n    [SUBPASS] ms/iter A: gather=" << sb[0] * per << " st1=" << sb[1] * per << " fft=" << sb[2] * per
               << " store=" << sb[3] * per << " max=" << sb[4] * per
               << " | mid: gather=" << sb[5] * per << " fwd=" << sb[6] * per << " sq=" << sb[7] * per
               << " inv=" << sb[8] * per << " store=" << sb[9] * per << " max=" << sb[10] * per
               << " | B': gather=" << sb[11] * per << " st1=" << sb[12] * per << " fft=" << sb[13] * per
               << " store=" << sb[14] * per << " max=" << sb[15] * per;
            prof_line = ps.str();
        }
#endif

        // =========================================================
        // 2. SEGMENT VERIFICATION (GERBICZ-LI + FFT round-off error check)
        // =========================================================
        bool block_ok = true;
        bool roundoff_fail = false;   // failed on the round-off error (deterministic: a repeat gives the same error)

        // A. FFT round-off error
        if (seg_max_err > 0.45) {
            block_ok = false;
            roundoff_fail = true;
            std::lock_guard<std::mutex> lock(g_io_mutex);
            clear_live_line();
            // a precision limit of the FFT length (IBDWT then moves to a longer one), not a hardware fault
            std::cerr << "\n[!] FFT round-off error: Max Err = " << seg_max_err << " > 0.45 - segment rejected" << std::endl;
        }

        // B. Gerbicz-Li: v_0 * d^(2^L) == d * v_M (mod N)
#ifdef LOTR_PROFILE
        auto t_check0 = clock::now();
#endif
        if (block_ok && !gec_off) {
            std::memcpy(gec_w.data(), gec_d.data(), seg_bytes);
            size_t hi_w = hi_init;
            if (ctx.ibdwt) {
                fft_square_repeat(ctx, gec_w.data(), n_limbs_safe, n, k, L);
                seg_max_err = std::max(seg_max_err, ctx.last_max_diff);
            }
            else {
                for (unsigned s = 0; s < L; ++s) square_step(gec_w.data(), hi_w, 0);
            }
            mul_values(gec_w.data(), gec_w.data(), backup_limbs.data());   // left side: v_0 * d^(2^L)
            mul_values(gec_d.data(), gec_d.data(), limbs);                 // right side: d * v_M

            if (seg_max_err > 0.45) {
                block_ok = false;
                roundoff_fail = true;
                std::lock_guard<std::mutex> lock(g_io_mutex);
                clear_live_line();
                std::cerr << "\n[!] FFT round-off error (in Gerbicz check): Max Err = " << seg_max_err
                          << " > 0.45 - segment rejected" << std::endl;
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
            // IBDWT: the same segment failing twice in a row is a precision limit, not a hardware glitch (the rounding
            // is deterministic, a repeat gives the same error): continue on the next longer IBDWT length. The working value
            // (limbs) does not depend on the FFT length; the limb count of the IBDWT is (n + 192)/64 rounded up for every
            // length, so n_limbs_safe still fits. A round-off failure switches at once (a repeat would give the same error);
            // a Gerbicz mismatch (possible hardware glitch) is repeated first. The longer length is for this test only: the
            // worker restores the calibrated context before the next candidate (dcore.cpp).
            if (ctx.ibdwt && (consecutive_failures >= 2 || roundoff_fail)) {
                size_t next_len = 0;
                for (size_t Lb = 256; Lb <= 4 * ctx.fft_len && next_len == 0; Lb *= 2) {
                    for (const size_t c : { Lb, Lb + Lb / 2 }) {
                        if (c > ctx.fft_len && fft_ibdwt_supported(c)) { next_len = c; break; }
                    }
                }
                const unsigned next_bits = next_len ? (unsigned)(((size_t)n + 192 + next_len - 1) / next_len) : 0;
                // the IBDWT square needs n_limbs*64 >= n + 192 only (ctx.digit_bits is the limb count of the start)
                if (next_len != 0 && ((size_t)n + 192 + 63) / 64 <= n_limbs_safe) {
                    const size_t old_len = ctx.fft_len;
                    fft_destroy(ctx);
                    ctx = FFTContext();
                    if (fft_init(ctx, FFTBackend::INTERNAL, next_len)) {
                        ctx.digit_bits = next_bits;
                        ctx.ibdwt = true;
                        std::cout << "    -> [FFT] " << k << "*2^" << n << "+1: Max Err " << seg_max_err << " on FFT " << old_len
                                  << ": continuing on the longer IBDWT FFT " << next_len << " from iteration " << current_i
                                  << " (result stays valid)" << std::endl;
                        consecutive_failures = 0;
                        std::memcpy(limbs, backup_limbs.data(), seg_bytes);
                        last_hi = hi_init;
                        continue;
                    }
                    std::cerr << "\n[ERROR] FFT init failed for len " << next_len << std::endl;
                    return false;
                }
            }
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
        bool ck_saved;
        if (scaled) {
            from_scaled(limbs, ckpt_plain.data());
            ck_saved = save_checkpoint((uint64_t)current_i, n_limbs_safe, ckpt_plain.data(), db_file);
        }
        else {
            ck_saved = save_checkpoint((uint64_t)current_i, n_limbs_safe, limbs, db_file);
        }
        if (ck_saved) save_checkpoint_seconds(db_file, prev_seconds + secd(clock::now() - test_start_time).count());
        // the card asked for this candidate (dispatch, --gpu): it resumes from the checkpoint just saved
        if (ck_saved && cpu_run_note(k, n, current_i) && current_i < n - 1) {
            t_handed_over = true;
            return false;
        }

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
    double elapsed_seconds = prev_seconds + secd(test_end_time - test_start_time).count();

   
    if (scaled) {   // back to the plain X (final check and RES64 as before)
        std::vector<uint64_t> plain(n_limbs_safe);
        from_scaled(limbs, plain.data());
        std::memcpy(limbs, plain.data(), n_limbs_safe * sizeof(uint64_t));
    }
    bool success = is_nm1_limbs(limbs, n_limbs_safe, n, k);
    report_result(k, n, success, limbs[0], elapsed_seconds, a, db_file);

    return success;
       

}