// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "dcore.h"
#include "worker.h"
#include "fft_engine.h"
#include "fft_factorizer.h"
#include "log_config.h"
#include "proth_logic.h"
#include "gpu_proth_engine.h"
#include <thread>
#include <vector>
#include <string>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <pthread.h>
#include <sched.h>    
#include <sys/mman.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <iomanip>
#include <cstdio>

extern std::mutex g_io_mutex;   // worker.cpp (the takeover line of the card, dispatch)

// CPU set of the process at start-up (taskset). OpenMP (e.g. OMP_PROC_BIND/OMP_PLACES in the environment) can pin
// the main thread to one core at the first parallel region, and new threads (workers) inherit that pinning -
// all workers then share one core (observed: affinity 0,12, 24 workers ~20x slower).
static cpu_set_t g_proc_cpus;
static bool g_proc_cpus_ok = false;

void capture_process_affinity() {
    CPU_ZERO(&g_proc_cpus);
    g_proc_cpus_ok = (sched_getaffinity(0, sizeof(cpu_set_t), &g_proc_cpus) == 0);
}

// n-th CPU (from 0, modulo the number of available CPUs) of the saved process CPU set; -1 if the set is unknown.
static int nth_process_cpu(int n) {
    if (!g_proc_cpus_ok) return -1;
    const int cnt = CPU_COUNT(&g_proc_cpus);
    if (cnt <= 0) return -1;
    int want = n % cnt;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &g_proc_cpus)) {
            if (want == 0) return c;
            --want;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU topology and pinning (--pin and --tune). Groups = L3 domains (CCDs) of the process CPU set; per group the
// physical cores (first allowed logical CPU of each core) and their SMT siblings, from /sys.
// ---------------------------------------------------------------------------------------------------------------------
struct CpuGroup { std::vector<int> prim, sec; };

static std::string read_sys_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

static std::vector<int> parse_cpu_list(const std::string& s) {   // "0-5,12-17"
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const size_t d = part.find('-');
        try {
            if (d == std::string::npos) out.push_back(std::stoi(part));
            else for (int c = std::stoi(part.substr(0, d)); c <= std::stoi(part.substr(d + 1)); ++c) out.push_back(c);
        }
        catch (...) {}
    }
    return out;
}

static std::vector<CpuGroup> cpu_topology() {
    std::vector<int> allowed;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (g_proc_cpus_ok ? CPU_ISSET(c, &g_proc_cpus) : c < (int)std::thread::hardware_concurrency()) allowed.push_back(c);
    auto is_allowed = [&](int c) { return std::find(allowed.begin(), allowed.end(), c) != allowed.end(); };
    std::map<std::string, size_t> idx;
    std::vector<CpuGroup> groups;
    for (int c : allowed) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
        int first = c;
        for (int s : parse_cpu_list(read_sys_line(base + "/topology/thread_siblings_list")))
            if (s < first && is_allowed(s)) first = s;
        std::string l3 = read_sys_line(base + "/cache/index3/shared_cpu_list");
        if (l3.empty()) l3 = "?";
        if (!idx.count(l3)) { idx[l3] = groups.size(); groups.emplace_back(); }
        CpuGroup& g = groups[idx[l3]];
        (c == first ? g.prim : g.sec).push_back(c);
    }
    return groups;
}

// CPUs of W workers with T threads each: physical cores in group order (a worker of T | cores-per-group threads stays in
// one CCD); with W*T above the physical cores, each group's cores followed by its SMT siblings. Fewer rows = no room.
static std::vector<std::vector<int>> cpu_layout(const std::vector<CpuGroup>& groups, int W, int T) {
    size_t P = 0;
    for (const CpuGroup& g : groups) P += g.prim.size();
    // Spread evenly over the CCDs (--tune-all: not all cores in use, e.g. 8 x 1 = 4 + 4): W/D workers per CCD, each on
    // T consecutive physical cores of its CCD.
    const int D = (int)groups.size();
    if (D > 1 && W % D == 0 && (size_t)W * T <= P) {
        const size_t per = (size_t)(W / D);
        bool fit = true;
        for (const CpuGroup& g : groups) fit = fit && g.prim.size() >= per * (size_t)T;
        if (fit) {
            std::vector<std::vector<int>> out;
            for (size_t s = 0; s < per; ++s)
                for (const CpuGroup& g : groups)
                    out.emplace_back(g.prim.begin() + s * T, g.prim.begin() + (s + 1) * T);
            return out;
        }
    }
    std::vector<int> list;
    for (const CpuGroup& g : groups) {
        list.insert(list.end(), g.prim.begin(), g.prim.end());
        if ((size_t)W * T > P) list.insert(list.end(), g.sec.begin(), g.sec.end());
    }
    std::vector<std::vector<int>> out;
    for (int w = 0; w < W && (size_t)(w + 1) * T <= list.size(); ++w)
        out.emplace_back(list.begin() + (size_t)w * T, list.begin() + (size_t)(w + 1) * T);
    return out;
}

static void pin_self(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &s);
}

// This thread and its OpenMP team (libgomp keeps the team's threads for the later parallel regions of this thread):
// one CPU each.
static void pin_team(const std::vector<int>& cpus) {
    omp_set_num_threads((int)cpus.size());
    pin_self(cpus[0]);
#pragma omp parallel
    pin_self(cpus[(size_t)omp_get_thread_num() % cpus.size()]);
}

static std::string cpu_list_str(const std::vector<int>& cpus) {   // 0,1,2,3 -> "0-3"
    std::string s;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) ++j;
        if (!s.empty()) s += ",";
        s += std::to_string(cpus[i]) + (j > i ? "-" + std::to_string(cpus[j]) : "");
        i = j + 1;
    }
    return s;
}

static size_t get_next_smooth_size(size_t current) {
    size_t p = 256;
    while (p <= current) {
        p *= 2;
    }
    return p;
}


// Largest digit width for the wide-digit path: LOTR_DIGIT_BITS=<16..20> (default 18; 16 = wide digits off).
// Off as well with the old digits (LOTR_NO_BALANCED).
static unsigned wide_digits_max() {
    if (std::getenv("LOTR_NO_BALANCED") != nullptr) return 16;
    // Default 19: 3*2^16 x 19 bits holds N up to 3.73M (instead of 2^18 above 3.54M), 7*2^15 x 19 up to 4.36M; every
    // width still has to pass the GMP self-test and max err <= 0.2 (17 bits on 3*2^16: ~0.006, about x4 per bit)
    unsigned m = 19;
    if (const char* e = std::getenv("LOTR_DIGIT_BITS")) m = (unsigned)std::strtoul(e, nullptr, 10);
    if (m < 16) m = 16;
    if (m > 20) m = 20;
    return m;
}

// Calibration of one FFT length: 100 squarings of a number filled with ones; returns the max rounding error
// (or -1 when the context cannot be created). With wide digits the first squaring also runs the GMP self-test;
// *selftest_ok reports its result.
static double calibrate_len(size_t len, unsigned digit_bits, unsigned actual_n, uint64_t sample_k, bool* selftest_ok,
                            bool ibdwt = false) {
    if (selftest_ok) *selftest_ok = true;
    FFTContext test_ctx;
    if (!fft_init(test_ctx, FFTBackend::INTERNAL, len)) return -1.0;
    test_ctx.digit_bits = digit_bits;
    test_ctx.ibdwt = ibdwt;
    test_ctx.quiet_selftest = ibdwt;   // a failed IBDWT self-test goes to the summary line of auto_tune_uncached
    const size_t n_limbs = (actual_n + 63) / 64;   // limbs filled with ones
    // Limbs passed to the square: as in the test (proth_logic.cpp: fft_len*digit_bits/64) for wide digits, so the GMP
    // self-test sees the real layout; 16-bit digits keep the old calibration (n_limbs).
    const size_t n_limbs_sq = fft_real_only(test_ctx) ? len * digit_bits / 64 : n_limbs;
    const size_t safe_alloc_limbs = std::max(len, (size_t)((2 * actual_n) / 64 + 64));
    const size_t limbs_bytes_size = safe_alloc_limbs * sizeof(uint64_t);
    uint64_t* test_limbs = (uint64_t*)mmap(NULL, limbs_bytes_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (test_limbs == MAP_FAILED) {
        fft_destroy(test_ctx);
        return -1.0;
    }
    std::memset(test_limbs, 0, limbs_bytes_size);
    for (size_t i = 0; i < n_limbs; i++) test_limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
    test_limbs[0] = sample_k | 1;
    for (int step = 0; step < 100; ++step) fft_square_karatsuba(test_ctx, test_limbs, n_limbs_sq, actual_n, sample_k);
    const double err = test_ctx.last_max_diff;
    if (selftest_ok && fft_real_only(test_ctx)) *selftest_ok = (test_ctx.sq_real_state > 0);
    munmap(test_limbs, limbs_bytes_size);
    fft_destroy(test_ctx);
    return err;
}

// Set by auto_tune_fft_len when the IBDWT square is selected (dispatch passes it to the worker contexts).
static bool g_ibdwt_mode = false;

static size_t auto_tune_classic(unsigned actual_n, uint64_t sample_k, unsigned* digit_bits_out);
static size_t auto_tune_uncached(unsigned actual_n, uint64_t sample_k, unsigned* digit_bits_out);

// ---------------------------------------------------------------------------------------------------------------------
// Calibration cache: the calibration runs on one thread with GMP self-tests, which takes an hour for giants
// (N = 1.07e9: 4 FFT lengths). A calibration that took >= 30 s is saved to fft_calibration.txt in the working directory
// as "n k fft_len digit_bits ibdwt env"; the next run with the same n, the same bit length of k and k <= the saved k
// (the IBDWT loses precision with k, so a length that passes for the saved k passes for a smaller one) and the same
// calibration variables (env) uses it. Still safe: the run repeats the GMP self-test of its square, watches Max Err and
// the Gerbicz-Li check, and moves to a longer IBDWT length when they fail. Deleting the file forces a new calibration.
// ---------------------------------------------------------------------------------------------------------------------
static const char* const CALIB_FILE = "fft_calibration.txt";

static std::string calib_env_sig() {   // the variables that change the calibration's choice
    std::string s;
    for (const char* v : { "LOTR_DIGIT_BITS", "LOTR_NO_BALANCED", "LOTR_NO_IBDWT", "LOTR_IB_MAXERR",
                           "LOTR_IB_MINGAIN", "LOTR_IB_SAFEERR" })
        if (const char* e = std::getenv(v)) s += std::string(s.empty() ? "" : ",") + v + "=" + e;
    return s.empty() ? "-" : s;
}

static unsigned bit_len(uint64_t k) { return k ? 64u - (unsigned)__builtin_clzll(k) : 0u; }

static bool calib_load(unsigned n, uint64_t k, size_t* len, unsigned* bits, bool* ib) {
    std::ifstream f(CALIB_FILE);
    std::string line, sig = calib_env_sig();
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        unsigned cn = 0, cb = 0;
        uint64_t ck = 0;
        size_t cl = 0;
        int cib = 0;
        std::string cs;
        if (!(ss >> cn >> ck >> cl >> cb >> cib >> cs)) continue;
        if (cn != n || ck < k || bit_len(ck) != bit_len(k) || cs != sig || !fft_len_supported(cl)) continue;
        *len = cl;
        *bits = cb;
        *ib = (cib != 0);
        return true;
    }
    return false;
}

static void calib_save(unsigned n, uint64_t k, size_t len, unsigned bits, bool ib) {
    std::ofstream f(CALIB_FILE, std::ios::app);
    if (f) f << n << " " << k << " " << len << " " << bits << " " << (ib ? 1 : 0) << " " << calib_env_sig() << "\n";
}

size_t auto_tune_fft_len(unsigned actual_n, uint64_t sample_k, unsigned* digit_bits_out) {
    // LOTR_FORCE_FFT_LEN sets the length without a calibration - nothing to save
    if (std::getenv("LOTR_FORCE_FFT_LEN") != nullptr) return auto_tune_uncached(actual_n, sample_k, digit_bits_out);
    size_t len = 0;
    unsigned bits = 16;
    bool ib = false;
    if (calib_load(actual_n, sample_k, &len, &bits, &ib)) {
        g_ibdwt_mode = ib;
        if (digit_bits_out) *digit_bits_out = bits;
        std::cout << " Calibration for N=" << actual_n << " read from " << CALIB_FILE << ": FFT " << len
                  << (ib ? " (IBDWT)" : " (" + std::to_string(bits) + "-bit digits)") << " - delete the file to calibrate again\n";
        return len;
    }
    const auto t0 = std::chrono::steady_clock::now();
    len = auto_tune_uncached(actual_n, sample_k, &bits);
    if (digit_bits_out) *digit_bits_out = bits;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (secs >= 30.0) {
        calib_save(actual_n, sample_k, len, bits, g_ibdwt_mode);
        std::cout << " Calibration took " << (int)(secs / 60) << " min " << (int)secs % 60 << " s - saved to " << CALIB_FILE
                  << " (the next run with this N and k uses it)\n";
    }
    return len;
}

// FFT length for the test: the classic choice (zero-padded real-FFT square, see auto_tune_classic), then the IBDWT
// on the shorter lengths, smallest first. The IBDWT holds (n + log2 k) bits in 2*len digits without zero padding,
// but loses about log2(k)/2 bits per digit of precision, so it wins for small k (e.g. 91*2^5435752+1: 3*2^16
// instead of 3*2^17, 1.78x; 7*2^20267500: 3*2^18 instead of 3*2^19, 1.82x) and is rejected for large k.
// An IBDWT length is taken only if its GMP self-test passes and the error after 100 squarings is <= 0.3.
// Safe because a run whose error grows past 0.45 (or whose Gerbicz-Li check fails twice in a row) continues on the
// next longer IBDWT length (proth_logic.cpp). LOTR_IB_MAXERR=<x> changes the limit; LOTR_NO_IBDWT=1 = classic only.
static size_t auto_tune_uncached(unsigned actual_n, uint64_t sample_k, unsigned* digit_bits_out) {
    g_ibdwt_mode = false;
    unsigned bits = 16;
    const size_t len = auto_tune_classic(actual_n, sample_k, &bits);
    if (digit_bits_out) *digit_bits_out = bits;
    if (g_ibdwt_mode) return len;   // forced (LOTR_FORCE_FFT_LEN + LOTR_IBDWT)
    if (std::getenv("LOTR_FORCE_FFT_LEN") != nullptr || std::getenv("LOTR_NO_IBDWT") != nullptr) return len;
    if (!fft_uses_scaled_form() || std::getenv("LOTR_NO_BALANCED") != nullptr) return len;

    const double l2k = std::log2((double)(sample_k | 1));
    double ib_maxerr = 0.3;
    if (const char* e = std::getenv("LOTR_IB_MAXERR")) ib_maxerr = std::strtod(e, nullptr);
    // The IBDWT only where it is clearly shorter than the classic length: on a similar length it costs more (weights,
    // twist, normalization) and runs closer to its error limit (100029*2^4190269+1: IBDWT 3*2^16 .205 ms/iter with Max Err
    // up to 0.46 in the run vs classic 7*2^15 (19-bit) .204 with 0.078). Default factor 1.2 (the classic length at least
    // 1.2x the IBDWT one); LOTR_IB_MINGAIN=<x> changes it.
    double ib_min_gain = 1.2;
    if (const char* e = std::getenv("LOTR_IB_MINGAIN")) ib_min_gain = std::strtod(e, nullptr);
    // ... but only where the IBDWT runs close to its error limit: with a small calibration error the shorter IBDWT wins
    // even at a similar length (4715*2^1027189: IBDWT 3*2^14, 1.17x shorter than classic 7*2^13, err 0.004 -> 0.05 vs
    // 0.07 ms/iter on 6 workers). An IBDWT length shorter than the classic one is taken when its calibration error is
    // <= LOTR_IB_SAFEERR (default 0.05: the 100 calibration squarings of a simple number see less than a run), otherwise
    // only if it is ib_min_gain times shorter.
    double ib_safe_err = 0.05;
    if (const char* e = std::getenv("LOTR_IB_SAFEERR")) ib_safe_err = std::strtod(e, nullptr);
    // Summary: one line of the IBDWT test and, when the IBDWT is not used, the PASS of the classic length; tagged [CPU ...]
    // so that next to the [GPU] lines it is clear which engine reports it.
    std::string ib_tried;
    auto classic_kept = [&](const char* why) {
        if (!ib_tried.empty())
            std::cout << " [CPU IBDWT] k=" << sample_k << " (" << bit_len(sample_k) << " bits):" << ib_tried << " - " << why << "\n"
                      << " [CPU CALIBRATION] classic FFT " << len << " (" << bits << "-bit digits): PASS\n";
        return len;
    };
    for (size_t L = 256; L < len; L *= 2) {
        for (const size_t c : { L, L + L / 2 }) {
            if (c >= len || !fft_ibdwt_supported(c)) continue;
            const bool need_low_err = ((double)c * ib_min_gain > (double)len);   // not clearly shorter
            const double bpd = ((double)actual_n + l2k) / (2.0 * (double)c);
            if (bpd > 20.0) continue;
            const unsigned ib_bits = (unsigned)(((size_t)actual_n + 192 + c - 1) / c);   // limb count only
            if (g_detail) std::cout << "  -> IBDWT test for FFT_Len: " << c << " (" << 2 * c << " digits of ~" << bpd << " bits)...\n";
            bool st_ok = false;
            const double err = calibrate_len(c, ib_bits, actual_n, sample_k, &st_ok, true);
            if (g_detail) std::cout << "  Max Err after 100 iterations = " << err << (st_ok ? "" : " (self-test FAILED)") << "\n";
            std::ostringstream t;
            t << " FFT " << c;
            if (!st_ok) t << " self-test vs GMP failed";
            else if (err < 0.0) t << " not available";
            else t << " Max Err " << std::setprecision(3) << err;
            if (st_ok && err >= 0.0 && err <= ib_maxerr && need_low_err && err > ib_safe_err) {
                if (g_detail) std::cout << "  IBDWT on FFT " << c << " only " << (double)len / (double)c << "x shorter than the classic "
                                        << len << " and Max Err " << err << " > " << ib_safe_err << " - classic kept\n";
                ib_tried += t.str();
                return classic_kept("not clearly shorter than the classic FFT");   // longer IBDWT lengths are not shorter by more
            }
            if (st_ok && err >= 0.0 && err <= ib_maxerr) {
                if (g_detail) std::cout << "  FFT Pass: " << c << " with IBDWT\n";
                std::ostringstream p;   // (not on std::cout: setprecision would stay for all later output)
                p << " [CPU IBDWT] k=" << sample_k << " (" << bit_len(sample_k) << " bits): FFT " << c << " (" << 2 * c
                  << " digits of ~" << std::setprecision(4) << bpd << " bits), self-test OK, Max Err "
                  << std::setprecision(3) << err << ": PASS - used instead of the classic FFT " << len << "\n";
                std::cout << p.str();
                g_ibdwt_mode = true;
                if (digit_bits_out) *digit_bits_out = ib_bits;
                return c;
            }
            if (g_detail) std::cout << "  IBDWT on FFT " << c << " rejected - trying a longer FFT\n";
            ib_tried += t.str() + ",";
        }
    }
    if (!ib_tried.empty() && ib_tried.back() == ',') ib_tried.pop_back();
    return classic_kept("not used");
}

static size_t auto_tune_classic(unsigned actual_n, uint64_t sample_k, unsigned* digit_bits_out) {
    if (digit_bits_out) *digit_bits_out = 16;
    if (g_detail) std::cout << "\n Calibration for N=" << actual_n << "...\n";

    // Testing: LOTR_FORCE_FFT_LEN=<2^p or 3*2^p> forces the FFT length (16-bit digits, e.g. a small N on a large FFT), so
    // every FFT path can be checked against a known RES64 in seconds. The result does not depend on the FFT length.
    if (const char* fe = std::getenv("LOTR_FORCE_FFT_LEN")) {
        const size_t forced = (size_t)std::strtoull(fe, nullptr, 10);
        // LOTR_IBDWT=1: the IBDWT square on this length (2*len digits of ~(n + log2 k)/(2 len) bits, no zero padding).
        // digit_bits then only sets the limb count: fft_len*digit_bits/64 >= (n + 192)/64.
        if (std::getenv("LOTR_IBDWT") != nullptr && fft_len_supported(forced) && fft_uses_scaled_form()) {
            const double bpd = ((double)actual_n + 64.0) / (2.0 * (double)forced);
            if (bpd <= 24.0) {
                g_ibdwt_mode = true;
                if (digit_bits_out) *digit_bits_out = (unsigned)(((size_t)actual_n + 192 + forced - 1) / forced);
                std::cout << " Selected (LOTR_FORCE_FFT_LEN, IBDWT): " << forced << " (" << 2 * forced << " digits of ~"
                          << bpd << " bits)\n\n";
                return forced;
            }
            std::cout << " [WARN] LOTR_IBDWT: " << bpd << " bits per digit on FFT " << forced << " - too many, ignored\n";
        }
        if (fft_len_supported(forced) && (double)forced * 16.0 >= (double)actual_n + 64.0) {
            std::cout << " Selected (LOTR_FORCE_FFT_LEN): " << forced << "\n\n";
            return forced;
        }
        std::cout << " [WARN] LOTR_FORCE_FFT_LEN=" << fe << " ignored (must be 2^p or 3*2^p >= 256 and >= N/16)\n";
    }
    omp_set_num_threads(1);
    size_t n_limbs = (actual_n + 63) / 64;

    // 16-bit digits hold N = k*2^n+1 only with room for the bits of k (len*16 >= n is not enough: a 45-bit k at n just
    // below 16*len does not fit); + 64 as for the wide digits below and LOTR_FORCE_FFT_LEN above
    size_t candidate_len = 256;
    while (candidate_len * 16 < (size_t)actual_n + 64) {
        candidate_len = get_next_smooth_size(candidate_len);
    }

    // SHORTER FFT LENGTHS than the 16-bit power of two above, smallest first:
    //  - 3*2^p lengths (radix-3; e.g. N = 3.33M on 3*2^16 instead of 2^18, ~1.3x faster), with 16..max-bit digits,
    //  - powers of two with wide digits B = 17..max bits (e.g. N = 4.39M: 2^18 x 17 bits instead of 2^19).
    // Digit width B = the smallest that fits N plus 64 bits of room (top of k*2^n+1, balanced-digit offset).
    // A length is taken only if the GMP self-test passes and the rounding error stays well below the limit (0.2 instead
    // of 0.45: margin for numbers the calibration does not cover). Off with the old digits / old square.
    const unsigned max_bits = wide_digits_max();
    const bool alt_allowed = (std::getenv("LOTR_NO_BALANCED") == nullptr);
    if (alt_allowed) {
        std::vector<size_t> alt;
        for (size_t L = 256; L < candidate_len; L *= 2) {
            alt.push_back(L);
            if (L + L / 2 < candidate_len) alt.push_back(L + L / 2);       // 3*2^(p-1), between 2^p and 2^(p+1)
            if (L + 3 * L / 4 < candidate_len) alt.push_back(L + 3 * L / 4);   // 7*2^(p-2) (radix-7)
        }
        for (size_t L : alt) {
            const bool pow2 = (L & (L - 1)) == 0;
            unsigned b = (unsigned)(((size_t)actual_n + 64 + L - 1) / L);
            if (b < 16) b = 16;
            if (b > max_bits) continue;
            if (pow2 && b == 16) continue;          // cannot happen below candidate_len; the classic path below covers it
            if (!fft_len_supported(L)) continue;
            if (g_detail) std::cout << "  -> Test for FFT_Len: " << L << " (" << b << "-bit digits)...\n";
            bool st_ok = false;
            const double err = calibrate_len(L, b, actual_n, sample_k, &st_ok);
            if (g_detail) std::cout << "  Max Err after 100 iterations = " << err << (st_ok ? "" : " (self-test FAILED)") << "\n";
            if (st_ok && err >= 0.0 && err <= 0.2) {
                if (g_detail) std::cout << "  FFT Pass: " << L << " with " << b << "-bit digits\n";
                if (digit_bits_out) *digit_bits_out = b;
                return L;
            }
            std::cout << "  [INFO] FFT " << L << " with " << b << "-bit digits rejected (max err " << err
                      << (st_ok ? "" : ", self-test failed") << ") - trying a longer FFT\n";
        }
    }

    // ==========================================
    // PRECISION LIMITS (bits per FFT word)
    // ==========================================
    double max_allowed_bits = 17.5;
    double auto_accept_bits = 11.0;

    // Balanced digits (pack/unpack) reduce the rounding error by two orders of magnitude, so the old N-dependent limits
    // (13.5 bits/word for N > 6M, 9.5 for N > 15M) apply only to the old digits (LOTR_NO_BALANCED=1).
    // The FFT length follows from 16-bit chunks (len*16 >= N); correctness is guarded by Max Err and the Gerbicz-Li check.
    const bool legacy_digits = (std::getenv("LOTR_NO_BALANCED") != nullptr);
    if (legacy_digits && actual_n > 15000000) {
        max_allowed_bits = 9.5;
        auto_accept_bits = 6.0;
    }
    else if (legacy_digits && actual_n > 6000000) {
        max_allowed_bits = 13.5;
        auto_accept_bits = 9.0;
    }
    // ==========================================

    while (true) {
        double bits_per_word = (double)actual_n / (double)candidate_len;

        if (bits_per_word < auto_accept_bits) {
            if (g_detail) std::cout << " Selected: " << candidate_len << "\n\n";
            return candidate_len;
        }

        if (bits_per_word > max_allowed_bits) {
            candidate_len = get_next_smooth_size(candidate_len);
            continue;
        }

        if (g_detail) std::cout << "  -> Karatsuba tests for FFT_Len: " << candidate_len << " (Bits/word: " << bits_per_word << ")...\n";

        FFTContext test_ctx;
        if (!fft_init(test_ctx, FFTBackend::INTERNAL, candidate_len)) {
            candidate_len = get_next_smooth_size(candidate_len);
            continue;
        }

        size_t safe_alloc_limbs = std::max(candidate_len, (size_t)((2 * actual_n) / 64 + 64));
        size_t limbs_bytes_size = safe_alloc_limbs * sizeof(uint64_t);
        uint64_t* test_limbs = (uint64_t*)mmap(NULL, limbs_bytes_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (test_limbs != MAP_FAILED) {
            std::memset(test_limbs, 0, limbs_bytes_size);
            for (size_t i = 0; i < n_limbs; i++) {
                test_limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
            }
            test_limbs[0] = sample_k | 1;

            for (int step = 0; step < 100; ++step) {
                fft_square_karatsuba(test_ctx, test_limbs, n_limbs, actual_n, sample_k);
            }

            double max_err_observed = test_ctx.last_max_diff;
            munmap(test_limbs, limbs_bytes_size);
            fft_destroy(test_ctx);
            if (g_detail) std::cout << "  Max Err after 10 iterations = " << max_err_observed << "\n";

            if (max_err_observed >= 0.0 && max_err_observed <= 0.45) {
                if (g_detail) std::cout << "  FFT Pass: " << candidate_len << "\n";
                return candidate_len;
            }
            else {
                std::cout << "     [Fail] The error is too high. Shifting up a gear...\n";
            }
        }
        else {
            // Free the FFT context also when mmap fails (no leak)
            fft_destroy(test_ctx);
        }

        size_t next_len = get_next_smooth_size(candidate_len);

        if (next_len <= candidate_len) {
            next_len = candidate_len * 2;
        }

        candidate_len = next_len;
    }
}


void dispatch(const std::vector<Task>& tasks, int max_concurrent_tasks, int threads_per_task, bool pin, bool gpu,
              bool gpu_only) {
    if (tasks.empty()) return;
    if (max_concurrent_tasks <= 0) max_concurrent_tasks = 1;
    if (threads_per_task <= 0) threads_per_task = 1;

    // The calibration uses the largest N of all tasks: an N range or a file with different N runs every candidate on one
    // FFT length, and a length calibrated for a smaller first N cannot hold the larger ones. A longer length holds the
    // smaller N too.
    unsigned actual_n = tasks[0].n;
    for (const Task& t : tasks) actual_n = std::max(actual_n, t.n);
    // The calibration uses the largest k of all tasks: the IBDWT loses precision with k (about log2(k)/2 bits per
    // digit), so a length that passes for the largest k passes for all of them.
    uint64_t sample_k = 3;
    for (const Task& t : tasks) {
        if (t.type == TaskType::PROTH_SINGLE) sample_k = std::max(sample_k, t.k);
        else if (t.type == TaskType::PROTH_BATCH) {
            for (uint64_t kk : t.k_list) sample_k = std::max(sample_k, kk);
        }
    }

    unsigned digit_bits = 16;
    size_t needed_fft_len = 0;
    if (!gpu_only) {   // --gpu-only: no CPU worker, so no CPU calibration either (an hour for the giants)
        needed_fft_len = auto_tune_fft_len(actual_n, sample_k, &digit_bits);
        FFTFactorPlan fft_plan = generate_fft_plan(needed_fft_len, true);
        if (fft_plan.has_unsupported_primes) {
            std::cout << "[MIXED-RADIX] Note: Factors other than 2, 3, 5 detected. Bluestein support required.\n";
        }
    }

    struct Item { size_t idx; Task task; bool cpu_only = false; };   // cpu_only: --gpu has no plan for it
    std::deque<Item> q;
    std::mutex m;
    std::condition_variable cv;
    bool enqueue_done = false;
    int gpu_busy = 0;   // --gpu: candidates the card is testing (it may still hand them back to the CPU workers)
    // --gpu, the tail of a batch: when the card is much faster than a CPU worker, it would stand idle for hours after the
    // queue while the CPU finishes its last candidates. A card worker with nothing left in the queue takes over the CPU
    // candidate furthest from its end (proth_logic.h cpu_run_request: the CPU hands it over after its next checkpoint, the
    // card resumes from it). Only with more than 3 segments of Gerbicz-Li (3 x 65536 squarings) left.
    // LOTR_GPU_TAKEOVER=0 turns it off.
    struct Run { bool active = false, taken = false; Item it; };
    std::vector<Run> runs((size_t)max_concurrent_tasks);
    std::deque<Item> handed;   // candidates a CPU worker handed over, waiting for the card worker that asked
    const char* tko = std::getenv("LOTR_GPU_TAKEOVER");
    const bool takeover = !(tko && *tko == '0');

    // --gpu: the card is one more worker on this queue (several for small transforms, gpu_streams: more candidates at
    // once fill the card better). The queue then holds single candidates: a batch of the CPU split would leave the faster
    // card idle at the end (or a CPU worker busy long after the rest). Without --gpu the queue holds the CPU split.
    std::vector<Task> gpu_items;
    const std::vector<Task>* items = &tasks;
    int n_gpu = 0;
    if (gpu) {
        std::string why;
        if (!gpu_probe(0, why)) {
            if (gpu_only) {
                std::cout << "[GPU] " << why << " - --gpu-only: nothing tested\n";
                return;
            }
            std::cout << "[GPU] " << why << " - testing on the CPU only\n";
            gpu = false;
        }
        else {
            for (const Task& t : tasks) {
                if (t.type != TaskType::PROTH_BATCH) { gpu_items.push_back(t); continue; }
                for (uint64_t kk : t.k_list) {
                    Task s;
                    s.type = TaskType::PROTH_SINGLE;
                    s.n = t.n;
                    s.k = kk;
                    s.report_step = t.report_step;
                    gpu_items.push_back(std::move(s));
                }
            }
            items = &gpu_items;
            // measured: a quick test of 1, 2, ... at once before a batch (gpu_streams_auto; --gpu-only: cli asked first with the
            // same n, k and this takes its kept result)
            n_gpu = (int)std::min((size_t)gpu_streams_auto(actual_n, sample_k, gpu_items.size()), gpu_items.size());
            if (n_gpu < 1) n_gpu = 1;
            if (gpu_only) std::cout << "[GPU] " << why << ": GPU only, " << n_gpu << " candidate(s) at once on the card\n";
            else if (n_gpu == 1) std::cout << "[GPU] " << why << ": one more worker on the queue\n";
            else std::cout << "[GPU] " << why << ": " << n_gpu << " more workers on the queue (" << n_gpu
                           << " candidates at once on the card)\n";
        }
    }

    std::vector<std::string> results(items->size());
    {
        std::lock_guard<std::mutex> lk(m);
        for (size_t i = 0; i < items->size(); ++i) {
            Task t = (*items)[i];
            q.push_back(Item{ i, t, false });
        }
        enqueue_done = true;
    }
    cv.notify_all();

    // --pin: every worker and its OpenMP threads on their own CPUs, the layout --tune measures (a worker of
    // T | cores-per-CCD threads stays in one CCD). Without it the OS places the threads.
    std::vector<std::vector<int>> pin_layout;
    if (pin) {
        const int w = (int)std::min((size_t)max_concurrent_tasks, items->size());
        pin_layout = cpu_layout(cpu_topology(), w, threads_per_task);
        if ((int)pin_layout.size() < w) {
            std::cout << "[PIN] " << w << " x " << threads_per_task << " threads do not fit the CPUs of this process - "
                      << "no pinning\n";
            pin_layout.clear();
        }
        else {
            std::vector<int> all_cpus;
            for (const auto& l : pin_layout) all_cpus.insert(all_cpus.end(), l.begin(), l.end());
            std::sort(all_cpus.begin(), all_cpus.end());
            std::cout << "[PIN] " << pin_layout.size() << " worker(s) x " << threads_per_task << " thread(s) pinned to CPUs "
                      << cpu_list_str(all_cpus) << "\n";
            if (g_detail)   // the CPUs of every worker only with --detail
                for (size_t i = 0; i < pin_layout.size(); ++i)
                    std::cout << "[PIN]   worker " << i << ": CPUs " << cpu_list_str(pin_layout[i]) << "\n";
        }
    }

    auto worker_fn = [&](int worker_id) {

        omp_set_num_threads(threads_per_task);

        if (!pin_layout.empty() && worker_id < (int)pin_layout.size()) {
            pin_team(pin_layout[(size_t)worker_id]);
        }
        // Without --pin: hard pinning only for a single 1-thread worker. With OpenMP threads (threads_per_task > 1) or
        // many 1-thread workers the OS scheduler places the threads: pinning a worker to one CPU would put all of its
        // OpenMP threads on the same single core.
        else if (threads_per_task == 1 && max_concurrent_tasks == 1) {
            // single 1-thread worker: pin to the first allowed CPU (taskset), not to CPU number worker_id
            const int cpu = nth_process_cpu(worker_id);
            if (cpu >= 0) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cpu, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            }
        }
        else if (g_proc_cpus_ok) {
            // other modes: the worker (and its OpenMP threads) may run on all CPUs of the process
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &g_proc_cpus);
        }

        FFTContext local_ctx;
        if (!fft_init(local_ctx, FFTBackend::INTERNAL, needed_fft_len)) {
            std::lock_guard<std::mutex> lk(m);
            std::cerr << "[ERROR] FFT init failed for len " << needed_fft_len << std::endl;
            return;
        }
        local_ctx.digit_bits = digit_bits;
        local_ctx.ibdwt = g_ibdwt_mode;

        size_t safe_alloc_limbs = std::max(needed_fft_len, (size_t)((2 * actual_n) / 64 + 64));
        size_t limbs_bytes_size = safe_alloc_limbs * sizeof(uint64_t);

        // Allocated ONCE before the task loop (no repeated mmap/munmap system calls)
        uint64_t* local_limbs = (uint64_t*)mmap(NULL, limbs_bytes_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (local_limbs == MAP_FAILED) {
            std::lock_guard<std::mutex> lk(m);
            std::cerr << "[ERROR] mmap failed for local_limbs" << std::endl;
            fft_destroy(local_ctx);
            return;
        }

        for (;;) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(m);
                // (--gpu: while the card tests, it may still hand its candidate back - wait for it)
                cv.wait(lk, [&] { return !q.empty() || (enqueue_done && gpu_busy == 0); });

                if (q.empty()) {
                    munmap(local_limbs, limbs_bytes_size);
                    fft_destroy(local_ctx);
                    return;
                }
                it = q.front();
                q.pop_front();
                runs[(size_t)worker_id].active = true;
                runs[(size_t)worker_id].taken = false;
                runs[(size_t)worker_id].it = it;
            }

            // The previous test may have moved to a longer IBDWT FFT (round-off safety net in proth_logic.cpp): back to the
            // calibrated context for the next candidate
            if (local_ctx.fft_len != needed_fft_len || local_ctx.ibdwt != g_ibdwt_mode) {
                fft_destroy(local_ctx);
                local_ctx = FFTContext();
                if (!fft_init(local_ctx, FFTBackend::INTERNAL, needed_fft_len)) {
                    std::lock_guard<std::mutex> lk(m);
                    std::cerr << "[ERROR] FFT init failed for len " << needed_fft_len << std::endl;
                    munmap(local_limbs, limbs_bytes_size);
                    return;
                }
                local_ctx.digit_bits = digit_bits;
                local_ctx.ibdwt = g_ibdwt_mode;
            }

            // Clear only the part that is used before the next task
            size_t limbs_to_clear = (2 * actual_n) / 64 + 10;
            std::memset(local_limbs, 0, limbs_to_clear * sizeof(uint64_t));

            std::string r = run_task_with_ctx(it.task, local_ctx, local_limbs);
            {
                std::lock_guard<std::mutex> lk(m);
                runs[(size_t)worker_id].active = false;
                if (it.task.type == TaskType::PROTH_SINGLE) cpu_run_clear(it.task.k, it.task.n);
                if (r == "handed") handed.push_back(it);   // to the card worker waiting for it
                else results[it.idx] = std::move(r);
            }

            cv.notify_all();
        }
        };


    // --gpu: the card's workers (n_gpu, each its own engine and stream). They take their first candidates before the
    // CPU workers start (a single candidate goes to the card). On a failure (-1: CUDA error, repeated Gerbicz-Li
    // failures) the candidate goes back to the front of the queue for the CPU workers and the card stops (the other card
    // worker finishes its candidate); a candidate without a plan for the card (-2) goes back marked cpu_only and the
    // card takes the next one.
    int gpu_started = 0;
    bool gpu_stop = false;
    auto gpu_fn = [&]() {
        bool first = true;
        for (;;) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(m);
                auto pos = gpu_stop ? q.end() : std::find_if(q.begin(), q.end(), [](const Item& x) { return !x.cpu_only; });
                if (first) { ++gpu_started; first = false; }
                if (pos != q.end()) {
                    it = *pos;
                    q.erase(pos);
                    ++gpu_busy;
                }
                else {
                    // the queue is empty for the card: take over the CPU candidate with the most squarings left
                    int w = -1;
                    unsigned long long best_left = 3ull * 65536;
                    for (size_t i = 0; takeover && !gpu_stop && i < runs.size(); ++i) {
                        const Run& r = runs[i];
                        if (!r.active || r.taken || r.it.cpu_only || r.it.task.type != TaskType::PROTH_SINGLE) continue;
                        const unsigned long long done = cpu_run_iter(r.it.task.k, r.it.task.n);
                        const unsigned long long all = r.it.task.n > 1 ? r.it.task.n - 1 : 0;
                        const unsigned long long left = all > done ? all - done : 0;
                        if (left > best_left) { best_left = left; w = (int)i; }
                    }
                    if (w < 0) {
                        lk.unlock();
                        cv.notify_all();
                        return;
                    }
                    Run& r = runs[(size_t)w];
                    r.taken = true;
                    const size_t want = r.it.idx;
                    const Task tk = r.it.task;
                    cpu_run_request(tk.k, tk.n);
                    ++gpu_busy;   // the CPU workers wait for the card (also the one that hands over)
                    {
                        std::ostringstream ms;
                        ms << std::fixed << std::setprecision(1) << "[GPU] the card is free: it takes over " << tk.k << "*2^"
                           << tk.n << "+1 from the CPU at "
                           << 100.0 * (double)(tk.n - 1 - best_left) / (double)(tk.n > 1 ? tk.n - 1 : 1)
                           << "% (after its next checkpoint)";
                        std::lock_guard<std::mutex> io(g_io_mutex);
                        std::cout << (g_stdout_tty ? "\r\033[K" : "") << ms.str() << std::endl;
                    }
                    cv.wait(lk, [&] { return !(runs[(size_t)w].active && runs[(size_t)w].it.idx == want); });
                    const auto h = std::find_if(handed.begin(), handed.end(), [&](const Item& x) { return x.idx == want; });
                    if (h == handed.end()) {   // the CPU finished it first
                        --gpu_busy;
                        lk.unlock();
                        cv.notify_all();
                        continue;
                    }
                    it = *h;
                    handed.erase(h);
                }
            }
            cv.notify_all();
            const int r = execute_proth_test_gpu(it.task.k, it.task.n, 0, it.task.report_step);
            {
                std::unique_lock<std::mutex> lk(m);
                --gpu_busy;
                if (r < 0) {
                    it.cpu_only = true;
                    q.push_front(std::move(it));
                    if (r == -1 && !gpu_stop)
                        std::cout << "[GPU] the card stops testing - the remaining candidates "
                                  << (gpu_only ? "are not tested (--gpu-only)\n" : "on the CPU\n");
                    if (r == -1) gpu_stop = true;
                    lk.unlock();
                    cv.notify_all();
                    if (r == -1) return;
                    continue;
                }
                results[it.idx] = (r == 1) ? "prime" : "PRP";
            }
            batch_note_done(r == 1, it.task.n);
        }
    };
    std::vector<std::thread> gpu_threads;
    if (gpu) {
        for (int i = 0; i < n_gpu; ++i) gpu_threads.emplace_back(gpu_fn);
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return gpu_started == n_gpu; });
    }

    std::vector<std::thread> pool;
    int active_workers = gpu_only ? 0 : (int)std::min((size_t)max_concurrent_tasks, items->size());

    pool.reserve((size_t)active_workers);
    for (int i = 0; i < active_workers; ++i) {
        pool.emplace_back(worker_fn, i);
    }

    for (auto& th : pool) th.join();
    for (auto& th : gpu_threads) th.join();
    if (gpu_only && !q.empty())   // no plan for the card, or the card stopped (checkpoints kept)
        std::cout << "[GPU] --gpu-only: " << q.size() << " candidate(s) not tested (run them without --gpu-only"
#ifdef LOTR_GPU_ONLY
                  << ", in the full build: ./build.sh on an AVX-512 CPU"
#endif
                  << ")\n";
}

// ---------------------------------------------------------------------------------------------------------------------
// --tune: which split of the CPUs tests a batch fastest - W candidates at once x T threads each. Every configuration
// runs W real contexts of the engine (the calibrated FFT length, IBDWT or classic) pinned as --pin pins them, squaring
// a number of this size for 'secs' seconds; nothing is tested for real (no checkpoints, no result files).
// Reports ms/iter, time per candidate, candidates per hour and the time of the whole batch, and the best options.
// ---------------------------------------------------------------------------------------------------------------------
static std::string fmt_duration(double s) {
    char buf[64];
    if (s < 90) std::snprintf(buf, sizeof buf, "%.0f s", s);
    else if (s < 5400) std::snprintf(buf, sizeof buf, "%.0f min", s / 60);
    else if (s < 2 * 86400) std::snprintf(buf, sizeof buf, "%.1f h", s / 3600);
    else std::snprintf(buf, sizeof buf, "%.1f days", s / 86400);
    return buf;
}

// W workers on 'lay' squaring for 'secs' seconds; returns the ms per iteration of each worker (0 = failed).
static std::vector<double> tune_measure(size_t len, unsigned digit_bits, bool ib, unsigned n, uint64_t k,
                                        const std::vector<std::vector<int>>& lay, double secs) {
    const int W = (int)lay.size();
    std::atomic<int> ready{ 0 };
    std::atomic<bool> go{ false }, stop{ false };
    std::vector<double> ms((size_t)W, 0.0);
    std::vector<std::thread> th;
    for (int w = 0; w < W; ++w) {
        th.emplace_back([&, w]() {
            pin_team(lay[(size_t)w]);
            FFTContext ctx;
            uint64_t* v = nullptr;
            const size_t alloc = std::max(len, (size_t)(2 * (size_t)n / 64 + 64));
            bool ok = fft_init(ctx, FFTBackend::INTERNAL, len);
            if (ok) {
                ctx.digit_bits = digit_bits;
                ctx.ibdwt = ib;
                v = (uint64_t*)mmap(NULL, alloc * sizeof(uint64_t), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                ok = (v != MAP_FAILED);
                if (!ok) v = nullptr;
            }
            const size_t nl = len * digit_bits / 64;   // as proth_logic.cpp (n_limbs_safe)
            int ch = 1;
            if (ok) {
                std::memset(v, 0, alloc * sizeof(uint64_t));
                uint64_t st = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)w << 32);
                for (size_t i = 0; i < n / 64; ++i) {   // pseudo-random value < 2^n (splitmix64)
                    uint64_t z = (st += 0x9E3779B97F4A7C15ULL);
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                    v[i] = z ^ (z >> 31);
                }
                // warm-up (the first square runs the engine's self-test) and a chunk of ~0.1 s between stop checks
                const auto a0 = std::chrono::steady_clock::now();
                fft_square_repeat(ctx, v, nl, n, k, 8);
                const double per = std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count() / 8;
                ch = (int)std::max(1.0, std::min(256.0, 0.1 / std::max(per, 1e-7)));
                ok = (ctx.last_max_diff < 0.45);
            }
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (ok) {
                const auto t0 = std::chrono::steady_clock::now();
                uint64_t it = 0;
                while (!stop.load()) {
                    fft_square_repeat(ctx, v, nl, n, k, (unsigned)ch);
                    it += (uint64_t)ch;
                }
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                ms[(size_t)w] = (it > 0 && ctx.last_max_diff < 0.45) ? 1000.0 * el / (double)it : 0.0;
            }
            if (v) munmap(v, alloc * sizeof(uint64_t));
            fft_destroy(ctx);
        });
    }
    while (ready.load() < W) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    go.store(true);
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop.store(true);
    for (auto& t : th) t.join();
    return ms;
}

int tune_configs(const std::vector<Task>& tasks, double secs, bool all) {
    if (tasks.empty()) return 0;
    const unsigned n = tasks[0].n;
    uint64_t kmax = 3;
    size_t cands = 0;
    for (const Task& t : tasks) {
        if (t.type == TaskType::PROTH_SINGLE) { kmax = std::max(kmax, t.k); ++cands; }
        else { for (uint64_t kk : t.k_list) kmax = std::max(kmax, kk); cands += t.k_list.size(); }
    }
    const std::vector<CpuGroup> topo = cpu_topology();
    size_t P = 0, C = ~(size_t)0;
    bool smt = false;
    for (const CpuGroup& g : topo) { P += g.prim.size(); C = std::min(C, g.prim.size()); smt = smt || !g.sec.empty(); }
    const int D = (int)topo.size();
    std::cout << "\n[TUNE] N=" << n << ", " << cands << " candidate(s), largest k=" << kmax << "\n[TUNE] CPUs of this process: "
              << D << " group(s) (L3 / CCD), " << P << " physical cores" << (smt ? " + SMT siblings" : "") << "\n";
    for (int g = 0; g < D; ++g)
        std::cout << "[TUNE]   CCD " << g << ": cores " << cpu_list_str(topo[(size_t)g].prim)
                  << (topo[(size_t)g].sec.empty() ? "" : ", SMT " + cpu_list_str(topo[(size_t)g].sec)) << "\n";

    unsigned digit_bits = 16;
    const size_t len = auto_tune_fft_len(n, kmax, &digit_bits);
    const bool ib = g_ibdwt_mode;
    std::cout << "[TUNE] FFT length " << len << (ib ? " (IBDWT)" : "") << ", " << secs << " s per configuration\n\n";

    std::vector<std::pair<int, int>> cfgs;   // (workers, threads per worker)
    auto add = [&](int W, int T) {
        if (W < 1 || T < 1) return;
        for (auto& c : cfgs) if (c.first == W && c.second == T) return;
        cfgs.push_back({ W, T });
    };
    add(1, (int)P);
    if (D > 1) add(D, (int)C);
    for (int T : { 6, 4, 3, 2, 1 })
        if ((size_t)T < C && C % (size_t)T == 0) add(D * (int)(C / (size_t)T), T);
    if (smt) { add(D, 2 * (int)C); add(2 * (int)P, 1); }
    // --tune-all: also with cores left free (the tests compete for L3 / RAM - fewer may give more), evenly per CCD:
    // m workers of T threads per CCD for every T <= cores per CCD and m*T <= cores per CCD
    if (all)
        for (int T = 1; T <= (int)C; ++T)
            for (int m = 1; m * T <= (int)C; ++m) add(D * m, T);

    struct Res { int W, T; double ms, cph; };
    std::vector<Res> res;
    for (auto [W, T] : cfgs) {
        const auto lay = cpu_layout(topo, W, T);
        if ((int)lay.size() < W) continue;
        std::cout << "[TUNE] " << std::setw(2) << W << " x " << std::setw(2) << T << " ..." << std::flush;
        const std::vector<double> ms = tune_measure(len, digit_bits, ib, n, kmax, lay, secs);
        double sum_ms = 0, cph = 0;
        int ok = 0;
        for (double m : ms) if (m > 0) { sum_ms += m; cph += 3600.0 / ((double)n * m / 1000.0); ++ok; }
        if (ok < W) { std::cout << " failed\n"; continue; }
        res.push_back({ W, T, sum_ms / ok, cph });
        std::cout << " " << std::fixed << std::setprecision(3) << sum_ms / ok << " ms/iter, "
                  << std::setprecision(2) << cph << " candidates/h\n";
    }
    if (res.empty()) { std::cout << "[TUNE] no configuration could be measured\n"; return 1; }

    std::sort(res.begin(), res.end(), [](const Res& a, const Res& b) { return a.cph > b.cph; });
    std::cout << "\n  workers x threads | cores | ms/iter (1 test) | 1 candidate | candidates/h | batch of " << cands
              << "\n  ------------------+-------+------------------+-------------+--------------+------------\n";
    for (const Res& r : res) {
        const double t_cand = (double)n * r.ms / 1000.0;
        const double t_batch = std::ceil((double)cands / r.W) * t_cand;   // rounds of W at once
        std::cout << "  " << std::setw(7) << r.W << " x " << std::setw(2) << r.T << "       | "
                  << std::setw(5) << r.W * r.T << " | "
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.ms << "     | "
                  << std::setw(11) << fmt_duration(t_cand) << " | " << std::setw(12) << std::setprecision(2) << r.cph
                  << " | " << fmt_duration(t_batch) << "\n";
    }
    const Res& b = res[0];
    std::cout << "\n[TUNE] Best: " << b.W << " candidate(s) at once x " << b.T << " thread(s). Use these options (with the"
              << " same taskset as now):\n         --workers " << b.W << " --threads " << b.T << " --pin\n";
    if (res.size() > 1)
        std::cout << "[TUNE] Second: " << res[1].W << " x " << res[1].T << " ("
                  << std::setprecision(1) << 100.0 * (1.0 - res[1].cph / b.cph) << "% fewer candidates per hour)\n";
    return 0;
}