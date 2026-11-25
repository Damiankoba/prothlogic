// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <unordered_set>
#include <unistd.h>
#include <sched.h>
#include <omp.h>

// Linux CPUID header for AVX-512 detection
#if defined(__linux__) || defined(linux)
#include <cpuid.h>
#endif

#include "presieve_core.h"
#include "task.h"
#include "worker.h"
#include "pipeline_proth.h"
#include "dcore.h"
#include "auto_config.h"
#include "log_config.h"
#include "fft_engine.h"
#include "gpu_proth_engine.h"
#include "checkpoint_manager.h"

bool g_detail = false;
bool g_progress = true;
bool g_progress_tty = false;
bool g_batch_agg = false;
bool g_stdout_tty = false;
unsigned long long g_batch_total = 0;
unsigned long long g_batch_total_iters = 0;
unsigned long long g_batch_full_iters = 0;
std::string g_primes_file = "primes.txt";

// Check for AVX-512F support to avoid SIGILL on older CPUs
bool check_avx512_support() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f") > 0;
#else
    return true; 
#endif
}

// Candidate lists. Two formats are read, also mixed in one file:
//   "k 2 n 1"                                  one candidate per line
//   "ABC <k>*2^<n>+1 [// comment]"             a header, then one line per candidate with the values of its variables:
//                                              <k> and <n> are numbers or variables $a, $b, ... (values in letter order),
//                                              e.g. "ABC 25902068371*2^$a+1" + lines "n", "ABC $a*2^3321928+1" + lines "k",
//                                              "ABC $a*2^$b+1" + lines "k n". "// Sieved to P" in the header is remembered.
// Other lines are ignored. Returns false when the file cannot be opened.
struct AbcTerm { bool var = false; int idx = 0; uint64_t val = 0; };
static bool abc_term(const std::string& s, AbcTerm& t) {
    if (s.size() == 2 && s[0] == '$' && s[1] >= 'a' && s[1] <= 'z') { t.var = true; t.idx = s[1] - 'a'; return true; }
    if (s.empty() || s.size() > 19 || s.find_first_not_of("0123456789") != std::string::npos) return false;
    t.var = false;
    t.val = std::stoull(s);   // at most 19 digits: always fits in 64 bits
    return true;
}
static bool read_candidate_list(const std::string& filename, std::vector<std::pair<uint64_t, unsigned>>& out,
                                std::string* sieved_to = nullptr) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        return false;
    }
    bool abc = false;
    AbcTerm tk, tn;
    int nvars = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.compare(0, 4, "ABC ") == 0) {
            std::string expr = line.substr(4), comment;
            const size_t cm = expr.find("//");
            if (cm != std::string::npos) { comment = expr.substr(cm + 2); expr = expr.substr(0, cm); }
            expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());
            const size_t star = expr.find("*2^");
            abc = star != std::string::npos && expr.size() > star + 5 && expr.compare(expr.size() - 2, 2, "+1") == 0 &&
                  abc_term(expr.substr(0, star), tk) && abc_term(expr.substr(star + 3, expr.size() - star - 5), tn);
            if (!abc) {
                std::cerr << "[WARN] " << filename << ": unsupported header \"" << line << "\" (expected ABC k*2^n+1) - "
                          << "its lines are skipped\n";
                tk = tn = AbcTerm{};
                continue;
            }
            nvars = std::max(tk.var ? tk.idx + 1 : 0, tn.var ? tn.idx + 1 : 0);
            const size_t st = comment.find("Sieved to");
            if (sieved_to && st != std::string::npos) {
                std::istringstream cs(comment.substr(st + 9));
                cs >> *sieved_to;
            }
            continue;
        }
        if (line.empty()) continue;
        std::istringstream iss(line);
        if (abc) {
            uint64_t v[26] = {};
            int got = 0;
            while (got < nvars && iss >> v[got]) ++got;
            if (got < nvars) continue;
            const uint64_t k = tk.var ? v[tk.idx] : tk.val, n = tn.var ? v[tn.idx] : tn.val;
            if (n <= 0xFFFFFFFFull) out.emplace_back(k, (unsigned)n);
            continue;
        }
        uint64_t k, b, c;
        uint32_t n;
        if (iss >> k >> b >> n >> c && b == 2 && c == 1) out.emplace_back(k, n);
    }
    return true;
}

// The sieve depth as it goes into an ABC header: 30000000000000 -> "3e13", 123456 -> "123456"
static std::string depth_str(uint64_t p) {
    int e = 0;
    uint64_t m = p;
    while (m != 0 && m % 10 == 0) { m /= 10; ++e; }
    return (e >= 6) ? std::to_string(m) + "e" + std::to_string(e) : std::to_string(p);
}

// Writes a sieve result in the ABC format: "ABC $a*2^$b+1 // Sieved to P", then "k n" per line. Only this general
// header: LLR2 1.3.3 does not read a constant in the header ("ABC 3*2^$a+1" is taken as an empty file).
// depth: "Sieved to ..." in the header (empty: no comment).
static void write_abc_list(const std::string& filename, const std::vector<std::pair<uint64_t, unsigned>>& kn,
                           const std::string& depth) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "[ERROR] Cannot write file: " << filename << "\n";
        return;
    }
    outfile << "ABC $a*2^$b+1";
    if (!depth.empty()) outfile << " // Sieved to " << depth;
    outfile << "\n";
    for (const auto& p : kn) outfile << p.first << " " << p.second << "\n";
    std::cout << "[SIEVE] Saving " << kn.size() << " surviving candidates to file: " << filename << "\n";
}

static std::vector<uint64_t> load_survivors_from_file(const std::string& filename, unsigned expected_n, std::string* sieved_to) {
    std::vector<uint64_t> candidates;
    std::vector<std::pair<uint64_t, unsigned>> pairs;
    if (!read_candidate_list(filename, pairs, sieved_to)) return candidates;
    for (const auto& p : pairs)
        if (expected_n == 0 || p.second == expected_n) candidates.push_back(p.first);
    std::cout << "[SYSTEM] Loaded " << candidates.size() << " candidates from file " << filename << ".\n";
    return candidates;
}

// --in-file restarted: the candidates whose result is already in composite.txt / the primes file (--out-primes) of the
// working directory ("k*2^n+1 is ...") are not tested again. A test interrupted in the middle resumes from its checkpoint.
static std::string kn_key(uint64_t k, unsigned n) { return std::to_string(k) + "*2^" + std::to_string(n); }
template <class T, class Key>
static void drop_already_tested(std::vector<T>& v, Key key) {
    std::unordered_set<std::string> done;   // "k*2^n" of every result line
    for (const std::string& name : { std::string("composite.txt"), g_primes_file }) {
        std::ifstream f(name);
        std::string line;
        while (std::getline(f, line)) {
            const size_t p = line.find("+1 is ");
            if (p != std::string::npos && p > 0) done.insert(line.substr(0, p));
        }
    }
    if (done.empty()) return;
    const size_t before = v.size();
    v.erase(std::remove_if(v.begin(), v.end(), [&](const T& x) { return done.count(key(x)) != 0; }), v.end());
    if (v.size() != before)
        std::cout << "[RESUME] " << (before - v.size()) << " candidate(s) already in composite.txt / " << g_primes_file << " - skipped, "
                  << v.size() << " left\n";
}

static std::vector<uint32_t> load_n_survivors_from_file(const std::string& filename, uint64_t expected_k, std::string* sieved_to) {
    std::vector<uint32_t> candidates;
    std::vector<std::pair<uint64_t, unsigned>> pairs;
    if (!read_candidate_list(filename, pairs, sieved_to)) return candidates;
    for (const auto& p : pairs)
        if (p.first == expected_k) candidates.push_back(p.second);
    return candidates;
}


struct ProthOptions {
    unsigned n_pow2 = 0;
    unsigned n_start = 0;  
    unsigned n_end = 0;
    uint64_t k_start = 0;
    uint64_t k_end = 0;
    uint64_t pstart = 0;
    uint64_t pmax = 50000000000;
    int workers = 0;              // --workers W: candidates tested at once (0 = automatic, get_auto_workers)
    int threads = 1;              // --threads T: threads per candidate
    int report_step = 0;          // 0 = no --report (single progress line); N > 0 = classic [STEP] lines every N %
    bool detail = false;          // --detail
    std::string out_primes = "primes.txt";
    std::string out_sieve = "";
    std::string in_file = "";
    bool sieve_only = false;
    bool sparse_sieve = false;
    bool tune = false;            // --tune: measure the CPU splits (workers x threads) instead of testing
    bool tune_all = false;        // --tune-all: --tune plus the splits that leave cores free (evenly per CCD)
    double tune_secs = 20.0;      // --tune-secs S: seconds per measured configuration
    bool pin = false;             // --pin: pin every worker and its threads (the layout --tune measures)
    bool gpu = false;             // --gpu: the NVIDIA card tests candidates too (one more worker on the same queue)
    bool gpu_only = false;        // --gpu-only: only the card tests (no CPU workers, no CPU calibration)
    std::vector<std::pair<uint64_t, unsigned>> file_pairs;   // --in-file alone, several k and several N: (k, n) per line
};

// --in-file without --n / --n-start: k and N come from the file (lines "k 2 n 1"). One N -> as --n N;
// one k with several N -> as --n-start/--n-end with that k; several k and several N -> every line its own test.
static bool detect_file_mode(ProthOptions& opt) {
    std::vector<std::pair<uint64_t, unsigned>> pairs;
    if (!read_candidate_list(opt.in_file, pairs)) return false;
    if (pairs.empty()) {
        std::cerr << "[ERROR] No candidates in " << opt.in_file << " (expected lines \"k 2 n 1\" or an ABC header)\n";
        return false;
    }
    bool one_k = true, one_n = true;
    unsigned n_lo = pairs[0].second, n_hi = pairs[0].second;
    for (const auto& p : pairs) {
        if (p.first != pairs[0].first) one_k = false;
        if (p.second != pairs[0].second) one_n = false;
        n_lo = std::min(n_lo, p.second);
        n_hi = std::max(n_hi, p.second);
    }
    if (one_n) opt.n_pow2 = pairs[0].second;   // the header shows N / the N range
    else if (one_k) {
        opt.k_start = opt.k_end = pairs[0].first;
        opt.n_start = n_lo;
        opt.n_end = n_hi;
    }
    else {
        if (opt.sparse_sieve || opt.pstart > 0 || opt.sieve_only) {
            std::cerr << "[ERROR] Sieving a file needs one N (or one k) in it; " << opt.in_file
                      << " has several of both - give --n N to take only that N\n";
            return false;
        }
        opt.file_pairs = std::move(pairs);
    }
    return true;
}

// One number written the mathematical way: K*2^N+1 or 2^N+1 (spaces allowed). K < 2^64, N < 2^32.
// An even K is moved into the power of two (6*2^10+1 = 3*2^11+1), as the test expects an odd k.
static bool parse_proth_expr(const std::string& text, uint64_t& k, unsigned& n) {
    std::string s;
    for (char c : text) if (c != ' ' && c != '\t') s += c;
    size_t pos = 0;
    auto num = [&](unsigned long long limit, unsigned long long& v) -> bool {
        const size_t start = pos;
        v = 0;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
            const unsigned d = (unsigned)(s[pos] - '0');
            if (v > (limit - d) / 10) return false;   // too large
            v = v * 10 + d;
            ++pos;
        }
        return pos > start;
    };
    unsigned long long kk = 1, nn = 0;
    if (s.compare(0, 2, "2^") != 0) {                  // K*2^N+1
        if (!num(0xFFFFFFFFFFFFFFFFull, kk) || pos >= s.size() || s[pos] != '*') return false;
        ++pos;
    }
    if (s.compare(pos, 2, "2^") != 0) return false;
    pos += 2;
    if (!num(0xFFFFFFFFull, nn) || s.compare(pos, std::string::npos, "+1") != 0) return false;
    if (kk == 0 || nn == 0) return false;
    while (kk % 2 == 0) { kk /= 2; ++nn; }
    if (nn > 0xFFFFFFFFull) return false;
    k = kk;
    n = (unsigned)nn;
    return true;
}

static bool parse_args_proth(int argc, char** argv, ProthOptions& opt) {
    int workers = -1, threads = -1, max_workers = -1;   // -1 = not given
    // N is 32-bit: a larger value was silently cut modulo 2^32 (--n 9073000000 tested 195*2^483065408+1)
    bool n_ok = true;
    std::string expr;                                   // a number written as K*2^N+1 (instead of --n / --k)
    auto read_n =[&](const std::string& flag, const char* s) -> unsigned {
        const unsigned long long v = std::stoull(s);
        if (v > 0xFFFFFFFFull) {
            std::cerr << "[ERROR] " << flag << " " << s << " is too large: N can be at most 4294967295\n";
            n_ok = false;
        }
        return (unsigned)v;
    };
    // --pmax / --pstart: a whole number (5000000000) or the short form (5e9, 1e13, 2.5e12), converted exactly
    auto read_p = [&](const std::string& flag, const std::string& s) -> uint64_t {
        const size_t ep = s.find_first_of("eE");
        const std::string mant = s.substr(0, ep);
        std::string ex = (ep == std::string::npos) ? "" : s.substr(ep + 1);
        if (!ex.empty() && ex[0] == '+') ex.erase(0, 1);   // 1e+13
        unsigned __int128 v = 0;
        int frac = -1;                       // digits after the point (-1: no point)
        bool ok = !mant.empty() && (ep == std::string::npos || !ex.empty());
        for (char c : mant) {
            if (c == '.' && frac < 0) { frac = 0; continue; }
            if (c < '0' || c > '9') { ok = false; break; }
            v = v * 10 + (unsigned)(c - '0');
            if (frac >= 0) ++frac;
            if (v > 0xFFFFFFFFFFFFFFFFull) { ok = false; break; }
        }
        int e = 0;
        for (char c : ex) {
            if (c < '0' || c > '9' || e > 100) { ok = false; break; }
            e = e * 10 + (c - '0');
        }
        if (frac > 0) e -= frac;
        for (; ok && e < 0; ++e) {           // 2.50e1 = 25: drop trailing zeros, a remainder is not a whole number
            if (v % 10 != 0) ok = false;
            v /= 10;
        }
        for (; ok && e > 0; --e) {
            v *= 10;
            if (v > 0xFFFFFFFFFFFFFFFFull) ok = false;
        }
        if (!ok) {
            std::cerr << "[ERROR] " << flag << " " << s << ": expected a whole number, e.g. 5000000000 or 5e9\n";
            n_ok = false;
            return 0;
        }
        return (uint64_t)v;
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) opt.n_pow2 = read_n(arg, argv[++i]);
        else if (arg == "--n-start" && i + 1 < argc) opt.n_start = read_n(arg, argv[++i]);
        else if (arg == "--n-end" && i + 1 < argc) opt.n_end = read_n(arg, argv[++i]);
        else if (arg == "--k-start" && i + 1 < argc) opt.k_start = std::stoull(argv[++i]);
        else if (arg == "--k-end" && i + 1 < argc) opt.k_end = std::stoull(argv[++i]);
        else if (arg == "--k" && i + 1 < argc) opt.k_start = opt.k_end = std::stoull(argv[++i]);   // one k
        else if (arg == "--pmax" && i + 1 < argc) opt.pmax = read_p(arg, argv[++i]);
        else if (arg == "--pstart" && i + 1 < argc) opt.pstart = read_p(arg, argv[++i]);
        else if (arg == "--workers" && i + 1 < argc) workers = std::stoi(argv[++i]);           // candidates tested at once
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);           // threads per candidate
        else if (arg == "--max-workers" && i + 1 < argc) max_workers = std::stoi(argv[++i]);   // old syntax (see below)
        else if (arg == "--report" && i + 1 < argc) opt.report_step = std::stoi(argv[++i]);
        else if (arg == "--out-primes" && i + 1 < argc) opt.out_primes = argv[++i];
        else if (arg == "--out-sieve" && i + 1 < argc) opt.out_sieve = argv[++i];
        else if (arg == "--in-file" && i + 1 < argc) opt.in_file = argv[++i];
        else if (arg == "--sieve-only") opt.sieve_only = true;
        else if (arg == "--sparse-sieve") opt.sparse_sieve = true;
        else if (arg == "--detail") opt.detail = true;
        else if (arg == "--tune") opt.tune = true;
        else if (arg == "--tune-all") { opt.tune = true; opt.tune_all = true; }
        else if (arg == "--tune-secs" && i + 1 < argc) opt.tune_secs = std::stod(argv[++i]);
        else if (arg == "--pin") opt.pin = true;
        else if (arg == "--gpu") opt.gpu = true;
        else if (arg == "--gpu-only") opt.gpu = opt.gpu_only = true;
        else if (!arg.empty() && arg[0] != '-') {      // the number itself, e.g. 25902068371*2^4000168+1
            if (!expr.empty()) {
                std::cerr << "[ERROR] Only one number can be given this way (" << expr << ", " << arg
                          << "); for a list use --in-file\n";
                return false;
            }
            expr = arg;
        }
    }
    if (!n_ok) return false;
    if (!expr.empty()) {
        uint64_t k = 0;
        unsigned n = 0;
        if (!parse_proth_expr(expr, k, n)) {
            std::cerr << "[ERROR] " << expr << ": expected a number written as K*2^N+1, e.g. 25902068371*2^4000168+1\n"
                      << "        (K at most 18446744073709551615, N at most 4294967295)\n";
            return false;
        }
        if (opt.n_pow2 || opt.n_start || opt.n_end || opt.k_start || opt.k_end || !opt.in_file.empty()) {
            std::cerr << "[ERROR] Give the number either as " << expr << " or with --n/--k/--in-file, not both\n";
            return false;
        }
        const std::string norm = std::to_string(k) + "*2^" + std::to_string(n) + "+1";
        std::string plain;
        for (char c : expr) if (c != ' ' && c != '\t') plain += c;
        if (plain != norm && plain != "2^" + std::to_string(n) + "+1")
            std::cout << "[INFO] " << plain << " = " << norm << " (k must be odd)\n";
        opt.n_pow2 = n;
        opt.k_start = opt.k_end = k;
    }
    // Old syntax (older versions): --workers T = threads per candidate, --max-workers W = candidates at once.
    // It is recognized by --max-workers; mixed with --threads it is ambiguous.
    if (max_workers >= 0) {
        if (threads >= 0) {
            std::cerr << "[ERROR] --max-workers is the old syntax and cannot be mixed with --threads.\n"
                      << "        Use: --workers <candidates at once> --threads <threads per candidate>\n";
            return false;
        }
        threads = workers;
        workers = max_workers;
        std::cout << "[WARN] Old syntax --workers T --max-workers W. Same meaning in the new syntax: --workers "
                  << workers << " --threads " << (threads > 0 ? threads : 1) << "\n";
    }
    if (workers > 0) opt.workers = workers;
    if (threads > 0) opt.threads = threads;
    return true;
}

// Candidate written as k*2^n+1 with its number of decimal digits (same formula as the final report)
static std::string proth_name(uint64_t k, unsigned n) {
    const unsigned long long digits = static_cast<unsigned long long>(n * 0.30102999566 + std::log10((double)k)) + 1;
    return std::to_string(k) + "*2^" + std::to_string(n) + "+1  (" + std::to_string(digits) + " decimal digits)";
}

static int run_proth(const ProthOptions& opt) {
    bool is_n_range = (opt.n_start > 0);
    unsigned display_n = is_n_range ? opt.n_start : opt.n_pow2;
    for (const auto& p : opt.file_pairs) display_n = std::max(display_n, p.second);   // several N: the largest
    int concurrent_tasks = opt.workers > 0 ? opt.workers : get_auto_workers(display_n);
    // a single candidate given by hand (--n N --k K, --n-start = --n-end): one worker, not the automatic count
    const bool single = opt.in_file.empty() && opt.k_start == opt.k_end && (!is_n_range || opt.n_start == opt.n_end);
    if (single && opt.workers <= 0) concurrent_tasks = 1;
    int threads_per_task = opt.threads > 0 ? opt.threads : 1;

#ifdef LOTR_GPU_ONLY
    std::cout << "\n  ProthLogic 5.8 & GPUsieve (GPU-only build: no AVX-512 CPU engine)" << std::endl;
#else
    std::cout << "\n  ProthLogic 5.8 & GPUsieve" << std::endl;
#endif

    // OUTPUT MODE
    if (!opt.file_pairs.empty()) {
        std::cout << "  " << opt.file_pairs.size() << " candidates from file: " << opt.in_file
                  << " (several k and N, largest N=" << display_n << ")" << std::endl;
    }
    else if (is_n_range && opt.n_start == opt.n_end && opt.in_file.empty()) {
        std::cout << "  " << proth_name(opt.k_start, opt.n_start) << std::endl;
    }
    else if (is_n_range) {
        std::cout << "  N Range=[" << opt.n_start << ".." << opt.n_end << "]  Fixed K=" << opt.k_start << std::endl;
    }
    else if (opt.in_file.empty() && opt.k_start == opt.k_end) {
        std::cout << "  " << proth_name(opt.k_start, opt.n_pow2) << std::endl;
    }
    else {
        std::cout << "  N=" << opt.n_pow2;
        if (opt.in_file.empty()) {
            std::cout << " Range k=[" << opt.k_start << ".." << opt.k_end << "]" << std::endl;
        }
        else {
            std::cout << " (Loaded from file: " << opt.in_file << ")" << std::endl;
        }
    }
    if (opt.gpu_only) {
        std::cout << "  GPU only (no CPU workers)\n" << std::endl;
        if (opt.workers > 0 || opt.threads > 1) std::cout << "[INFO] --gpu-only: --workers / --threads are ignored\n";
    }
    else
        std::cout << "  Workers (candidates at once): " << concurrent_tasks << (opt.workers > 0 || single ? "" : " (auto)")
                  << "  |  Threads per candidate: " << threads_per_task << "\n" << std::endl;

    std::vector<Task> tasks;
    std::string depth;   // sieve depth: from the header of the input file, or the --pmax of a sieve run here (ABC output)
    if (!opt.file_pairs.empty()) {
        // --in-file alone with several k and several N: every line is its own test, in the file order (no sieve)
        std::vector<std::pair<uint64_t, unsigned>> pairs = opt.file_pairs;
        drop_already_tested(pairs, [](const std::pair<uint64_t, unsigned>& p) { return kn_key(p.first, p.second); });
        for (const auto& p : pairs) {
            Task t;
            t.type = TaskType::PROTH_SINGLE;
            t.n = p.second;
            t.k = p.first;
            t.report_step = opt.report_step;
            tasks.push_back(std::move(t));
        }
    }
    // ==========================================
    // FIXED K, RANGE OF N
    // ==========================================
    else if (is_n_range) {
        std::vector<uint32_t> final_survivors_n;

        if (opt.n_start == opt.n_end) {
            if (!opt.in_file.empty()) {
                std::cout << "[WARN] --n-start/--n-end = a single N with fixed k (" << opt.k_start << "): file " << opt.in_file
                          << " is IGNORED.\n       To test the k values from the file at a fixed N use: --n " << opt.n_start
                          << " --in-file " << opt.in_file << "\n";
            }
            final_survivors_n.push_back(opt.n_start);
        }
        else {
            std::vector<uint32_t> loaded_ns;
            if (!opt.in_file.empty()) {
                loaded_ns = load_n_survivors_from_file(opt.in_file, opt.k_start, &depth);
                std::cout << "[SYSTEM] Loaded " << loaded_ns.size() << " N-candidates from " << opt.in_file
                          << (depth.empty() ? "" : " (sieved to " + depth + ")") << std::endl;
            }

            if (!opt.in_file.empty() && !(opt.sparse_sieve || opt.pstart > 0 || opt.sieve_only)) {
                // as in the fixed-N mode: a file of survivors is tested directly (a deeper sieve: --pstart / --sieve-only)
                std::cout << "[INFO] Bypassing GPU sieve. Directly loading candidates for primality test.\n";
                for (uint32_t n_val : loaded_ns)
                    if (n_val >= opt.n_start && n_val <= opt.n_end) final_survivors_n.push_back(n_val);
                drop_already_tested(final_survivors_n, [&](uint32_t n_val) { return kn_key(opt.k_start, n_val); });
            }
            else {
                auto t_sieve_0 = std::chrono::high_resolution_clock::now();
                final_survivors_n = execute_n_range_streaming_sieve(opt.k_start, opt.n_start, opt.n_end, opt.pstart, opt.pmax, loaded_ns);
                auto t_sieve_1 = std::chrono::high_resolution_clock::now();
                std::cout << "[DONE] Sieve time: " << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
                depth = depth_str(opt.pmax);
            }
        }

        if (!opt.out_sieve.empty() && !final_survivors_n.empty()) {
            std::vector<std::pair<uint64_t, unsigned>> kn;
            for (uint32_t n_val : final_survivors_n) kn.emplace_back(opt.k_start, n_val);
            write_abc_list(opt.out_sieve, kn, depth);
        }

        if (opt.sieve_only) {
            std::cout << "[INFO] The --sieve-only flag is active. Stopping before the primality test.\n";
            return 0;
        }

        // We throw each surviving N as a separate, clean PROTH_SINGLE task
        for (uint32_t n_val : final_survivors_n) {
            Task t;
            t.type = TaskType::PROTH_SINGLE;
            t.n = n_val;
            t.k = opt.k_start;
            t.report_step = opt.report_step;
            tasks.push_back(std::move(t));
        }
    }
    else {
        // ==========================================
        // FIXED N, RANGE OF K
        // ==========================================
        std::vector<uint64_t> final_survivors;

        if (!opt.in_file.empty()) {
            std::vector<uint64_t> loaded_ks = load_survivors_from_file(opt.in_file, opt.n_pow2, &depth);
            if (opt.sparse_sieve || opt.pstart > 0 || opt.sieve_only) {
                auto t_sieve_0 = std::chrono::high_resolution_clock::now();
                if (!loaded_ks.empty()) {
                    final_survivors = execute_sparse_streaming_sieve(loaded_ks, opt.n_pow2, opt.pstart, opt.pmax);
                    depth = depth_str(opt.pmax);
                }
                auto t_sieve_1 = std::chrono::high_resolution_clock::now();
                std::cout << "[DONE] Sparse Sieve time: " << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
            }
            else {
                std::cout << "[INFO] Bypassing GPU sieve. Directly loading candidates for primality test.\n";
                final_survivors = loaded_ks;
                drop_already_tested(final_survivors, [&](uint64_t k) { return kn_key(k, opt.n_pow2); });
            }
        }
        else if (opt.k_start == opt.k_end) {
            Task t;
            t.type = TaskType::PROTH_SINGLE;
            t.n = opt.n_pow2;
            t.k = opt.k_start;
            t.report_step = opt.report_step;
            tasks.push_back(std::move(t));
        }
        else {
            auto t_sieve_0 = std::chrono::high_resolution_clock::now();
            final_survivors = execute_full_streaming_sieve(opt.k_start, opt.k_end, opt.n_pow2, opt.pmax);
            auto t_sieve_1 = std::chrono::high_resolution_clock::now();
            std::cout << "[DONE] Sieve time: " << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
            depth = depth_str(opt.pmax);
        }

        if (!opt.out_sieve.empty() && !final_survivors.empty()) {
            std::vector<std::pair<uint64_t, unsigned>> kn;
            for (uint64_t k : final_survivors) kn.emplace_back(k, opt.n_pow2);
            write_abc_list(opt.out_sieve, kn, depth);
        }

        if (opt.sieve_only) return 0;

        if (!final_survivors.empty()) {
            build_proth_ff_tasks_for_range(opt.n_pow2, final_survivors, concurrent_tasks, tasks, opt.report_step);
        }
    }

    // ==========================================
    // COMMON DISPATCH (primality tests)
    // ==========================================
    if (tasks.empty()) {
        std::cout << "[INFO] No candidates survived the sieve. Finishing." << std::endl;
        return 0;
    }

    if (opt.tune) return tune_configs(tasks, opt.tune_secs, opt.tune_all);   // measure the CPU splits, test nothing

    // Proth's theorem proves primality only for k < 2^n; for a larger k a "prime" result is a probable prime
    {
        size_t big_k = 0;
        std::string example;
        auto note = [&](uint64_t k, unsigned n) {
            if (n < 64 && (k >> n) != 0) {
                if (big_k++ == 0) example = std::to_string(k) + "*2^" + std::to_string(n) + "+1";
            }
        };
        for (const Task& t : tasks) {
            if (t.type == TaskType::PROTH_BATCH) { for (const uint64_t kk : t.k_list) note(kk, t.n); }
            else note(t.k, t.n);
        }
        if (big_k > 0)
            std::cout << "[WARN] " << big_k << " candidate(s) with k >= 2^n (e.g. " << example << "): Proth's theorem does "
                      << "not apply, a \"prime\" result there is only a probable prime\n";
    }

    // In-place progress line only in a terminal and with one candidate at a time (several workers would overwrite it)
    size_t total_cands = 0;
    for (const Task& t : tasks) total_cands += (t.type == TaskType::PROTH_BATCH) ? t.k_list.size() : 1;
    // --gpu: the card is one more worker and the queue holds single candidates (dispatch)
    // --gpu-only: the card alone - how many at once: gpu_streams_auto (a quick test of 1, 2, ... at once before a batch),
    // with the n and k of dispatch (the largest of all tasks), which then takes the kept result
    unsigned gpu_n = 0;
    uint64_t gpu_k = 3;
    for (const Task& t : tasks) {
        gpu_n = std::max(gpu_n, t.n);
        if (t.type == TaskType::PROTH_BATCH) { for (const uint64_t kk : t.k_list) gpu_k = std::max(gpu_k, kk); }
        else gpu_k = std::max(gpu_k, t.k);
    }
    const size_t gpu_at_once = opt.gpu_only && !tasks.empty()
        ? std::max<size_t>(1, std::min<size_t>((size_t)gpu_streams_auto(gpu_n, gpu_k, total_cands), total_cands)) : 0;
    g_stdout_tty = (isatty(STDOUT_FILENO) != 0);
    g_progress_tty = g_stdout_tty && (opt.gpu_only ? gpu_at_once == 1
                                      : opt.gpu ? total_cands == 1 : (concurrent_tasks == 1 || tasks.size() == 1));

    const size_t at_once = opt.gpu_only ? 0 : std::min<size_t>((size_t)concurrent_tasks, opt.gpu ? total_cands : tasks.size());
    // Several workers and many candidates: one shared batch progress line (unless --report is given)
    g_batch_total = total_cands;
    // Squarings of the whole batch (n - 1 per candidate) for the batch ETA, minus those already in checkpoints: a test
    // resumed at iteration i adds only its new squarings to the count.
    g_batch_total_iters = 0;
    g_batch_full_iters = 0;
    auto add_cand = [&](uint64_t k, unsigned n) {
        const unsigned long long it = n > 0 ? n - 1 : 0;
        const unsigned long long ck = checkpoint_iteration("check_k" + std::to_string(k) + "_n" + std::to_string(n) + ".bin");
        g_batch_total_iters += (ck < it) ? it - ck : it;
        g_batch_full_iters += it;
    };
    for (const Task& t : tasks) {
        if (t.type == TaskType::PROTH_BATCH) { for (const uint64_t k : t.k_list) add_cand(k, t.n); }
        else add_cand(t.k, t.n);
    }
    g_batch_agg = g_progress && (opt.gpu_only ? gpu_at_once : at_once + (opt.gpu ? 1 : 0)) > 1 && total_cands > 1;
    batch_progress_start();   // also one at a time: elapsed / total ETA on the live line
    if (opt.gpu_only)
        std::cout << "\n  Starting Proth Test: " << total_cands << " candidate(s), " << gpu_at_once << " at once on the GPU"
                  << std::endl;
    else
        std::cout << "\n  Starting Proth Test: " << total_cands << " candidate(s) in " << tasks.size()
                  << " task(s), " << at_once << " at once x " << threads_per_task << " thread(s)" << (opt.gpu ? " + GPU" : "")
                  << std::endl;
    {   // more compute threads than CPUs: they take turns on the cores and every test runs much slower
        cpu_set_t allowed;
        const size_t cpus = (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) ? (size_t)CPU_COUNT(&allowed) : 0;
        if (cpus > 0 && at_once * (size_t)threads_per_task > cpus)
            std::cout << "[WARN] " << at_once << " x " << threads_per_task << " = " << at_once * threads_per_task
                      << " threads on " << cpus << " CPU(s): slower. Choose --workers W --threads T with W x T <= "
                      << cpus << std::endl;
    }

    auto t_calc_0 = std::chrono::steady_clock::now();
    dispatch(tasks, concurrent_tasks, threads_per_task, opt.pin, opt.gpu, opt.gpu_only);
    auto t_calc_1 = std::chrono::steady_clock::now();

    double total_seconds = std::chrono::duration<double>(t_calc_1 - t_calc_0).count();

    std::cout << "\nCalculations finished." << std::endl;
    std::cout << "Total computation time: " << std::fixed << std::setprecision(2) << total_seconds << " s" << std::endl;

    return 0;
}


int main(int argc, char** argv) {
    // OpenMP thread-binding variables in the environment (e.g. set by a Docker image) pin every thread that enters a
    // parallel region to the first "place" (core 0 = CPUs {0,12}): all batch-mode workers then ran on one core
    // (~20x slower). OpenMP reads them before main(), so we remove them and re-execute the program.
    // LOTR_KEEP_OMP_BIND=1 keeps them unchanged.
    if (std::getenv("LOTR_KEEP_OMP_BIND") == nullptr && std::getenv("LOTR_OMP_REEXEC") == nullptr &&
        (std::getenv("OMP_PROC_BIND") || std::getenv("OMP_PLACES") || std::getenv("GOMP_CPU_AFFINITY"))) {
        // OpenMP has already pinned the main thread (when the library was loaded) and the CPU mask survives execv.
        // The original CPU set (e.g. from taskset) = union of all OpenMP places - restore it before execv.
        const int np = omp_get_num_places();
        if (np > 0) {
            cpu_set_t all;
            CPU_ZERO(&all);
            for (int p = 0; p < np; ++p) {
                const int cnt = omp_get_place_num_procs(p);
                if (cnt <= 0) continue;
                std::vector<int> ids((size_t)cnt);
                omp_get_place_proc_ids(p, ids.data());
                for (int id : ids) if (id >= 0 && id < CPU_SETSIZE) CPU_SET(id, &all);
            }
            if (CPU_COUNT(&all) > 0) sched_setaffinity(0, sizeof(cpu_set_t), &all);
        }
        unsetenv("OMP_PROC_BIND");
        unsetenv("OMP_PLACES");
        unsetenv("GOMP_CPU_AFFINITY");
        setenv("LOTR_OMP_REEXEC", "1", 1);   // guard against an exec loop
        execv("/proc/self/exe", argv);
        // if execv fails we simply continue (with pinning)
    }

    capture_process_affinity();   // before any OpenMP/CUDA activity (see dcore.cpp)
    if (std::getenv("LOTR_KBENCH") != nullptr) {   // FFT kernel microbenchmark only
        fft_kernel_bench();
        return 0;
    }
#ifndef LOTR_GPU_ONLY
    if (!check_avx512_support()) {
        std::cerr << "[ERROR] this CPU has no AVX-512: ProthLogic needs AVX-512 F/DQ/VBMI/IFMA\n"
                  << "        (sieve and tests on the NVIDIA card only: ./build.sh gpu)\n";
        return 1;
    }
#endif

    if (argc <= 1) {
        std::cout << "Usage: prothlogic --in-file file.txt   k and N from the file (lines: k 2 n 1)\n"
                  << "       prothlogic \"k*2^n+1\"           one candidate, e.g. prothlogic \"25902068371*2^4000168+1\"\n"
                  << "       prothlogic --n <pow2> --k <k>   one candidate k*2^n+1 (same as above)\n"
                  << "       prothlogic --n <pow2> --k-start <a> --k-end <b> [--pmax P] [--in-file file.txt]\n"
                  << "       prothlogic --n-start <a> --n-end <b> --k-start <k> [--in-file file.txt]   fixed k, range of N\n"
                  << "       [--workers W] candidates tested at once (default: automatic)\n"
                  << "       [--threads T] threads per candidate (default 1)\n"
                  << "       [--detail]    calibration, FFT engine, speed estimate, Max Err every 25%\n"
                  << "       [--report N]  classic [STEP] lines every N% (benchmarks), no live progress line\n"
                  << "       [--tune [--tune-secs S]]  measure workers x threads splits for these candidates, test nothing\n"
                  << "       [--tune-all]  as --tune, also the splits that leave cores free (e.g. 10 x 1, 2 x 4)\n"
                  << "       [--pin]       pin each worker and its threads to their CPUs (one CCD per worker when it fits)\n"
                  << "       [--gpu]       the NVIDIA card tests candidates too (one more worker on the same queue)\n"
                  << "       [--gpu-only]  only the card tests (no CPU workers; e.g. taskset -c 11,23 next to CPU tests)\n";
#ifdef LOTR_GPU_ONLY
        std::cout << "  GPU-only build (./build.sh gpu): the card sieves and tests (always --gpu-only), no --tune\n";
#endif
        return 0;
    }

    ProthOptions opt;
    if (!parse_args_proth(argc, argv, opt)) return 1;
#ifdef LOTR_GPU_ONLY
    // GPU-only build (./build.sh gpu for CPUs without AVX-512): no CPU engine, the card sieves and tests
    if (opt.tune) {
        std::cerr << "[ERROR] --tune measures the CPU engine, which this GPU-only build (./build.sh gpu) does not have\n";
        return 1;
    }
    opt.gpu = opt.gpu_only = true;
#endif

    // Output: --report N = classic [STEP] lines every N % (benchmarks); without it a single progress line,
    // and with --detail additionally [STEP] lines with Max Err every 25 %.
    g_detail = opt.detail;
    g_primes_file = opt.out_primes;
    g_progress = (opt.report_step <= 0);
    if (opt.report_step <= 0) opt.report_step = opt.detail ? 25 : 0;


    if (!opt.in_file.empty() && opt.n_pow2 == 0 && opt.n_start == 0 && !detect_file_mode(opt)) return 1;
    if (opt.n_pow2 == 0 && opt.n_start == 0 && opt.file_pairs.empty()) {
        std::cerr << "[ERROR] Missing required parameter --n, --n-start or --in-file" << std::endl;
        return 1;
    }
    if (opt.in_file.empty() && opt.k_start == 0) {
        std::cerr << "[ERROR] Missing required parameter --k-start (unless --in-file is used)" << std::endl;
        return 1;
    }

    if (opt.k_end < opt.k_start) opt.k_end = opt.k_start;
    if (opt.n_end < opt.n_start) opt.n_end = opt.n_start;


    if (opt.n_start > 0 && opt.k_start != opt.k_end) {
        std::cout << "[INFO] N-range mode selected. Forcing K to be fixed (k_end = k_start).\n";
        opt.k_end = opt.k_start;
    }

    return run_proth(opt);
}