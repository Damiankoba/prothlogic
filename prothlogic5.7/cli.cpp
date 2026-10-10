// Copyright (C) Damian Koba.

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
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
#include "progress_core.h"
#include "dcore.h"
#include "auto_config.h"
#include "log_config.h"

bool g_detail = false;
bool g_progress = true;
bool g_progress_tty = false;
bool g_batch_agg = false;
bool g_stdout_tty = false;
unsigned long long g_batch_total = 0;

// Check for AVX-512F support to avoid SIGILL on older CPUs
bool check_avx512_support() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f") > 0;
#else
    return true; 
#endif
}

std::vector<uint64_t> load_survivors_from_file(const std::string& filename, unsigned expected_n) {
    std::vector<uint64_t> candidates;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        return candidates;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);

        uint64_t k, b, c;
        uint32_t n;

        //  ABC: k b n c (eg. 5031 2 136729 1)
        if (iss >> k >> b >> n >> c) {
            
            if (b == 2 && c == 1 && (expected_n == 0 || n == expected_n)) {
                candidates.push_back(k);
            }
        }
    }

    file.close();
    std::cout << "[SYSTEM] Loaded " << candidates.size() << " candidates from file " << filename << ".\n";
    return candidates;
}

std::vector<uint32_t> load_n_survivors_from_file(const std::string& filename, uint64_t expected_k) {
    std::vector<uint32_t> candidates;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        return candidates;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        uint64_t k, b, c;
        uint32_t n;
        if (iss >> k >> b >> n >> c) {
            if (b == 2 && c == 1 && k == expected_k) {
                candidates.push_back(n);
            }
        }
    }
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
    int workers = 1;       
    int max_workers = 0;
    int proth_bases = 16;
    int report_step = 0;          // 0 = no --report (single progress line); N > 0 = classic [STEP] lines every N %
    bool detail = false;          // --detail
    std::string out_primes = "primes.txt";
    std::string out_sieve = "";
    std::string in_file = "";
    bool sieve_only = false;
    bool sparse_sieve = false;
};



static void parse_args_proth(int argc, char** argv, ProthOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) opt.n_pow2 = std::stoul(argv[++i]);
        else if (arg == "--n-start" && i + 1 < argc) opt.n_start = std::stoul(argv[++i]); 
        else if (arg == "--n-end" && i + 1 < argc) opt.n_end = std::stoul(argv[++i]);     
        else if (arg == "--k-start" && i + 1 < argc) opt.k_start = std::stoull(argv[++i]);
        else if (arg == "--k-end" && i + 1 < argc) opt.k_end = std::stoull(argv[++i]);
        else if (arg == "--pmax" && i + 1 < argc) opt.pmax = std::stoull(argv[++i]);
        else if (arg == "--pstart" && i + 1 < argc) opt.pstart = std::stoull(argv[++i]);
        else if (arg == "--workers" && i + 1 < argc) opt.workers = std::stoi(argv[++i]);          // threads per candidate
        else if (arg == "--max-workers" && i + 1 < argc) opt.max_workers = std::stoi(argv[++i]);  // candidates tested at once
        else if (arg == "--report" && i + 1 < argc) opt.report_step = std::stoi(argv[++i]);
        else if (arg == "--out-primes" && i + 1 < argc) opt.out_primes = argv[++i];
        else if (arg == "--out-sieve" && i + 1 < argc) opt.out_sieve = argv[++i];
        else if (arg == "--in-file" && i + 1 < argc) opt.in_file = argv[++i];
        else if (arg == "--sieve-only") opt.sieve_only = true;
        else if (arg == "--sparse-sieve") opt.sparse_sieve = true;
        else if (arg == "--detail") opt.detail = true;
    }
}

// Candidate written as k*2^n+1 with its number of decimal digits (same formula as the final report)
static std::string proth_name(uint64_t k, unsigned n) {
    const unsigned long long digits = static_cast<unsigned long long>(n * 0.30102999566 + std::log10((double)k)) + 1;
    return std::to_string(k) + "*2^" + std::to_string(n) + "+1  (" + std::to_string(digits) + " decimal digits)";
}

static int run_proth(const ProthOptions& opt) {
    bool is_n_range = (opt.n_start > 0);
    unsigned display_n = is_n_range ? opt.n_start : opt.n_pow2;
    int concurrent_tasks = opt.max_workers > 0 ? opt.max_workers : get_auto_workers(display_n);
    int threads_per_task = opt.workers > 0 ? opt.workers : 1;

    std::cout << "\n  ProthLogic 5.7 CPU & GPUsieve" << std::endl;

    // OUTPUT MODE
    if (is_n_range && opt.n_start == opt.n_end && opt.in_file.empty()) {
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
    std::cout << "  Optimal workers to use in batch mode: " << concurrent_tasks << "  |  Threads per candidate: " << threads_per_task << "\n" << std::endl;

    std::vector<Task> tasks;
    // ==========================================
        //  NEW MODE: FIXED K, VARIABLE N
        // ==========================================
    if (is_n_range) {
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
                loaded_ns = load_n_survivors_from_file(opt.in_file, opt.k_start);
                std::cout << "[SYSTEM] Loaded " << loaded_ns.size() << " N-candidates from " << opt.in_file << std::endl;
            }

            auto t_sieve_0 = std::chrono::high_resolution_clock::now();
            std::cout << "Starting Extreme GPU N-Range Sieve..." << std::endl;
            final_survivors_n = execute_n_range_streaming_sieve(opt.k_start, opt.n_start, opt.n_end, opt.pstart, opt.pmax, loaded_ns);
            auto t_sieve_1 = std::chrono::high_resolution_clock::now();
            std::cout << "[DONE] Sieve time: " << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
        }

        if (!opt.out_sieve.empty() && !final_survivors_n.empty()) {
            std::ofstream outfile(opt.out_sieve);
            if (outfile.is_open()) {
                for (uint32_t n_val : final_survivors_n) outfile << opt.k_start << " 2 " << n_val << " 1\n";
                outfile.close();
                std::cout << "[SIEVE] Saving " << final_survivors_n.size() << " surviving candidates to file: " << opt.out_sieve << "\n";
            }
        }

        if (opt.sieve_only) {
            std::cout << "[INFO] The --sieve-only flag is active. Stopping before the FFT.\n";
            return 0;
        }

        // We throw each surviving N as a separate, clean PROTH_SINGLE task
        for (uint32_t n_val : final_survivors_n) {
            Task t;
            t.type = TaskType::PROTH_SINGLE;
            t.n = n_val;
            t.k = opt.k_start;
            t.out_primes = opt.out_primes;
            t.report_step = opt.report_step;
            tasks.push_back(std::move(t));
        }
    }
    else {
        // ==========================================
        //  OLD MODE: K VARIABLES, N CONSTANTS
        // ==========================================
        std::vector<uint64_t> final_survivors;

        if (!opt.in_file.empty()) {
            std::vector<uint64_t> loaded_ks = load_survivors_from_file(opt.in_file, opt.n_pow2);
            if (opt.sparse_sieve || opt.pstart > 0 || opt.sieve_only) {
                std::cout << "Starting Extreme GPU SPARSE Sieve..." << std::endl;
                auto t_sieve_0 = std::chrono::high_resolution_clock::now();
                if (!loaded_ks.empty()) {
                    final_survivors = execute_sparse_streaming_sieve(loaded_ks, opt.n_pow2, opt.pstart, opt.pmax);
                }
                auto t_sieve_1 = std::chrono::high_resolution_clock::now();
                std::cout << "[DONE] Sparse Sieve time: " << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
            }
            else {
                std::cout << "[INFO] Bypassing GPU sieve. Directly loading candidates for FFT test.\n";
                final_survivors = loaded_ks;
            }
        }
        else if (opt.k_start == opt.k_end) {
            Task t;
            t.type = TaskType::PROTH_SINGLE;
            t.n = opt.n_pow2;
            t.k = opt.k_start;
            t.out_primes = opt.out_primes;
            t.report_step = opt.report_step;
            tasks.push_back(std::move(t));
        }
        else {
            auto t_sieve_0 = std::chrono::high_resolution_clock::now();
            std::cout << "Starting Extreme GPU Sieve..." << std::endl;
            final_survivors = execute_full_streaming_sieve(opt.k_start, opt.k_end, opt.n_pow2, opt.pmax);
            auto t_sieve_1 = std::chrono::high_resolution_clock::now();
            std::cout << "[DONE] Sieve time: " << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
        }

        if (!opt.out_sieve.empty() && !final_survivors.empty()) {
            std::ofstream outfile(opt.out_sieve);
            if (outfile.is_open()) {
                for (uint64_t k : final_survivors) outfile << k << " 2 " << opt.n_pow2 << " 1\n";
                outfile.close();
                std::cout << "[SIEVE] Saving " << final_survivors.size() << " surviving candidates to file: " << opt.out_sieve << "\n";
            }
        }

        if (opt.sieve_only) return 0;

        if (!final_survivors.empty()) {
            build_proth_ff_tasks_for_range(opt.n_pow2, final_survivors, opt.out_primes, concurrent_tasks, tasks, opt.report_step);
        }
    }

    // ==========================================
    //  COMMON DISPATCH (Running FFT)
    // ==========================================
    if (tasks.empty()) {
        std::cout << "[INFO] No candidates survived the sieve. Finishing." << std::endl;
        return 0;
    }

    // In-place progress line only in a terminal and with one candidate at a time (several workers would overwrite it)
    g_stdout_tty = (isatty(STDOUT_FILENO) != 0);
    g_progress_tty = g_stdout_tty && (concurrent_tasks == 1 || tasks.size() == 1);

    progress_init((uint64_t)tasks.size(), 500, display_n, opt.k_start, opt.k_end);
    size_t total_cands = 0;
    for (const Task& t : tasks) total_cands += (t.type == TaskType::PROTH_BATCH) ? t.k_list.size() : 1;
    const size_t at_once = std::min<size_t>((size_t)concurrent_tasks, tasks.size());
    // Several workers and many candidates: one shared batch progress line (unless --report is given)
    g_batch_total = total_cands;
    g_batch_agg = g_progress && at_once > 1 && total_cands > 1;
    if (g_batch_agg) batch_progress_start();
    std::cout << "\n  Starting Proth Test: " << total_cands << " candidate(s) in " << tasks.size()
              << " task(s), " << at_once << " at once x " << threads_per_task << " thread(s)" << std::endl;

    auto t_calc_0 = std::chrono::steady_clock::now();
    dispatch(tasks, concurrent_tasks, threads_per_task);
    auto t_calc_1 = std::chrono::steady_clock::now();

    double total_seconds = std::chrono::duration<double>(t_calc_1 - t_calc_0).count();

    std::cout << "\nCalculations finished." << std::endl;
    std::cout << "Total computation time: " << std::fixed << std::setprecision(2) << total_seconds << " s" << std::endl;

    return 0;
}


bool use_avx2_fallback = false;

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
    if (!check_avx512_support()) {
        use_avx2_fallback = true;
        std::cout << "[INFO] AVX-512 not detected on this CPU. Seamlessly switching to high-performance AVX2 fallback engine.\n";
    }

    // Note: libgomp reads its environment when the library is loaded, so these two calls have no effect on this
    // process ("PASIVE" is also not a valid value). All measurements were made with the default wait policy.
    setenv("OMP_WAIT_POLICY", "PASIVE", 1);
    setenv("OMP_DYNAMIC", "FALSE", 1);

    if (argc <= 1) {
        std::cout << "Usage: prothlogic --n <pow2> --k-start <a> --k-end <b> [--workers W] [--pmax P] [--in-file file.txt]\n"
                  << "       [--detail]    calibration, FFT engine, speed estimate, Max Err every 25%\n"
                  << "       [--report N]  classic [STEP] lines every N% (benchmarks), no live progress line\n";
        return 0;
    }

    ProthOptions opt;
    parse_args_proth(argc, argv, opt);

    // Output: --report N = classic [STEP] lines every N % (benchmarks); without it a single progress line,
    // and with --detail additionally [STEP] lines with Max Err every 25 %.
    g_detail = opt.detail;
    g_progress = (opt.report_step <= 0);
    if (opt.report_step <= 0) opt.report_step = opt.detail ? 25 : 0;


    if (opt.n_pow2 == 0 && opt.n_start == 0) {
        std::cerr << "[ERROR] Missing required parameter --n or --n-start" << std::endl;
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