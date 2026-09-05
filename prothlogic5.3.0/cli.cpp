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
struct ProthOptions {
    unsigned n_pow2 = 0;
    uint64_t k_start = 0;
    uint64_t k_end = 0;
    uint64_t pstart = 0;
    uint64_t pmax = 50000000000;
    int workers = 1;       
    int max_workers = 0;
    int proth_bases = 16;
    int report_step = 25;
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
        else if (arg == "--k-start" && i + 1 < argc) opt.k_start = std::stoull(argv[++i]);
        else if (arg == "--k-end" && i + 1 < argc) opt.k_end = std::stoull(argv[++i]);
        else if (arg == "--pmax" && i + 1 < argc) opt.pmax = std::stoull(argv[++i]);
        else if (arg == "--pstart" && i + 1 < argc) opt.pstart = std::stoull(argv[++i]);
        else if (arg == "--workers" && i + 1 < argc) opt.workers = std::stoi(argv[++i]);
        else if (arg == "--max-workers" && i + 1 < argc) opt.max_workers = std::stoi(argv[++i]); // workers per candidate
        else if (arg == "--report" && i + 1 < argc) opt.report_step = std::stoi(argv[++i]);
        else if (arg == "--out-primes" && i + 1 < argc) opt.out_primes = argv[++i];
        else if (arg == "--out-sieve" && i + 1 < argc) opt.out_sieve = argv[++i];
        else if (arg == "--in-file" && i + 1 < argc) opt.in_file = argv[++i];
        else if (arg == "--sieve-only") opt.sieve_only = true;
        else if (arg == "--sparse-sieve") opt.sparse_sieve = true;
    }
}

static int run_proth(const ProthOptions& opt) {
    int concurrent_tasks = opt.max_workers > 0 ? opt.max_workers : get_auto_workers(opt.n_pow2);
    int threads_per_task = opt.workers > 0 ? opt.workers : 1;

    std::cout << "\n  ProthLogic 5.3 CPU & GPUsieve " << std::endl;
    std::cout << "  N=" << opt.n_pow2;
    if (opt.in_file.empty()) {
        std::cout << " Range k=[" << opt.k_start << ".." << opt.k_end << "]" << std::endl;
    }
    else {
        std::cout << " (Loaded from file: " << opt.in_file << ")" << std::endl;
    }
    std::cout << "  Optimal workers to use in batch mode: " << concurrent_tasks << "  |  Threads per candidate: " << threads_per_task << "\n" << std::endl;

    std::vector<Task> tasks;
    std::vector<uint64_t> final_survivors;

    if (!opt.in_file.empty()) {
        std::vector<uint64_t> loaded_ks = load_survivors_from_file(opt.in_file, opt.n_pow2);

        
        if (opt.sparse_sieve || opt.pstart > 0 || opt.sieve_only) {
            std::cout << "Starting Extreme GPU SPARSE Sieve..." << std::endl;
            auto t_sieve_0 = std::chrono::high_resolution_clock::now();
            if (!loaded_ks.empty()) {
                final_survivors = execute_sparse_streaming_sieve(
                    loaded_ks,
                    opt.n_pow2,
                    opt.pstart,
                    opt.pmax
                );
            }
            auto t_sieve_1 = std::chrono::high_resolution_clock::now();
            std::cout << "[DONE] Sparse Sieve time: "
                << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
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
        final_survivors = execute_full_streaming_sieve(
            opt.k_start,
            opt.k_end,
            opt.n_pow2,
            opt.pmax
        );
        auto t_sieve_1 = std::chrono::high_resolution_clock::now();
        std::cout << "[DONE] Sieve time: "
            << std::chrono::duration<double>(t_sieve_1 - t_sieve_0).count() << " s\n";
    }

    if (!opt.out_sieve.empty() && !final_survivors.empty()) {
        std::ofstream outfile(opt.out_sieve);
        if (outfile.is_open()) {
            for (uint64_t k : final_survivors) {
                // EXPORT to standard ABC Format: k 2 n 1
                outfile << k << " 2 " << opt.n_pow2 << " 1\n";
            }
            outfile.close();
            std::cout << "[SIEVE] Saving " << final_survivors.size()
                << " surviving candidates to file: " << opt.out_sieve << "\n";
        }
        else {
            std::cerr << "[ERROR] Cannot open file for writing: " << opt.out_sieve << "\n";
        }
    }

    if (opt.sieve_only) {
        std::cout << "[INFO] The --sieve-only flag is active. Stopping before the FFT. Execution complete.\n";
        return 0;
    }

    if (!final_survivors.empty()) {
        build_proth_ff_tasks_for_range(
            opt.n_pow2,
            final_survivors,
            opt.out_primes,
            concurrent_tasks,
            tasks,
            opt.report_step
        );
    }

    if (tasks.empty()) {
        std::cout << "[INFO] No candidates survived the sieve. Finishing." << std::endl;
        return 0;
    }

    progress_init((uint64_t)tasks.size(), 500, opt.n_pow2, opt.k_start, opt.k_end);
    std::cout << "\n  Starting Proth Test (" << tasks.size() << " candidates/task to test)" << std::endl;
    auto t_calc_0 = std::chrono::steady_clock::now();
    dispatch(tasks, concurrent_tasks, threads_per_task);

    auto t_calc_1 = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(t_calc_1 - t_calc_0).count();

    std::cout << "\nCalculations finished." << std::endl;
    std::cout << "Total computation time: "
        << std::fixed << std::setprecision(2) << total_seconds << " s" << std::endl;

    return 0;
}


bool use_avx2_fallback = false;

int main(int argc, char** argv) {
    if (!check_avx512_support()) {
        use_avx2_fallback = true;
        std::cout << "[INFO] AVX-512 not detected on this CPU. Seamlessly switching to high-performance AVX2 fallback engine.\n";
    }

    setenv("OMP_WAIT_POLICY", "PASIVE", 1);
    setenv("OMP_DYNAMIC", "FALSE", 1);

    if (argc <= 1) {
        std::cout << "Usage: prothlogic --n <pow2> --k-start <a> --k-end <b> [--workers W] [--pmax P] [--in-file file.txt]\n";
        return 0;
    }

    ProthOptions opt;
    parse_args_proth(argc, argv, opt);

    
    if (opt.n_pow2 == 0) {
        std::cerr << "[ERROR] Missing required parameter --n" << std::endl;
        return 1;
    }
    if (opt.in_file.empty() && opt.k_start == 0) {
        std::cerr << "[ERROR] Missing required parameter --k-start (unless --in-file is used)" << std::endl;
        return 1;
    }

    if (opt.k_end < opt.k_start) opt.k_end = opt.k_start;

    return run_proth(opt);
}