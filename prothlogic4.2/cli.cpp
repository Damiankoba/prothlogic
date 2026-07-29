// Copyright (C) Damian Koba.

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iomanip>
#include "presieve_core.h"
#include "task.h"
#include "worker.h"
#include "pipeline_proth.h"
#include "progress_core.h"
#include "dcore.h"
#include "auto_config.h"
#include <omp.h>
#include <fstream>
#include <cstdlib>

std::vector<uint64_t> load_survivors_from_file(const std::string& filename) {
    std::vector<uint64_t> candidates;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file : " << filename << "\n";
        return candidates;
    }

    uint64_t k_val;
    while (file >> k_val) {
        candidates.push_back(k_val);
    }

    file.close();
    std::cout << "[SYSTEM] Loaded " << candidates.size() << " candidates from a file " << filename << ".\n";
    return candidates;
}
struct ProthOptions {
    unsigned n_pow2 = 0;
    uint64_t k_start = 0;
    uint64_t k_end = 0;
    uint64_t pstart = 0;
    uint64_t pmax = 50000000000;
    int workers = 0;       
    int proth_bases = 16;
    int report_step = 25;  
    std::string out_primes = "primes.txt";
    std::string out_sieve = "";  
    std::string in_file = "";      
    bool sieve_only = false;       
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
        else if (arg == "--report" && i + 1 < argc) opt.report_step = std::stoi(argv[++i]);
        else if (arg == "--out-primes" && i + 1 < argc) opt.out_primes = argv[++i];
        else if (arg == "--out-sieve" && i + 1 < argc) opt.out_sieve = argv[++i]; 
        else if (arg == "--in-file" && i + 1 < argc) opt.in_file = argv[++i];
        else if (arg == "--sieve-only") opt.sieve_only = true;                    
        
    }
}

static int run_proth(const ProthOptions& opt) {    
    int final_workers = opt.workers > 0 ? opt.workers : get_auto_workers(opt.n_pow2);

    std::cout << "\n  ProthLogic4.2 CPU & GPUsieve " << std::endl;
    std::cout << "  N=" << opt.n_pow2 << " Range k=[" << opt.k_start << ".." << opt.k_end << "]" << std::endl;
    std::cout << "  Workers: " << final_workers << "  |  Sieve Pmax: " << opt.pmax << "\n" << std::endl;

    std::vector<Task> tasks;
    std::vector<uint64_t> final_survivors;     
    if (!opt.in_file.empty()) {
        std::cout << "Starting Extreme GPU SPARSE Sieve..." << std::endl;
        auto t_sieve_0 = std::chrono::high_resolution_clock::now();
        std::vector<uint64_t> loaded_ks = load_survivors_from_file(opt.in_file);
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
    
    else if (opt.k_start == opt.k_end) {        
        Task t;
        t.label = "Proth";
        t.payload["k"] = std::to_string(opt.k_start);
        t.payload["n"] = std::to_string(opt.n_pow2);
        t.payload["out_primes"] = opt.out_primes;
        t.payload["bases"] = "auto";
        t.payload["report_step"] = std::to_string(opt.report_step);
        tasks.push_back(t);
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
                outfile << k << "\n";
            }
            outfile.close();
            std::cout << "[SIEVE] Saving " << final_survivors.size()
                << " surviving candidates to a file: " << opt.out_sieve << "\n";
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
            final_workers,
            tasks,
            opt.report_step
        );
    }

    if (tasks.empty()) {
        std::cout << "[INFO] No candidates survived the sieve. Finishing." << std::endl;
        return 0;
    }    
    progress_init((uint64_t)tasks.size(), 500, opt.n_pow2, opt.k_start, opt.k_end);    
    std::cout << "\nStarting Proth Test (" << tasks.size() << " candidates to test)" << std::endl;
    auto t_calc_0 = std::chrono::steady_clock::now();    
    dispatch(tasks, final_workers);

    auto t_calc_1 = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(t_calc_1 - t_calc_0).count();

    std::cout << "\nCalculations finished." << std::endl;
    std::cout << "Total computation time: "
        << std::fixed << std::setprecision(2) << total_seconds << " s" << std::endl;

    return 0;
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        std::cout << "Usage: lotr_jeden --n <pow2> --k-start <a> --k-end <b> [--workers W] [--pmax P]\n";
        return 0;
    }
    ProthOptions opt;
    parse_args_proth(argc, argv, opt);        
    if (opt.n_pow2 == 0 || opt.k_start == 0) {
        std::cerr << "[ERROR] Missing required parameters --n and --k-start" << std::endl;
        return 1;
    }    
    if (opt.k_end < opt.k_start) opt.k_end = opt.k_start;

    return run_proth(opt);
}