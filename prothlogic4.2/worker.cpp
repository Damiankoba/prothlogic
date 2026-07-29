// Copyright (C) Damian Koba.

#include "worker.h"
#include "proth_logic.h"
#include "fft_engine.h"
#include <mutex>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstring>


extern "C" void lotr_progress_tick(long k_now);
std::mutex g_io_mutex;
static std::vector<uint64_t> parse_k_list(const std::string& s) {
    std::vector<uint64_t> out;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoull(tok));
    }
    return out;
}

// GŁÓWNA FUNKCJA (Dostosowana do dcore.cpp)
std::string run_task_with_ctx(const Task& t, FFTContext& ctx, uint64_t* limbs) {    
    unsigned n = 0;
    int report_step = 25; 

    if (t.payload.count("n")) {
        n = (unsigned)std::stoul(t.payload.at("n"));
    }   
    if (t.payload.count("report_step")) {
        report_step = std::stoi(t.payload.at("report_step"));
    }

   
    if (t.label == "Proth") {
        uint64_t k = (uint64_t)std::stoull(t.payload.at("k"));
        size_t limbs_to_clear = (n / 64) + 1;        
        bool is_prime = execute_proth_test_fft(k, n, limbs, ctx, report_step);
        if (is_prime) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            return "prime";
        }
        return "PRP";
    }

    // 2. OBSŁUGA ZADANIA BATCH (Dla dużych zakresów z sita)
    if (t.label == "PROTH_FF_BATCH") {
        std::vector<uint64_t> ks = parse_k_list(t.payload.at("k_list"));
        size_t limbs_to_clear = (n / 64) + 1;

        for (auto k : ks) {
            std::memset(limbs, 0, limbs_to_clear * sizeof(uint64_t));

            
            if (execute_proth_test_fft(k, n, limbs, ctx, report_step)) {
                std::lock_guard<std::mutex> lock(g_io_mutex);
                std::cout << "[BATCH-RESULT] PRIME FOUND: k=" << k << "\n";
            }           
            if ((k & 1023) == 0)
                lotr_progress_tick((long)k);
        }
        return "BATCH";
    }

    return "fail";
}