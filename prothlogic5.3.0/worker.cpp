// Copyright (C) Damian Koba.
#include "worker.h"
#include "proth_logic.h"
#include "fft_engine.h"
#include <mutex>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>

extern "C" void lotr_progress_tick(long k_now);
std::mutex g_io_mutex;


std::string run_task_with_ctx(const Task& t, FFTContext& ctx, uint64_t* limbs) {
    unsigned n = t.n;
    int report_step = t.report_step;

    
    if (t.type == TaskType::PROTH_SINGLE) {
        uint64_t k = t.k;
        size_t limbs_to_clear = (n / 64) + 1;
        
        bool is_prime = execute_proth_test_fft(k, n, limbs, ctx, report_step);

        
        {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            std::cout << "\n--- FFT ---" << std::endl;
            std::cout << "Pakowanie (Pack): " << ctx.prof_pack << " s" << std::endl;
            std::cout << "FFT Wprost (Fwd) : " << ctx.prof_fwd << " s" << std::endl;
            std::cout << "Mnożenie (Square): " << ctx.prof_square << " s" << std::endl;
            std::cout << "FFT Odwrotne (Inv): " << ctx.prof_inv << " s" << std::endl;
            std::cout << "Rozpakowywanie (Unpack): " << ctx.prof_unpack << " s" << std::endl;
        }
        

        if (is_prime) {
            std::lock_guard<std::mutex> lock(g_io_mutex);
            return "prime";
        }
        return "PRP";
    }

    
    if (t.type == TaskType::PROTH_BATCH) {
        size_t limbs_to_clear = (n / 64) + 1;

        for (uint64_t k : t.k_list) {
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

