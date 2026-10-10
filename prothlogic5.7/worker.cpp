// Copyright (C) Damian Koba.
#include "worker.h"
#include "proth_logic.h"
#include "fft_engine.h"
#include "log_config.h"
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

        bool is_prime = execute_proth_test_fft(k, n, limbs, ctx, report_step);
        batch_note_done(is_prime);

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

            const bool is_prime = execute_proth_test_fft(k, n, limbs, ctx, report_step);
            batch_note_done(is_prime);
            if (is_prime && !g_batch_agg) {   // with the shared batch line "[!] FOUND PRIME" is enough (it clears the line)
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