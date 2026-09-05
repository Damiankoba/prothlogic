// Copyright (C) Damian Koba.

#include "dcore.h"
#include "worker.h"
#include "fft_engine.h"
#include "fft_factorizer.h"
#include <thread>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <pthread.h>  
#include <sched.h>    
#include <sys/mman.h> 
#include <omp.h>

static size_t get_next_smooth_size(size_t current) {
    size_t p = 256;
    while (p <= current) {
        p *= 2;
    }
    return p;
}


size_t auto_tune_fft_len(unsigned actual_n, uint64_t sample_k) {
    std::cout << "\n Calibration for N=" << actual_n << "...\n";
    omp_set_num_threads(1);
    size_t n_limbs = (actual_n + 63) / 64;

    size_t candidate_len = 256;
    
    while (candidate_len * 16 < actual_n) {
        candidate_len = get_next_smooth_size(candidate_len);
    }

        while (true) {
        double bits_per_word = (double)actual_n / (double)candidate_len;

        if (bits_per_word < 11.0) { 
            std::cout << " Selected: " << candidate_len << "\n\n";
            return candidate_len;
        }

       
        if (bits_per_word > 17.0) { 
            candidate_len = get_next_smooth_size(candidate_len);
            continue; 
        }
      

        std::cout << "  -> Karatsuba tests for FFT_Len: " << candidate_len << " (Bits/word: " << bits_per_word << ")...\n";

        
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
            std::cout << "  Max Err after 100 iterations = " << max_err_observed << "\n";

            if (max_err_observed >= 0.0 && max_err_observed <= 0.45) {
                std::cout << "  FFT Pass: " << candidate_len << "\n";
                return candidate_len;
            }
            else {
                std::cout << "     [Fail] The error is too high. Shifting up a gear...\n";
            }
        }
        else {
            fft_destroy(test_ctx);
        }

       
        size_t next_len = get_next_smooth_size(candidate_len);

        
        if (next_len <= candidate_len) {
            next_len = candidate_len * 2;
        }

        candidate_len = next_len;
        }
}


void dispatch(const std::vector<Task>& tasks, int max_concurrent_tasks, int threads_per_task) {
    if (tasks.empty()) return;
    if (max_concurrent_tasks <= 0) max_concurrent_tasks = 1;
    if (threads_per_task <= 0) threads_per_task = 1;

    unsigned actual_n = tasks[0].n;
    uint64_t sample_k = 3;
    if (tasks[0].type == TaskType::PROTH_SINGLE) {
        sample_k = tasks[0].k;
    }
    else if (tasks[0].type == TaskType::PROTH_BATCH && !tasks[0].k_list.empty()) {
        sample_k = tasks[0].k_list[0];
    }

    size_t needed_fft_len = auto_tune_fft_len(actual_n, sample_k);

    
    FFTFactorPlan fft_plan = generate_fft_plan(needed_fft_len, true);
    if (fft_plan.has_unsupported_primes) {
        std::cout << "[MIXED-RADIX] Note: Factors other than 2, 3, 5 detected. Bluestein support required.\n";
    }
    
       
    struct Item { size_t idx; Task task; };
    std::queue<Item> q;
    std::mutex m;
    std::condition_variable cv;
    bool enqueue_done = false;

    std::vector<std::string> results(tasks.size());
    {
        std::lock_guard<std::mutex> lk(m);
        for (size_t i = 0; i < tasks.size(); ++i) {
            Task t = tasks[i];
            q.push(Item{ i, t });
        }
        enqueue_done = true;
    }
    cv.notify_all();

    auto worker_fn = [&](int worker_id) {
        
        omp_set_num_threads(threads_per_task);

        if (threads_per_task == 1) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(worker_id, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            int core_to_pin = worker_id * 8;
            CPU_SET(core_to_pin, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }

        FFTContext local_ctx;
        if (!fft_init(local_ctx, FFTBackend::INTERNAL, needed_fft_len)) {
            std::lock_guard<std::mutex> lk(m);
            std::cerr << "[ERROR] FFT init failed for len " << needed_fft_len << std::endl;
            return;
        }

        size_t safe_alloc_limbs = std::max(needed_fft_len, (size_t)((2 * actual_n) / 64 + 64));
        size_t limbs_bytes_size = safe_alloc_limbs * sizeof(uint64_t);

        uint64_t* local_limbs = (uint64_t*)mmap(NULL, limbs_bytes_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (local_limbs == MAP_FAILED) {
            std::lock_guard<std::mutex> lk(m);
            std::cerr << "[ERROR] mmap failed for local_limbs" << std::endl;
            fft_destroy(local_ctx);
            return;
        }

        std::memset(local_limbs, 0, limbs_bytes_size);

        for (;;) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return !q.empty() || enqueue_done; });

                if (q.empty() && enqueue_done) {
                    munmap(local_limbs, limbs_bytes_size);
                    fft_destroy(local_ctx);
                    return;
                }
                it = q.front();
                q.pop();
            }

            size_t limbs_to_clear = (2 * actual_n) / 64 + 10;
            std::memset(local_limbs, 0, limbs_to_clear * sizeof(uint64_t));
                 
            std::string r = run_task_with_ctx(it.task, local_ctx, local_limbs);
            results[it.idx] = std::move(r);

            cv.notify_one();
        }
    };
  

    std::vector<std::thread> pool;
    int active_workers = std::min((size_t)max_concurrent_tasks, tasks.size());

    pool.reserve((size_t)active_workers);
    for (int i = 0; i < active_workers; ++i) {
        pool.emplace_back(worker_fn, i);
    }

    for (auto& th : pool) th.join();
}