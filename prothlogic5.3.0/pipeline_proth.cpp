// Copyright (C) Damian Koba.


#include "pipeline_proth.h"
#include "checkpoint_manager.h" 
#include <iostream>
#include <algorithm>
#include <thread>



static std::string join_k_list(const std::vector<uint64_t>& ks, size_t start, size_t end) {
    std::string res;
    for (size_t i = start; i < end; ++i) {
        if (i > start) res += ",";
        res += std::to_string(ks[i]);
    }
    return res;
}

void build_proth_ff_tasks_for_range(
    unsigned n_pow2,
    const std::vector<uint64_t>& surviving_ks,
    const std::string& out_primes,
    int actual_workers,
    std::vector<Task>& out_tasks,
    int report_step)
{
    if (surviving_ks.empty()) return;

    // 1. LOGISTIC
    int workers = actual_workers > 0 ? actual_workers : (int)std::thread::hardware_concurrency();
    if (workers <= 0) workers = 7;

    size_t target_batches = (size_t)(workers * 8);
    size_t min_chunk = 1;
    auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };
    size_t chunksize = std::max<size_t>(min_chunk, ceil_div(surviving_ks.size(), target_batches));

    // 2. Generating tasks
    // If k is small, we send single tasks
    if (surviving_ks.size() <= (size_t)workers) {
        out_tasks.reserve(out_tasks.size() + surviving_ks.size());
        for (uint64_t k : surviving_ks) {
            Task t;
            t.type = TaskType::PROTH_SINGLE;
            t.n = n_pow2;
            t.k = k;
            t.out_primes = out_primes;
            t.report_step = report_step;
            out_tasks.push_back(std::move(t));
        }
        return;
    }

    // If k is large, we create a BATCH
    out_tasks.reserve(out_tasks.size() + ceil_div(surviving_ks.size(), chunksize));
    for (size_t i = 0; i < surviving_ks.size(); i += chunksize) {
        size_t j = std::min(i + chunksize, surviving_ks.size());
        Task t;
        t.type = TaskType::PROTH_BATCH;
        t.n = n_pow2;
        // Quickly assign a vector slice
        t.k_list.assign(surviving_ks.begin() + i, surviving_ks.begin() + j);
        t.out_primes = out_primes;
        t.report_step = report_step;
        out_tasks.push_back(std::move(t));
    }
}