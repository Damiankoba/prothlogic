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

    // 1. LOGISTYKA: Ile wątków i jak duże paczki
    int workers = actual_workers > 0 ? actual_workers : (int)std::thread::hardware_concurrency();
    if (workers <= 0) workers = 7;

    size_t target_batches = (size_t)(workers * 8);
    size_t min_chunk = 1;
    auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };
    size_t chunksize = std::max<size_t>(min_chunk, ceil_div(surviving_ks.size(), target_batches));

    // 2. GENEROWANIE ZADAŃ
    // Jeśli k jest mało, wysyłamy pojedyncze zadania
    if (surviving_ks.size() <= (size_t)workers) {
        out_tasks.reserve(out_tasks.size() + surviving_ks.size());
        for (uint64_t k : surviving_ks) {
            Task t;
            t.n = n_pow2;
            t.k = k;
            t.label = "Proth";
            t.payload = {
                {"n", std::to_string(n_pow2)},
                {"k", std::to_string(k)},
                {"out_primes", out_primes},
                {"report_step", std::to_string(report_step)} 
            };
            out_tasks.push_back(std::move(t));
        }
        return;
    }

    // Jeśli k jest dużo, tworzymy BATCHE
    out_tasks.reserve(out_tasks.size() + ceil_div(surviving_ks.size(), chunksize));
    for (size_t i = 0; i < surviving_ks.size(); i += chunksize) {
        size_t j = std::min(i + chunksize, surviving_ks.size());
        Task t;
        t.n = n_pow2;
        t.k = surviving_ks[i];
        t.label = "PROTH_FF_BATCH";
        t.payload = {
            {"n", std::to_string(n_pow2)},
            {"k_list", join_k_list(surviving_ks, i, j)},
            {"out_primes", out_primes},
            {"report_step", std::to_string(report_step)} 
        };
        out_tasks.push_back(std::move(t));
    }
}