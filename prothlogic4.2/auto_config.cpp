// Copyright (C) Damian Koba.

#include "auto_config.h"
#include <thread>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <cstdlib>

static size_t calculate_fft_len(unsigned N) {
    size_t fft_len = 16384;
    while (true) {
        double bits_per_word = (double)N / (double)fft_len;
        if (bits_per_word <= 13.8) break;
        fft_len *= 2;
    }
    return fft_len;
}

static int calculate_optimal_workers(unsigned n_pow2) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    int optimal_workers = (hw >= 2) ? (hw - 1) : 1;

    size_t fft_len = calculate_fft_len(n_pow2);
    size_t task_bytes = (fft_len + 64) * sizeof(double) * 18 + (fft_len * sizeof(uint64_t));
    task_bytes = (task_bytes * 125) / 100;

#if defined(__linux__) || defined(linux)
    std::set<std::string> unique_l3_domains;
    size_t l3_size_per_domain = 0;

    for (unsigned i = 0; i < hw; ++i) {
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cache/index3/";
        std::ifstream f_shared(path + "shared_cpu_list");
        std::ifstream f_size(path + "size");

        if (f_shared.is_open() && f_size.is_open()) {
            std::string shared_list;
            f_shared >> shared_list;
            unique_l3_domains.insert(shared_list);

            if (l3_size_per_domain == 0) {
                std::string size_str;
                f_size >> size_str;
                size_t val = std::stoull(size_str);
                if (size_str.back() == 'K') l3_size_per_domain = val * 1024;
                else if (size_str.back() == 'M') l3_size_per_domain = val * 1024 * 1024;
                else l3_size_per_domain = val;
            }
        }
    }

    if (!unique_l3_domains.empty() && l3_size_per_domain > 0) {
        size_t num_domains = unique_l3_domains.size();
        size_t total_l3_cache = num_domains * l3_size_per_domain;
        size_t possible_workers = total_l3_cache / task_bytes;

        size_t task_mb = task_bytes / 1024 / 1024;
        if (task_mb >= 16) {
            size_t bandwidth_limit = 24;
            if (possible_workers > bandwidth_limit) {
                possible_workers = bandwidth_limit;
            }
        }

        if (possible_workers == 0) possible_workers = 1;
        if (possible_workers > hw) possible_workers = hw;

        optimal_workers = possible_workers;

        if (task_mb >= 16) {
            std::cout << "[HARDWARE RADAR] Bandwidth Limit for proth test on CPU\n";
            std::cout << "[HARDWARE RADAR] For a number of this size, the memory bandwidth limit allows the Proth test to run safely on " << optimal_workers << " threads.\n";
        }
        else {
            std::cout << "[HARDWARE RADAR] For a number of this size, the memory bandwidth limit allows the Proth test to run safely on "
                << optimal_workers << " of " << hw << " threads.\n";
        }
    }
#endif

    return optimal_workers;
}

int get_auto_workers(unsigned n_pow2) {
    const char* wenv = std::getenv("LOTR_WORKERS");
    int w_from_env = (wenv ? std::max(1, std::atoi(wenv)) : 0);

    if (w_from_env > 0) {
        return w_from_env;
    }

    return calculate_optimal_workers(n_pow2);
}