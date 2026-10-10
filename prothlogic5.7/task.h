// Copyright (C) Damian Koba.
#pragma once
#include <string>
#include <vector>
#include <cstdint>

enum class TaskType {
    PROTH_SINGLE,
    PROTH_BATCH
};

struct Task {
    TaskType type;
    unsigned int n;

    
    uint64_t k;                       // Using with PROTH_SINGLE
    std::vector<uint64_t> k_list;     // Using with PROTH_BATCH

    std::string out_primes;
    int report_step;
};