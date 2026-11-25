// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
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

    int report_step;
};