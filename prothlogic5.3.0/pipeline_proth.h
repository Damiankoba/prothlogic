// Copyright (C) Damian Koba.

#ifndef PIPELINE_PROTH_H
#define PIPELINE_PROTH_H

#include <vector>
#include <cstdint>
#include <string>
#include "task.h"

void build_proth_ff_tasks_for_range(
    unsigned n_pow2,
    const std::vector<uint64_t>& surviving_ks,
    const std::string& out_primes,
    int actual_workers,
    std::vector<Task>& out_tasks,
    int report_step);

#endif