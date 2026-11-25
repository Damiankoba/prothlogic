// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once
#include <vector>
#include <cstdint>


std::vector<uint64_t> execute_full_streaming_sieve(
    uint64_t k_start,
    uint64_t k_end_inclusive,
    unsigned n_pow2,
    uint64_t pmax
);

std::vector<uint64_t> execute_sparse_streaming_sieve(
    const std::vector<uint64_t>& input_ks,
    unsigned n_pow2,
    uint64_t pstart,
    uint64_t pmax
);

std::vector<uint32_t> execute_n_range_streaming_sieve(
    uint64_t k_val,
    uint32_t n_start,
    uint32_t n_end_inclusive,
    uint64_t pstart,
    uint64_t pmax,
    const std::vector<uint32_t>& input_ns = {}
);