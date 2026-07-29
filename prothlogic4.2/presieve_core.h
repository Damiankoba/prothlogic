// Copyright (C) Damian Koba.

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
    uint64_t pmax);