// Copyright (C) Damian Koba.

#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <gmpxx.h>
#include "prime_candidate.h"

void append_line_txt(const std::string& filename, 
                    const std::string& line);

void save_proth_result_csv(const std::string& filename,
    unsigned n_pow2, 
    uint64_t k, 
    const mpz_t N,
    uint64_t witness,
    int tried);


        