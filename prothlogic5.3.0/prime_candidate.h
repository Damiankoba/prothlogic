// Copyright (C) Damian Koba.

#pragma once
#include <gmpxx.h>
#include <cstdint>

struct PrimeCandidate {
    mpz_class N;      
    uint64_t k;       // k  Proth
    uint32_t n;       // n  Proth
    uint64_t a;       
};