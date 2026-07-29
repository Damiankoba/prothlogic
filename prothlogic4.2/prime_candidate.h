// Copyright (C) Damian Koba.

#pragma once
#include <gmpxx.h>
#include <cstdint>

struct PrimeCandidate {
    mpz_class N;      // pełna liczba pierwsza
    uint64_t k;       // k z Protha
    uint32_t n;       // n z Protha
    uint64_t a;       // świadek Protha
};