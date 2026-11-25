// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "fft_factorizer.h"
#include <vector>
#include <iostream>


FFTFactorPlan generate_fft_plan(size_t n, bool verbose) {
    FFTFactorPlan plan;
    plan.total_n = n;
    plan.has_unsupported_primes = false; 
    size_t temp = n;

    while (temp % 7 == 0) { plan.factors.push_back(7); temp /= 7; }   // radix-7 batch-8 columns (7*2^p)
    while (temp % 5 == 0) { plan.factors.push_back(5); temp /= 5; }
    while (temp % 3 == 0) { plan.factors.push_back(3); temp /= 3; }
    while (temp % 2 == 0) { plan.factors.push_back(2); temp /= 2; }

    for (size_t d = 11; d * d <= temp; d += 2) {
        while (temp % d == 0) {
            plan.factors.push_back(d);
            temp /= d;
            plan.has_unsupported_primes = true;
        }
    }
    if (temp > 1) {
        plan.factors.push_back(temp);
        plan.has_unsupported_primes = true;
    }

    
    if (verbose) {
        

        if (plan.has_unsupported_primes) {
            std::cout << "  [WARNING] N contains large prime factors (>5). Will require Bluestein's algorithm fallback or specialized kernels.\n";
        }
        
    }

    return plan;
}