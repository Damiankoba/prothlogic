// Copyright (C) Damian Koba.
#include "fft_factorizer.h"
#include <vector>
#include <iostream>
#include <algorithm>


FFTFactorPlan generate_fft_plan(size_t n, bool verbose) {
    FFTFactorPlan plan;
    plan.total_n = n;
    plan.has_unsupported_primes = false; 
    size_t temp = n;

    while (temp % 5 == 0) { plan.factors.push_back(5); temp /= 5; }
    while (temp % 3 == 0) { plan.factors.push_back(3); temp /= 3; }
    while (temp % 2 == 0) { plan.factors.push_back(2); temp /= 2; }

    for (size_t d = 7; d * d <= temp; d += 2) {
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
        
        for (size_t f : plan.factors) {
            
        }
        std::cout << "\n";

        if (plan.has_unsupported_primes) {
            std::cout << "  [WARNING] N contains large prime factors (>5). Will require Bluestein's algorithm fallback or specialized kernels.\n";
        }
        
    }

    return plan;
}