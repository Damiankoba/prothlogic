// Copyright (C) Damian Koba.
#ifndef FFT_FACTORIZER_H
#define FFT_FACTORIZER_H
#include <cstddef>
#include "fft_types.h" 

FFTFactorPlan generate_fft_plan(size_t n, bool verbose = false);

#endif