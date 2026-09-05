// Copyright (C) Damian Koba.

#ifndef LLR_FFT_PACK_H
#define LLR_FFT_PACK_H

#include "fft_types.h"
#include <cstdint>
#include <cstddef>


void pack_karatsuba_limbs(const uint64_t* in, size_t n_limbs, FFTContext& ctx);

#endif