#pragma once
#include <cstdint>
#include <cstddef>
#include "fft_engine.h" 

void pack_karatsuba_limbs_vbmi(const uint64_t* in, size_t n_limbs, FFTContext& ctx, int target_bits);
void unpack_karatsuba_overlap_add_vbmi(uint64_t* out, size_t n_limbs, FFTContext& ctx, unsigned n_bits, uint64_t k_val, int target_bits);