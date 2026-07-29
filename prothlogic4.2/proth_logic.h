// Copyright (C) Damian Koba.

#pragma once

#ifndef PROTH_LOGIC_H
#define PROTH_LOGIC_H

#include <gmpxx.h>
#include <cstdint>
#include "fft_engine.h"

bool execute_proth_test_fft(uint64_t k, unsigned n, uint64_t* limbs, FFTContext& ctx, int report_step);

#endif