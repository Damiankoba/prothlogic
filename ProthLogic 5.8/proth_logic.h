// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#ifndef PROTH_LOGIC_H
#define PROTH_LOGIC_H

#include <gmpxx.h>
#include <cstdint>
#include "fft_engine.h"

bool execute_proth_test_fft(uint64_t k, unsigned n, uint64_t* limbs, FFTContext& ctx, int report_step);
// --gpu: the same test on CUDA device `device` (gpu_proth_engine.cu); 1 = prime, 0 = not prime, -1 = the card failed
// (no result written, checkpoint kept - the caller hands the candidate to a CPU worker), -2 = no plan for the card
// (the candidate to the CPU, the card goes on)
int execute_proth_test_gpu(uint64_t k, unsigned n, int device, int report_step);

// --gpu, the tail of a batch: the card is free and the queue empty, while a CPU worker still tests a
// candidate - the card takes it over. dispatch asks with cpu_run_request; execute_proth_test_fft stops after its next
// checkpoint (or before its first segment) and returns with cpu_handed_over() == true (no result, the checkpoint and its
// .time kept); the card resumes from that checkpoint, so the result's Time is the CPU part + the card part.
unsigned long long cpu_run_iter(uint64_t k, unsigned n);   // the last checkpoint of a running CPU test (0: none yet)
void cpu_run_request(uint64_t k, unsigned n);              // ask that CPU test to hand its candidate over
void cpu_run_clear(uint64_t k, unsigned n);                // drop the entry (a request that came too late)
bool cpu_handed_over();                                    // this thread's last execute_proth_test_fft handed over

#endif