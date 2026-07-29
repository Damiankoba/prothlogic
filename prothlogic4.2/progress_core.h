// Copyright (C) Damian Koba.

#pragma once
#include <cstdint>
#include <string>
#include <atomic>
#include <chrono>

struct ProgressSnapshot {
    uint64_t done = 0;
    uint64_t total = 0;
    uint64_t n_pow2 = 0;
    uint64_t k_start = 0;
    uint64_t k_end = 0;
    uint64_t k_now = 0;
    double   percent = 0.0;
    double   eta_seconds = 0.0; 
};

void set_progress_step(int step);
int  get_progress_step();
void progress_init_range(uint64_t n_pow2, uint64_t k_start, uint64_t k_end, int step);
void progress_set_total(uint64_t total);
void progress_init(uint64_t total, int step, uint64_t n_pow2, uint64_t k_start, uint64_t k_end);
void progress_done_one();
ProgressSnapshot progress_snapshot();
std::string snapshot_to_string(const ProgressSnapshot& s);
extern "C" void lotr_progress_tick(long k_now);
std::string progress_message();
