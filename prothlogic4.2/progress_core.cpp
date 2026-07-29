// Copyright (C) Damian Koba.

#include "progress_core.h"
#include <atomic>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <string>

static std::atomic<uint64_t> g_done{0};
static std::atomic<uint64_t> g_total{0};
static std::atomic<uint64_t> g_k_now{0};
static int g_step = 200; 
static std::atomic<bool>     g_final_printed{false};
static std::atomic<uint64_t> g_last_bucket{~uint64_t(0)}; 

static uint64_t g_n_pow2 = 0;
static uint64_t g_k_start = 0;
static uint64_t g_k_end = 0;
static std::chrono::steady_clock::time_point g_t0;

void set_progress_step(int step) {
    if (step > 0) g_step = step;
}
int get_progress_step() { return g_step; }

void progress_init_range(uint64_t n_pow2, uint64_t k_start, uint64_t k_end, int step) {
    g_n_pow2  = n_pow2;
    g_k_start = k_start;
    g_k_end   = k_end;
    if (step > 0) g_step = step;
    g_t0 = std::chrono::steady_clock::now();
    g_k_now = k_start;

    g_last_bucket   = ~uint64_t(0);
    g_final_printed = false;
}

void progress_set_total(uint64_t total) {
    g_total = total;
    g_done  = 0;
    g_t0 = std::chrono::steady_clock::now();

    g_last_bucket   = ~uint64_t(0);
    g_final_printed = false;
}

void progress_init(uint64_t total, int step, uint64_t n_pow2,
                   uint64_t k_start, uint64_t k_end) {
    progress_init_range(n_pow2, k_start, k_end, step);
    progress_set_total(total);
}

void progress_done_one() {
    (void)++g_done;
}

ProgressSnapshot progress_snapshot() {
    ProgressSnapshot s;
    s.done    = g_done.load();
    s.total   = g_total.load();
    s.n_pow2  = g_n_pow2;
    s.k_start = g_k_start;
    s.k_end   = g_k_end;
    s.k_now   = g_k_now.load();
    
    double pct = 0.0;
    if (s.k_end >= s.k_start) {
        uint64_t span = s.k_end - s.k_start + 1;
        uint64_t pos  = 0;
        if (s.k_now >= s.k_start)
            pos = (s.k_now > s.k_end ? s.k_end : s.k_now) - s.k_start + 1;
        pct = 100.0 * double(pos) / double(span);
    }
    if (pct < 0.0)   pct = 0.0;
    if (pct > 100.0) pct = 100.0;
    s.percent = pct;

    s.eta_seconds = 0.0; // ETA off
    return s;
}

std::string snapshot_to_string(const ProgressSnapshot& s_in) {
    ProgressSnapshot s = s_in;   
    double pct_k = 0.0;
    if (s.k_end >= s.k_start) {
        uint64_t span   = s.k_end - s.k_start + 1;
        uint64_t kclamp = (s.k_now > s.k_end ? s.k_end :
                           (s.k_now < s.k_start ? s.k_start : s.k_now));
        uint64_t pos    = kclamp - s.k_start + 1;
        pct_k = 100.0 * double(pos) / double(span);
        if (pct_k > 100.0) pct_k = 100.0;
        if (pct_k < 0.0)   pct_k = 0.0;
    }

    std::ostringstream oss;
    oss << "[PROGRESS] "
        << "n=" << s.n_pow2
        << " k=" << s.k_now
        << " (" << std::fixed << std::setprecision(2) << pct_k << "%)";
    return oss.str();
}
extern "C" void lotr_progress_tick(long k_now) {
    if (k_now >= 0) g_k_now = static_cast<uint64_t>(k_now);
}

std::string progress_message() {
    auto s = progress_snapshot();
    if (s.total == 0) return {};

    uint64_t bucket = (s.done >= s.total)
                      ? (uint64_t)g_step
                      : (uint64_t)((__int128)s.done * g_step / std::max<uint64_t>(s.total,1));

    if (s.done >= s.total) {
        bool first = !g_final_printed.exchange(true);
        if (!first) return {};        
        return snapshot_to_string(s);
    }

    uint64_t last = g_last_bucket.load();
    if (bucket == last) return {};
    g_last_bucket.store(bucket);
    return snapshot_to_string(s);
}
