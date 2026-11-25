// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <string>

// Output settings (defined in cli.cpp):
//   g_detail        --detail: FFT calibration, chosen engine, [ESTIMATE], [SQ_REAL] self-test, Max Err ([STEP] every 25 %)
//   g_progress      single progress line (off when --report N is given - then the classic [STEP] lines for benchmarks)
//   g_progress_tty  progress line refreshed in place (terminal and one candidate at a time); otherwise plain lines every 10 %
extern bool g_detail;
extern bool g_progress;
extern bool g_progress_tty;
//   g_batch_agg     batch mode (several workers at once, many candidates): one shared batch progress line instead of
//                   per-candidate lines; "not prime" results only with --detail (they always go to composite.txt)
//   g_stdout_tty    stdout is a terminal (line refreshed in place); otherwise plain lines
//   g_batch_total   number of candidates in the batch
//   g_batch_total_iters  squarings left in the whole batch (n - 1 per candidate minus the checkpoint): the batch ETA
//                        counts the running tests too
//   g_batch_full_iters   n - 1 per candidate (checkpoints not subtracted): total ETA of the one-at-a-time line
extern bool g_batch_agg;
extern bool g_stdout_tty;
extern unsigned long long g_batch_total;
extern unsigned long long g_batch_total_iters;
extern unsigned long long g_batch_full_iters;
//   g_primes_file   where primes are written (--out-primes, default primes.txt); composites always go to composite.txt
extern std::string g_primes_file;

// Batch progress line (proth_logic.cpp): start the timer and report a finished candidate (worker.cpp).
void batch_progress_start();
void batch_note_done(bool prime, unsigned n);   // n: the candidate's exponent (squarings n - 1, for the total ETA)
