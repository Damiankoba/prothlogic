#pragma once

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
extern bool g_batch_agg;
extern bool g_stdout_tty;
extern unsigned long long g_batch_total;

// Batch progress line (proth_logic.cpp): start the timer and report a finished candidate (worker.cpp).
void batch_progress_start();
void batch_note_done(bool prime);
