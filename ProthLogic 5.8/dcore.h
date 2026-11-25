// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <vector>
#include "task.h"
// pin = --pin: each worker and its OpenMP threads on their own CPUs (a worker stays in one CCD when it fits)
// gpu = --gpu: the NVIDIA card is one more worker on the same queue (execute_proth_test_gpu)
// gpu_only = --gpu-only: only the card (no CPU workers, no CPU calibration); candidates it cannot test stay untested
void dispatch(const std::vector<Task>& tasks, int max_concurrent_tasks, int threads_per_task, bool pin = false,
              bool gpu = false, bool gpu_only = false);

// --tune: measures W candidates at once x T threads for the CPUs of this process and prints the best options;
// all = --tune-all: also the splits that leave cores free (evenly per CCD)
int tune_configs(const std::vector<Task>& tasks, double secs, bool all = false);

// Saves the set of CPUs the process may run on (e.g. from taskset). Call at the very beginning of main(),
// before OpenMP/CUDA pin anything; workers restore this set for themselves.
void capture_process_affinity();