// Copyright (C) Damian Koba.
#pragma once
#include <vector>
#include "task.h"
void dispatch(const std::vector<Task>& tasks, int max_concurrent_tasks, int threads_per_task);

// Saves the set of CPUs the process may run on (e.g. from taskset). Call at the very beginning of main(),
// before OpenMP/CUDA pin anything; workers restore this set for themselves.
void capture_process_affinity();