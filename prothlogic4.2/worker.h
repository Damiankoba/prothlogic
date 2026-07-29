// Copyright (C) Damian Koba.

#pragma once
#include "task.h"
#include "fft_engine.h" 
#include <string>

extern bool PROTH_QUIET;

std::string run_task(const Task& task);

std::string run_task_with_ctx(const Task& t, FFTContext& ctx, uint64_t* limbs);