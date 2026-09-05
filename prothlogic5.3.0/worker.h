// Copyright (C) Damian Koba.
#pragma once
#include "task.h"
#include "fft_engine.h"
#include <string>

std::string run_task_with_ctx(const Task& t, FFTContext& ctx, uint64_t* limbs);