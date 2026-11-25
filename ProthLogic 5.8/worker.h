// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "task.h"
#include "fft_engine.h"
#include <string>

std::string run_task_with_ctx(const Task& t, FFTContext& ctx, uint64_t* limbs);