// Copyright (C) Damian Koba.
#pragma once
#include <vector>
#include "task.h"
void dispatch(const std::vector<Task>& tasks, int max_concurrent_tasks, int threads_per_task);