// Copyright (C) Damian Koba.
#pragma once
#include <vector>
#include "task.h"
void dispatch(const std::vector<Task>& tasks, int workers = 0);