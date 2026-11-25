// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef CHECKPOINT_MANAGER_H
#define CHECKPOINT_MANAGER_H

#include <cstdint>
#include <cstddef>
#include <string>

struct CheckpointHeader {
    uint64_t current_i; 
    size_t n_limbs;     
};

bool save_checkpoint(uint64_t i, size_t n_limbs, uint64_t* data, const std::string& filename);
bool load_checkpoint(uint64_t& out_i, size_t n_limbs, uint64_t* data, const std::string& filename);
// Iteration stored in the checkpoint header (0 = no file / unreadable); for the batch ETA
uint64_t checkpoint_iteration(const std::string& filename);
// Test time of the earlier runs: "<checkpoint>.time" next to the checkpoint, the seconds as text (written with each
// checkpoint, removed with it by report_result, can be set by hand); 0 = no file / unreadable
double checkpoint_seconds(const std::string& filename);
void save_checkpoint_seconds(const std::string& filename, double seconds);

#endif