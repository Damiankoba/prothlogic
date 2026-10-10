// Copyright (C) Damian Koba.

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

#endif