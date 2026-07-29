// Copyright (C) Damian Koba.

#include "checkpoint_manager.h"
#include <cstdio>
#include <iostream>
#include <unistd.h>

bool save_checkpoint(uint64_t i, size_t n_limbs, uint64_t* data, const std::string& filename) {
    std::string tmp_filename = filename + ".tmp";
    FILE* f = fopen(tmp_filename.c_str(), "wb");
    if (!f) return false;

    CheckpointHeader header = { i, n_limbs };
    fwrite(&header, sizeof(CheckpointHeader), 1, f);
    fwrite(data, sizeof(uint64_t), n_limbs, f); 

    fflush(f);
    fsync(fileno(f));
    fclose(f);

    std::rename(tmp_filename.c_str(), filename.c_str());
    return true;
}

bool load_checkpoint(uint64_t& out_i, size_t n_limbs, uint64_t* data, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;

    CheckpointHeader header;
    if (fread(&header, sizeof(CheckpointHeader), 1, f) != 1) {
        fclose(f); return false;
    }

    if (header.n_limbs != n_limbs) {
        fclose(f); return false;
    }

    if (fread(data, sizeof(uint64_t), n_limbs, f) != n_limbs) {
        fclose(f); return false;
    }

    out_i = header.current_i;
    fclose(f);
    return true;
}