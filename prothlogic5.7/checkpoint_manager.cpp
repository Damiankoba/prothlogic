// Copyright (C) Damian Koba.

#include "checkpoint_manager.h"
#include <cstdio>
#include <iostream>

bool save_checkpoint(uint64_t i, size_t n_limbs, uint64_t* data, const std::string& filename) {
    std::string tmp_filename = filename + ".tmp";
    FILE* f = fopen(tmp_filename.c_str(), "wb");
    if (!f) return false;

    CheckpointHeader header = { i, n_limbs };

    // Make sure the write succeeded before the old file is replaced
    size_t w_head = fwrite(&header, sizeof(CheckpointHeader), 1, f);
    size_t w_data = fwrite(data, sizeof(uint64_t), n_limbs, f);

    if (w_head != 1 || w_data != n_limbs) {
        fclose(f);
        remove(tmp_filename.c_str());
        std::cerr << "[ERROR] Checkpoint write failed (Disk full or I/O error). Aborting save to protect old DB.\n";
        return false;
    }

    // No fsync() (expensive, and wears flash drives): the OS writes the buffer asynchronously, so the worker
    // threads are not blocked. Write errors can still show up in fflush/fclose (e.g. disk full) and in rename.
    const int flush_rc = fflush(f);
    const int close_rc = fclose(f);
    if (flush_rc != 0 || close_rc != 0) {
        remove(tmp_filename.c_str());
        std::cerr << "[ERROR] Checkpoint flush/close failed (Disk full or I/O error). Old checkpoint kept.\n";
        return false;
    }

    if (std::rename(tmp_filename.c_str(), filename.c_str()) != 0) {
        remove(tmp_filename.c_str());
        std::cerr << "[ERROR] Checkpoint rename failed: " << filename << ". Old checkpoint kept.\n";
        return false;
    }
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