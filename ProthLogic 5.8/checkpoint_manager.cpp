// Copyright (C) Damian Koba.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "checkpoint_manager.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

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

    // fsync() only for large checkpoints (>= 1 MiB, n >= ~8.4M): without it a host crash right after the rename can
    // leave a 0-byte checkpoint (delayed allocation: the rename reaches the disk before the data). Large checkpoints
    // are written minutes apart, so the fsync costs nothing; the small ones (many short candidates per second on all
    // workers) skip it, a lost short test is cheap to redo.
    const bool durable = n_limbs * sizeof(uint64_t) >= (size_t(1) << 20);
    const int flush_rc = fflush(f);
    const int sync_rc = (durable && flush_rc == 0) ? fsync(fileno(f)) : 0;
    const int close_rc = fclose(f);
    if (flush_rc != 0 || sync_rc != 0 || close_rc != 0) {
        remove(tmp_filename.c_str());
        std::cerr << "[ERROR] Checkpoint flush/sync/close failed (Disk full or I/O error). Old checkpoint kept.\n";
        return false;
    }

    if (std::rename(tmp_filename.c_str(), filename.c_str()) != 0) {
        remove(tmp_filename.c_str());
        std::cerr << "[ERROR] Checkpoint rename failed: " << filename << ". Old checkpoint kept.\n";
        return false;
    }
    if (durable) {   // the rename itself is durable only after the directory is synced
        const size_t slash = filename.find_last_of('/');
        const std::string dir = (slash == std::string::npos) ? "." : filename.substr(0, slash + 1);
        const int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) { fsync(dfd); close(dfd); }
    }
    return true;
}

// A checkpoint that exists but cannot be read (e.g. 0 bytes after a host crash): say so loudly and keep it as
// "<name>.bad", instead of silently starting the test from the beginning over it.
static void report_bad_checkpoint(const std::string& filename, const char* why) {
    const std::string bad = filename + ".bad";
    const bool moved = (std::rename(filename.c_str(), bad.c_str()) == 0);
    std::cerr << "[CHECKPOINT] WARNING: " << filename << " is unreadable (" << why << ") - "
              << (moved ? "moved to " + bad + ", " : std::string()) << "the test starts from the beginning."
              << " If you have a copy of the checkpoint, stop the program and put the copy back.\n";
}

bool load_checkpoint(uint64_t& out_i, size_t n_limbs, uint64_t* data, const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return false;

    CheckpointHeader header;
    if (fread(&header, sizeof(CheckpointHeader), 1, f) != 1) {
        fclose(f); report_bad_checkpoint(filename, "no header"); return false;
    }

    // The saved value (a residue < N) does not depend on the FFT length; only the number of stored limbs does.
    // A checkpoint from another FFT length / digit width (e.g. 2^19 x 16-bit -> 2^18 x 17-bit) is read as long as
    // the value fits: missing limbs are zero, extra stored limbs must be zero. Read into a temporary buffer first,
    // so a failed load leaves 'data' (the start value) untouched.
    const size_t stored = (size_t)header.n_limbs;
    if (stored == 0 || stored > (size_t(1) << 32)) {
        fclose(f); report_bad_checkpoint(filename, "bad header"); return false;
    }
    std::vector<uint64_t> tmp(stored);
    if (fread(tmp.data(), sizeof(uint64_t), stored, f) != stored) {
        fclose(f); report_bad_checkpoint(filename, "file cut short"); return false;
    }
    fclose(f);
    for (size_t i = n_limbs; i < stored; ++i) {
        if (tmp[i] != 0) {
            std::cerr << "[CHECKPOINT] " << filename << ": the stored value does not fit " << n_limbs
                      << " limbs - starting from the beginning\n";
            return false;
        }
    }
    if (stored != n_limbs) {
        std::cout << "[CHECKPOINT] " << filename << ": stored with " << stored << " limbs, now " << n_limbs
                  << " (other FFT length) - converted\n";
    }
    const size_t n_copy = std::min(stored, n_limbs);
    std::memcpy(data, tmp.data(), n_copy * sizeof(uint64_t));
    for (size_t i = n_copy; i < n_limbs; ++i) data[i] = 0;

    out_i = header.current_i;
    return true;
}

uint64_t checkpoint_iteration(const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) return 0;
    CheckpointHeader header;
    const bool ok = (fread(&header, sizeof(CheckpointHeader), 1, f) == 1);
    fclose(f);
    return ok ? header.current_i : 0;
}

double checkpoint_seconds(const std::string& filename) {
    FILE* f = fopen((filename + ".time").c_str(), "r");
    if (!f) return 0.0;
    double s = 0.0;
    const bool ok = (fscanf(f, "%lf", &s) == 1);
    fclose(f);
    return (ok && s > 0.0) ? s : 0.0;
}

void save_checkpoint_seconds(const std::string& filename, double seconds) {
    // as the checkpoint: a temporary file and rename, so a crash never leaves a half-written time
    const std::string name = filename + ".time", tmp = name + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f) return;
    const bool ok = (fprintf(f, "%.3f\n", seconds) > 0);
    const int close_rc = fclose(f);
    if (!ok || close_rc != 0 || std::rename(tmp.c_str(), name.c_str()) != 0) remove(tmp.c_str());
}