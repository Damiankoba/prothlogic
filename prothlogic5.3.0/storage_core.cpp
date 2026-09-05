// Copyright (C) Damian Koba.

#include "storage_core.h"
#include <fstream>
#include <mutex>
#include <iostream>

static std::mutex g_csv_mutex;
static std::mutex g_txt_mutex;

void append_line_txt(const std::string& filename,
                     const std::string& line)
{
    std::lock_guard<std::mutex> lock(g_txt_mutex);
    std::ofstream f(filename, std::ios::app);
    if (!f) {
        std::cerr << "[ERROR] I can't open it: " << filename << "\n";
        return;
    }
    f << line << "\n";
}

void save_proth_result_csv(const std::string& filename,
    unsigned n_pow2,
    uint64_t k,
    const mpz_t N,
    uint64_t witness,
    int tried)
{
    std::lock_guard<std::mutex> lock(g_csv_mutex);
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        std::cerr << "[ERROR] I can't open it: " << filename << "\n";
        return;
    }

    char* s = mpz_get_str(nullptr, 10, N);
    file << n_pow2 << "," << k << "," << s << "," << witness << "," << tried << "\n";
    free(s);
}