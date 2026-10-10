// Copyright (C) Damian Koba.

#include "storage_core.h"
#include <fstream>
#include <mutex>
#include <iostream>

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

