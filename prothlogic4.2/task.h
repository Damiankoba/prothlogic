// Copyright (C) Damian Koba.

#pragma once
#include <string>
#include <map>
#include <cstdint> 

enum ProveMode {
    PROVE_PROTH_FF = 1,
};

struct Task {
    unsigned int n;
    uint64_t k;
    std::string label;
    std::map<std::string, std::string> payload;   
};