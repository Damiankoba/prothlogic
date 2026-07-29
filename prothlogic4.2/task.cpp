// Copyright (C) Damian Koba.

#include <string>
#include <map>
#include <variant>


struct Task {
    std::string label;
    std::map<std::string, std::string> payload;
};
