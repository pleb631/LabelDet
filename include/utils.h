#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <filesystem>
#include <iostream>
#include <string>

typedef double FP64;
namespace fs = std::filesystem;

struct point {
    int x = 0;
    int y = 0;
    point() = default;
    point(int _x, int _y) : x(_x), y(_y) {}
};

struct Span {
    point p1;
    point p2;
    double scale = 1.0;

    Span() = default;
    Span(int w, int h) {
        reset(w, h);
    }

    void reset(int w, int h) {
        scale = 1.0;
        p1 = point(0, 0);
        p2 = point(w, h);
    }
};


fs::path imgpath2txtpath(fs::path& img_path);
void getFilesInDirectory(const fs::path& directory, std::vector<fs::path>& files, const std::vector<std::string>& extensions);

#endif // UTILS_H
