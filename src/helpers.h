#ifndef GPGPUTASKS_HELPERS_H
#define GPGPUTASKS_HELPERS_H

#include <iostream>
#include <string>


template <typename T>
void print_vector(std::string name, int sz, T* data) {
#ifdef DEBUG_PRINT
    std::cout << name << ": " << '\n';
    for (int i = 0; i < sz; i++) {
        std::cout << data[i] << '\t';
    }
    std::cout << '\n';
#endif
}

template <typename T>
void print_matrix(std::string name, int h, int w, int x0, int x1, int y0, int y1, T* data) {
#ifdef DEBUG_PRINT
    std::cout << name << ": " << '\n';
    for (size_t y = y0; y < y1; y++) {
        for (size_t x = x0; x < x1; ++x) {
            std::cout << data[y * w + x] << '\t';
        }
        std::cout << '\n';
    }
#endif
}

template <typename T>
void print_matrix(std::string name, int h, int w, int x1, int y1, T* data) {
    print_matrix(name, h, w, 0, x1, 0, y1, data);
}

template <typename T>
void print_matrix(std::string name, int h, int w, T* data) {
    print_matrix(name, h, w, 0, w, 0, h, data);
}
#define GPGPUTASKS_HELPERS_H

#endif // GPGPUTASKS_HELPERS_H
