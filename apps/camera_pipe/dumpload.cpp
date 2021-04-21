#include "dumpload.h"
#include <cassert>
#include <cstdio>
#include <string>

using std::string;

struct bufsize {
    int x, y;
};

void dump_raw_image(const string filename, const Halide::Buffer<uint16_t> &buf) {
    struct bufsize size = {.x=buf.width(), .y=buf.height()};
    FILE *f = fopen(filename.c_str(), "w");
    assert(f);
    fwrite(&size, sizeof(size), 1, f);
    fwrite(buf.begin(), buf.static_halide_type().bytes(), buf.number_of_elements(), f);
    fclose(f);
}

Halide::Runtime::Buffer<uint16_t> load_raw_image(const string filename) {
    struct bufsize size;
    FILE *f = fopen(filename.c_str(), "r");
    assert(f);
    fread(&size, sizeof(size), 1, f);
    Halide::Runtime::Buffer<uint16_t> rv(size.x, size.y);
    fread(rv.begin(), rv.static_halide_type().bytes(), rv.number_of_elements(), f);
    fclose(f);
    return rv;
}
