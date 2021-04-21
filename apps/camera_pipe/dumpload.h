#ifndef _DUMPLOAD_H
#define _DUMPLOAD_H

#include <string>
#include <Halide.h>

void dump_raw_image(const std::string filename, const Halide::Buffer<uint16_t> &buf);
Halide::Runtime::Buffer<uint16_t> load_raw_image(const std::string filename);

#endif /* _DUMPLOAD_H */
