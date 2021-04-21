#include <cassert>
#include <cstdio>

#include "halide_image_io.h"
#include "dumpload.h"

using Halide::Buffer;
using Halide::Tools::load_and_convert_image;

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: ./dump raw.png out.raw\n");
        return 0;
    }

    printf("input: %s\n", argv[1]);
    Buffer<uint16_t> input = load_and_convert_image(argv[1]);
    printf("image has size %dx%d\n", input.width(), input.height());
    printf("output: %s\n", argv[2]);
    dump_raw_image(argv[2], input);
}
