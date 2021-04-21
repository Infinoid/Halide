#include "fcam/Demosaic.h"

#include "halide_benchmark.h"

#include "camera_pipe.h"

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_malloc_trace.h"
#include "distributed.h"
#include "dumpload.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    if (argc < 8) {
        printf("Usage: ./process raw.png color_temp gamma contrast sharpen timing_iterations output.png\n"
               "e.g. ./process raw.png 3700 2.0 50 1.0 5 output.png\n");
        return 0;
    }

#ifdef HL_MEMINFO
    halide_enable_malloc_trace();
#endif

    distributed_init(&argc, &argv);

    bool runHalide, runC;
    char *benchmark_type = getenv("BENCHMARK_TYPE");
    if (benchmark_type == NULL) {
      runHalide = true;
      runC = true;
    } else if (strncasecmp(benchmark_type,"both",5) == 0) {
      runHalide = true;
      runC = true;
    } else if (strncasecmp(benchmark_type,"halide",7) == 0) {
      runHalide = true;
      runC = false;
    } else if (strncasecmp(benchmark_type,"c",2) == 0) {
      runHalide = false;
      runC = true;
    } else {
      fprintf_rank0(stderr, "Invalide benchmark type of %s. Must be in {halide,c,both}\n", benchmark_type);
      exit(0);
    }

    fprintf_rank0(stderr, "input: %s\n", argv[1]);
    Buffer<uint16_t> input;
    if(strstr(argv[1], ".raw") == argv[1]+strlen(argv[1])-4) {
        input = load_raw_image(argv[1]);
    } else {
        input = load_and_convert_image(argv[1]);
    }
    fprintf_rank0(stderr, "       %d %d\n", input.width(), input.height());

#ifdef HL_MEMINFO
    info(input, "input");
    stats(input, "input");
    // dump(input, "input");
#endif

    // These color matrices are for the sensor in the Nokia N900 and are
    // taken from the FCam source.
    float _matrix_3200[][4] = {{1.6697f, -0.2693f, -0.4004f, -42.4346f},
                               {-0.3576f, 1.0615f, 1.5949f, -37.1158f},
                               {-0.2175f, -1.8751f, 6.9640f, -26.6970f}};

    float _matrix_7000[][4] = {{2.2997f, -0.4478f, 0.1706f, -39.0923f},
                               {-0.3826f, 1.5906f, -0.2080f, -25.4311f},
                               {-0.0888f, -0.7344f, 2.2832f, -20.0826f}};
    Buffer<float> matrix_3200(4, 3), matrix_7000(4, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            matrix_3200(j, i) = _matrix_3200[i][j];
            matrix_7000(j, i) = _matrix_7000[i][j];
        }
    }

    float color_temp = (float)atof(argv[2]);
    float gamma = (float)atof(argv[3]);
    float contrast = (float)atof(argv[4]);
    float sharpen = (float)atof(argv[5]);
    int timing_iterations = atoi(argv[6]);
    int blackLevel = 25;
    int whiteLevel = 1023;

    int full_image_width  = ((input.width () - 32) / 32) * 32;
    int full_image_height = ((input.height() - 24) / 32) * 32;
    int full_image_channels = 3;
    Buffer<uint8_t> output(nullptr, {full_image_width, full_image_height, 3});
    output.set_distributed({full_image_width, full_image_height, 3});
    // Query local buffer size
    camera_pipe(input, matrix_3200, matrix_7000,
                color_temp, gamma, contrast, sharpen, blackLevel, whiteLevel,
                output);
    output.allocate();
    int local_xmin = output.dim(0).min();
    int local_xmax = output.dim(0).min()+output.dim(0).extent();
    int local_ymin = output.dim(1).min();
    int local_ymax = output.dim(1).min()+output.dim(1).extent();
    int local_cmin = output.dim(2).min();
    int local_cmax = output.dim(2).min()+output.dim(2).extent();
    float halide_min = -1, halide_max = -1, cpp_min = -1, cpp_max = -1, mpi_min = -1, mpi_max = -1;
    if(numranks > 1) {
        fprintf_rank0(stderr, "Running in distributed node with %d processes\n", numranks);
        fprintf_rank(stderr, "local output shape is [%d,%d,%d]-[%d,%d,%d]\n",
            local_xmin, local_ymin, local_cmin,
            local_xmax, local_ymax, local_cmax);
    }
    double best;

    if (runHalide) {
      best = benchmark(timing_iterations, 1, [&]() {
          camera_pipe(input, matrix_3200, matrix_7000,
                      color_temp, gamma, contrast, sharpen, blackLevel, whiteLevel,
                      output);
          output.device_sync();
      });
      report_distributed_time("Halide (manual) ISP", best, &halide_min, &halide_max);
    } 

    if (runC) {
          best = benchmark(timing_iterations, 1, [&]() {
          FCam::demosaic(input, output, color_temp, contrast, true, blackLevel, whiteLevel, gamma);
      });
      report_distributed_time("C++ ISP", best, &cpp_min, &cpp_max);
    }

    Buffer<uint8_t> *full_output;
    MPI_Barrier(MPI_COMM_WORLD);
    best = benchmark(1, 1, [&]() {
        full_output = gather_to_rank0(output, full_image_width, full_image_height);
    });
    report_distributed_time("MPI gathering data to node 0", best, &mpi_min, &mpi_max);

#ifdef GENERATE_OUTPUT_FILE
    if(full_output != NULL) {
        // node 0 writes output to file
        fprintf(stderr, "output: %s\n", argv[7]);
        convert_and_save_image(*full_output, argv[7]);
        fprintf(stderr, "        %d %d\n", full_output->width(), full_output->height());
    }
#endif /* GENERATE_OUTPUT_FILE */

    distributed_done();

    fprintf_rank0(stderr, "all timings: halide: %f - %f  c++: %f - %f  mpi: %f - %f\n",
        halide_min, halide_max, cpp_min, cpp_max, mpi_min, mpi_max);
    if(!rank) printf("Success!\n");
    return 0;
}
