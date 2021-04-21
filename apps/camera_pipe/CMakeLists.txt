cmake_minimum_required(VERSION 3.16)
project(camera_pipe LANGUAGES CXX CUDA)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Libraries
find_package(Halide REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)


# Generator
add_executable(camera_pipe.generator camera_pipe_generator.cpp)
target_link_libraries(camera_pipe.generator
                      PRIVATE
                      Halide::Generator
                      Halide::Tools)

# Filters
add_halide_library(camera_pipe FROM camera_pipe.generator FEATURES large_buffers)

#FCAM
add_library(fcam fcam/Demosaic.cpp fcam/Demosaic.cu)
set_target_properties(fcam PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                      CUDA_ARCHITECTURES "75")
target_link_libraries(fcam PRIVATE Halide::Halide OpenMP::OpenMP_CXX)

#fast image dump/load library
add_library(dumpload dumpload.cpp)
target_link_libraries(dumpload PRIVATE Halide::Halide)

add_executable(dump dump.cpp)
target_link_libraries(dump PRIVATE dumpload Halide::ImageIO Halide::Halide)

# Main executable
add_executable(camera_pipe_process process.cpp distributed.cpp distributed.h)
set_target_properties(camera_pipe_process PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(camera_pipe_process
                      PRIVATE
                      fcam
                      Halide::ImageIO
                      dumpload
                      camera_pipe)

# Test that the app actually works!
set(IMAGE ${CMAKE_CURRENT_LIST_DIR}/../images/bayer_raw.png)
if (EXISTS ${IMAGE})
    configure_file(${IMAGE} bayer_raw.png COPYONLY)
    add_test(NAME camera_pipe_process
             COMMAND camera_pipe_process bayer_raw.png 3700 2.0 50 1.0 5 out.png)
    set_tests_properties(camera_pipe_process PROPERTIES
                         LABELS camera_pipe
                         PASS_REGULAR_EXPRESSION "Success!"
                         SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
endif ()
