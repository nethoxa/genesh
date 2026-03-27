cmake_minimum_required(VERSION 3.25)
project(gens)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3070 Ti = SM 86

include_directories(.)

enable_language(CUDA)

add_executable(gens
        defs.h
        gen_compare.cpp
        gen_compare_gpu.cu
        gen_compare_gpu.h)
