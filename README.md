## Overview

Takes two DNA sequences (composed of bases A, C, G, T) and finds all common subsequences that meet a minimum length threshold. For every pair of starting positions across both sequences, it extends forward while consecutive bases match and records hits as (position1, position2, length) triplets. The CPU implementation uses nested loops; the GPU version assigns one position pair per CUDA thread with per-block shared-memory reduction to collect results. Both run the same algorithm so their outputs and timings can be compared directly.

## Prerequisites

- NVIDIA GPU (Compute Capability 5.0+)
- CUDA Toolkit (11.x+)
- CMake 3.25 or later
- C++14-compatible compiler (GCC 5+, Clang 3.4+, or MSVC with CUDA support)

## Building

```bash
mkdir build && cd build
cmake ../src
make
```

This produces the `gens` executable in the build directory. Running it generates two random DNA sequences of 10,000 bases each, finds all matching subsequences of length >= 9, and prints:

- Last 3 matches from each implementation (CPU and GPU)
- CPU execution time and total matches
- GPU execution time and total matches
- GPU speedup factor

