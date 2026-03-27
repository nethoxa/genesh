#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gen_compare_gpu.h"

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                __FILE__, __LINE__,                         \
                cudaGetErrorString(err));                   \
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while (0)

#define MAX_MATCHES 1000000

__global__ void findMatchesKernel(int* genes1, int* genes2, int* result,
                                   int min, int maxcg1, int maxcg2,
                                   int seqLen1, int seqLen2, int maxResults) {
    // 2D grid: threads in the same warp share cg1, differ in cg2,
    // improving memory coalescing on genes2 reads
    int cg2 = blockDim.x * blockIdx.x + threadIdx.x;
    int cg1 = blockDim.y * blockIdx.y + threadIdx.y;

    // Per-block shared counters to reduce global atomic contention
    __shared__ int blockCount;
    __shared__ int blockOffset;

    if (threadIdx.x == 0 && threadIdx.y == 0) blockCount = 0;
    __syncthreads();

    int gl = 0;
    int localIdx = -1;

    if (cg1 < maxcg1 && cg2 < maxcg2) {
        int icg1 = cg1;
        int icg2 = cg2;
        // Use full sequence lengths for extension (not maxcgX which is just the start bound)
        while (icg1 < seqLen1 && icg2 < seqLen2 && genes1[icg1] == genes2[icg2]) {
            ++icg1;
            ++icg2;
        }

        gl = icg1 - cg1;
        if (gl >= min) {
            localIdx = atomicAdd(&blockCount, 1);
        }
    }
    __syncthreads();

    // One thread per block reserves space in the global output array
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockCount > 0) {
        blockOffset = atomicAdd(result, blockCount * 3);
    }
    __syncthreads();

    // Write results at pre-reserved positions (no global atomic contention)
    if (localIdx >= 0) {
        int base = blockOffset + localIdx * 3;
        if (base + 3 <= maxResults * 3) {
            result[base + 1] = cg1;
            result[base + 2] = cg2;
            result[base + 3] = gl;
        }
    }
}

std::vector<int> findMatchesGPU(GeneSequence &genes1, GeneSequence &genes2, int min) {
    int seqLen1 = genes1.size();
    int seqLen2 = genes2.size();
    int maxcg1 = seqLen1 - min;
    int maxcg2 = seqLen2 - min;

    int* d_genes1;
    int* d_genes2;
    CUDA_CHECK(cudaMalloc(&d_genes1, seqLen1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_genes2, seqLen2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_genes1, genes1.data(), seqLen1 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_genes2, genes2.data(), seqLen2 * sizeof(int), cudaMemcpyHostToDevice));

    // Capped result buffer: 1 counter + 3 ints per match (12 MB vs ~1.1 GB before)
    const int maxMatches = MAX_MATCHES;
    int resultBufferSize = (1 + maxMatches * 3) * sizeof(int);
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, resultBufferSize));
    CUDA_CHECK(cudaMemset(d_result, 0, resultBufferSize));

    // 2D grid matching the 2D problem structure (RTX 3070 Ti, SM 86)
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (maxcg2 + blockSize.x - 1) / blockSize.x,
        (maxcg1 + blockSize.y - 1) / blockSize.y
    );
    findMatchesKernel<<<gridSize, blockSize>>>(d_genes1, d_genes2, d_result,
                                                min, maxcg1, maxcg2,
                                                seqLen1, seqLen2, maxMatches);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int resultSize;
    CUDA_CHECK(cudaMemcpy(&resultSize, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    if (resultSize > maxMatches * 3) {
        fprintf(stderr, "Warning: result buffer overflow, results truncated to %d matches\n", maxMatches);
        resultSize = maxMatches * 3;
    }

    std::vector<int> result(resultSize);
    if (resultSize > 0) {
        CUDA_CHECK(cudaMemcpy(result.data(), d_result + 1, resultSize * sizeof(int), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_genes1));
    CUDA_CHECK(cudaFree(d_genes2));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}
