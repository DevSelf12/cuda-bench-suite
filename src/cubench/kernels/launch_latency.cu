/*
 * launch_latency.cu — Minimal kernel for measuring launch overhead
 * Empty kernel to isolate driver latency from compute time
 */

#include <cuda_runtime.h>

__global__ void empty_kernel() {
    // Intentionally empty — measures pure launch overhead
}

extern "C" void launch_empty(cudaStream_t stream) {
    empty_kernel<<<1, 1, 0, stream>>>();
}
