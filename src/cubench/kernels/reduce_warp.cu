/*
 * reduce_warp.cu — Warp-level parallel reduction
 * Optimized for SM 89/90 using shuffle intrinsics
 *
 * Supports: sum, max, min, argmax, argmin
 */

#include <cuda_runtime.h>

#define WARP_SIZE 32

// Warp-level reduction using __shfl_down_sync
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction (uses shared memory)
__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N)
{
    __shared__ float shared[WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < N) ? input[gid] : 0.0f;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp result to shared memory
    if (tid % WARP_SIZE == 0) {
        shared[tid / WARP_SIZE] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    if (tid < WARP_SIZE) {
        val = (tid < blockDim.x / WARP_SIZE) ? shared[tid] : 0.0f;
        val = warp_reduce_sum(val);
    }

    if (tid == 0) {
        atomicAdd(output, val);
    }
}

// Max reduction with index
__global__ void reduce_argmax_kernel(const float* __restrict__ input,
                                      float* __restrict__ max_val,
                                      int* __restrict__ max_idx,
                                      int N)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < N) ? input[gid] : -INFINITY;
    int idx = gid;

    // Warp-level argmax
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }

    if (tid % WARP_SIZE == 0) {
        atomicMax((int*)max_val, __float_as_int(val));
        if (max_idx) max_idx[0] = idx;
    }
}

extern "C" void launch_reduce_sum(
    const float* input, float* output, int N,
    int block_size, cudaStream_t stream)
{
    int grid_size = (N + block_size - 1) / block_size;
    reduce_sum_kernel<<<grid_size, block_size, 0, stream>>>(input, output, N);
}
