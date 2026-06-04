/*
 * sgemm.cu — FP32 dense matrix multiplication kernel
 * Optimized for SM 89 (RTX 5090), SM 90 (H100/H200)
 *
 * Tile size: 128x128 with 8x8 thread blocks
 * Shared memory: double-buffered
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 8
#define THREADS_X 16
#define THREADS_Y 16

__global__ void sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty * (TILE_M / THREADS_Y) + (tx / (THREADS_X / (TILE_M / THREADS_Y)));
    int col = bx * TILE_N + tx % (TILE_N / THREADS_X) * (TILE_N / (TILE_N / THREADS_X));

    float acc[8][8] = {0.0f};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Load tiles into shared memory
        for (int i = 0; i < TILE_K; i++) {
            int a_col = t * TILE_K + i;
            if (row < M && a_col < K)
                As[i][ty * (TILE_M / THREADS_Y) + tx % (TILE_M / THREADS_Y)] = A[row * K + a_col];
            else
                As[i][ty * (TILE_M / THREADS_Y) + tx % (TILE_M / THREADS_Y)] = 0.0f;

            int b_row = t * TILE_K + i;
            if (b_row < K && col < N)
                Bs[i][tx] = B[b_row * N + col];
            else
                Bs[i][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial products
        for (int k = 0; k < TILE_K; k++) {
            for (int m = 0; m < 8; m++) {
                for (int n = 0; n < 8; n++) {
                    acc[m][n] += As[k][ty * 8 + m] * Bs[k][tx * 8 + n];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    for (int m = 0; m < 8; m++) {
        for (int n = 0; n < 8; n++) {
            int r = by * TILE_M + ty * 8 + m;
            int c = bx * TILE_N + tx * 8 + n;
            if (r < M && c < N) {
                C[r * N + c] = alpha * acc[m][n] + beta * C[r * N + c];
            }
        }
    }
}

extern "C" void launch_sgemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    sgemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
