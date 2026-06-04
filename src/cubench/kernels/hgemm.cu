/*
 * hgemm.cu — FP16 dense matrix multiplication kernel
 * Uses Tensor Cores via WMMA when available (SM 75+)
 * Fallback: FP16 CUDA cores
 *
 * Optimized for H100 (SM 90) and RTX 5090 (SM 89)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda;
#endif

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#if __CUDA_ARCH__ >= 700
__global__ void hgemm_wmma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) / 4;
    int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / 4;

    if (warp_m * WMMA_M >= M || warp_n * WMMA_N >= N) return;

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warp_m * WMMA_M * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_n * WMMA_N, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_m * WMMA_M * N + warp_n * WMMA_N, c_frag, N,
                            wmma::mem_row_major);
}
#endif

// Generic FP16 kernel (no Tensor Cores)
__global__ void hgemm_naive_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

extern "C" void launch_hgemm(
    const half* A, const half* B, float* C,
    int M, int N, int K,
    bool use_tensor_cores,
    cudaStream_t stream)
{
    if (use_tensor_cores) {
#if __CUDA_ARCH__ >= 700
        dim3 block(32, 8);
        dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
        hgemm_wmma_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
#else
        // Fallback
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        hgemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
#endif
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        hgemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
}
