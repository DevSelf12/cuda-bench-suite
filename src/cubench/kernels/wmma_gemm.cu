/*
 * wmma_gemm.cu — Tensor Core GEMM using WMMA API
 * Supports bf16, fp16, tf32 on SM 80+
 *
 * H100 (SM 90): Gen 4 Tensor Cores, up to 989 TFLOPS (bf16)
 * H200 (SM 90a): Same compute, 141 GB HBM3e bandwidth
 * RTX 5090 (SM 89): Gen 5 Tensor Cores
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if __CUDA_ARCH__ >= 800
#include <mma.h>
using namespace nvcuda;
#endif

// BF16 GEMM via WMMA
#if __CUDA_ARCH__ >= 800
__global__ void wmma_bf16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int warp_row = (blockIdx.y * 4 + threadIdx.y / 8) * 16;
    int warp_col = (blockIdx.x * 4 + threadIdx.x / 8) * 16;

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}
#endif

// TF32 GEMM via WMMA (SM 80+)
#if __CUDA_ARCH__ >= 800
__global__ void wmma_tf32_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 8, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, float, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int warp_row = (blockIdx.y * 4 + threadIdx.y / 8) * 16;
    int warp_col = (blockIdx.x * 4 + threadIdx.x / 8) * 16;

    for (int k = 0; k < K; k += 8) {
        wmma::load_matrix_sync(a_frag, A + warp_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
}
#endif
