/*****************************************************************************

CUDA utility funcs

code referenced from : https://github.com/mapillary/inplace_abn

*****************************************************************************/
#pragma once

#include <cuda_runtime_api.h>

// Checks
#ifndef AT_CHECK
    #define AT_CHECK AT_ASSERT
#endif
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
 * General settings
 */
const int WARP_SIZE = 32;
const int MAX_BLOCK_SIZE = 512;

template <typename T>
struct Pair {
  T v1, v2;
  __device__ Pair() {}
  __device__ Pair(T _v1, T _v2) : v1(_v1), v2(_v2) {}
  __device__ Pair(T v) : v1(v), v2(v) {}
  __device__ Pair(int v) : v1(v), v2(v) {}
  __device__ Pair &operator+=(const Pair<T> &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

/*
 * Utility functions
 */
template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask,
                                           int width = warpSize,
                                           unsigned int mask = 0xffffffff) {
#if CUDART_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

__device__ __forceinline__ int getMSB(int val) { return 31 - __clz(val); }

static int getNumThreads(int nElem) {
  int threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template <typename T>
static __device__ __forceinline__ Pair<T> warpSum(Pair<T> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template <typename T, typename Op>
__device__ T reduce(Op op, int plane, int N, int C, int S) {
  T sum = (T)0;
  for (int batch = 0; batch < N; ++batch) {
    for (int x = threadIdx.x; x < S; x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}