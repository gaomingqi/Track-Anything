/*****************************************************************************

CUDA SyncBN code

code referenced from : https://github.com/mapillary/inplace_abn

*****************************************************************************/
#include <ATen/ATen.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <vector>
#include "cuda/common.h"

// Utilities
void get_dims(at::Tensor x, int64_t &num, int64_t &chn, int64_t &sp) {
  num = x.size(0);
  chn = x.size(1);
  sp = 1;
  for (int64_t i = 2; i < x.ndimension(); ++i) sp *= x.size(i);
}

/// SyncBN

template <typename T>
struct SqSumOp {
  __device__ SqSumOp(const T *t, int c, int s) : tensor(t), chn(c), sp(s) {}
  __device__ __forceinline__ Pair<T> operator()(int batch, int plane, int n) {
    T x = tensor[(batch * chn + plane) * sp + n];
    return Pair<T>(x, x * x);  // x, x^2
  }
  const T *tensor;
  const int chn;
  const int sp;
};

template <typename T>
__global__ void syncbn_sum_sqsum_kernel(const T *x, T *sum, T *sqsum,
                                        int num, int chn, int sp) {
  int plane = blockIdx.x;
  Pair<T> res =
      reduce<Pair<T>, SqSumOp<T>>(SqSumOp<T>(x, chn, sp), plane, num, chn, sp);
  __syncthreads();
  if (threadIdx.x == 0) {
    sum[plane] = res.v1;
    sqsum[plane] = res.v2;
  }
}

std::vector<at::Tensor> syncbn_sum_sqsum_cuda(const at::Tensor &x) {
  CHECK_INPUT(x);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Prepare output tensors
  auto sum = at::empty({chn}, x.options());
  auto sqsum = at::empty({chn}, x.options());

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(
      x.type(), "syncbn_sum_sqsum_cuda", ([&] {
        syncbn_sum_sqsum_kernel<scalar_t><<<blocks, threads>>>(
            x.data<scalar_t>(), sum.data<scalar_t>(),
            sqsum.data<scalar_t>(), num, chn, sp);
    }));
  return {sum, sqsum};
}

template <typename T>
__global__ void syncbn_forward_kernel(T *z, const T *x, const T *weight,
                                      const T *bias, const T *mean,
                                      const T *var, bool affine, float eps,
                                      int num, int chn, int sp) {
  int plane = blockIdx.x;
  T _mean = mean[plane];
  T _var = var[plane];
  T _weight = affine ? weight[plane] : T(1);
  T _bias = affine ? bias[plane] : T(0);
  float _invstd = T(0);
  if (_var || eps) {
    _invstd = rsqrt(_var + eps);
  }
  for (int batch = 0; batch < num; ++batch) {
    for (int n = threadIdx.x; n < sp; n += blockDim.x) {
      T _x = x[(batch * chn + plane) * sp + n];
      T _xhat = (_x - _mean) * _invstd;
      T _z = _xhat * _weight + _bias;
      z[(batch * chn + plane) * sp + n] = _z;
    }
  }
}

at::Tensor syncbn_forward_cuda(const at::Tensor &x, const at::Tensor &weight,
                               const at::Tensor &bias, const at::Tensor &mean,
                               const at::Tensor &var, bool affine, float eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  CHECK_INPUT(mean);
  CHECK_INPUT(var);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  auto z = at::zeros_like(x);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(
      x.type(), "syncbn_forward_cuda", ([&] {
        syncbn_forward_kernel<scalar_t><<<blocks, threads>>>(
            z.data<scalar_t>(), x.data<scalar_t>(),
            weight.data<scalar_t>(), bias.data<scalar_t>(),
            mean.data<scalar_t>(), var.data<scalar_t>(),
            affine, eps, num, chn, sp);
      }));
  return z;
}

template <typename T>
struct XHatOp {
  __device__ XHatOp(T _weight, T _bias, const T *_dz, const T *_x, int c, int s)
      : weight(_weight), bias(_bias), x(_x), dz(_dz), chn(c), sp(s) {}
  __device__ __forceinline__ Pair<T> operator()(int batch, int plane, int n) {
    // xhat = (x - bias) * weight
    T _xhat = (x[(batch * chn + plane) * sp + n] - bias) * weight;
    // dxhat * x_hat
    T _dz = dz[(batch * chn + plane) * sp + n];
    return Pair<T>(_dz, _dz * _xhat);
  }
  const T weight;
  const T bias;
  const T *dz;
  const T *x;
  const int chn;
  const int sp;
};

template <typename T>
__global__ void syncbn_backward_xhat_kernel(const T *dz, const T *x,
                                            const T *mean, const T *var,
                                            T *sum_dz, T *sum_dz_xhat,
                                            float eps, int num, int chn,
                                            int sp) {
  int plane = blockIdx.x;
  T _mean = mean[plane];
  T _var = var[plane];
  T _invstd = T(0);
  if (_var || eps) {
    _invstd = rsqrt(_var + eps);
  }
  Pair<T> res = reduce<Pair<T>, XHatOp<T>>(
      XHatOp<T>(_invstd, _mean, dz, x, chn, sp), plane, num, chn, sp);
  __syncthreads();
  if (threadIdx.x == 0) {
    // \sum(\frac{dJ}{dy_i})
    sum_dz[plane] = res.v1;
    // \sum(\frac{dJ}{dy_i}*\hat{x_i})
    sum_dz_xhat[plane] = res.v2;
  }
}

std::vector<at::Tensor> syncbn_backward_xhat_cuda(const at::Tensor &dz,
                                                  const at::Tensor &x,
                                                  const at::Tensor &mean,
                                                  const at::Tensor &var,
                                                  float eps) {
  CHECK_INPUT(dz);
  CHECK_INPUT(x);
  CHECK_INPUT(mean);
  CHECK_INPUT(var);
  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);
  // Prepare output tensors
  auto sum_dz = at::empty({chn}, x.options());
  auto sum_dz_xhat = at::empty({chn}, x.options());
  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(
      x.type(), "syncbn_backward_xhat_cuda", ([&] {
        syncbn_backward_xhat_kernel<scalar_t><<<blocks, threads>>>(
            dz.data<scalar_t>(), x.data<scalar_t>(), mean.data<scalar_t>(),
            var.data<scalar_t>(), sum_dz.data<scalar_t>(),
            sum_dz_xhat.data<scalar_t>(), eps, num, chn, sp);
      }));
  return {sum_dz, sum_dz_xhat};
}

template <typename T>
__global__ void syncbn_backward_kernel(const T *dz, const T *x, const T *weight,
                                       const T *bias, const T *mean,
                                       const T *var, const T *sum_dz,
                                       const T *sum_dz_xhat, T *dx, T *dweight,
                                       T *dbias, bool affine, float eps,
                                       int num, int chn, int sp) {
  int plane = blockIdx.x;
  T _mean = mean[plane];
  T _var = var[plane];
  T _weight = affine ? weight[plane] : T(1);
  T _sum_dz = sum_dz[plane];
  T _sum_dz_xhat = sum_dz_xhat[plane];
  T _invstd = T(0);
  if (_var || eps) {
    _invstd = rsqrt(_var + eps);
  }
  /*
    \frac{dJ}{dx_i} = \frac{1}{N\sqrt{(\sigma^2+\epsilon)}} (
      N\frac{dJ}{d\hat{x_i}} -
      \sum_{j=1}^{N}(\frac{dJ}{d\hat{x_j}}) -
      \hat{x_i}\sum_{j=1}^{N}(\frac{dJ}{d\hat{x_j}}\hat{x_j})
    )
    Note : N is omitted here since it will be accumulated and
    _sum_dz and _sum_dz_xhat expected to be already normalized
    before the call.
  */
  if (dx) {
    T _mul = _weight * _invstd;
    for (int batch = 0; batch < num; ++batch) {
      for (int n = threadIdx.x; n < sp; n += blockDim.x) {
        T _dz = dz[(batch * chn + plane) * sp + n];
        T _xhat = (x[(batch * chn + plane) * sp + n] - _mean) * _invstd;
        T _dx = (_dz - _sum_dz - _xhat * _sum_dz_xhat) * _mul;
        dx[(batch * chn + plane) * sp + n] = _dx;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (affine) {
      T _norm = num * sp;
      dweight[plane] += _sum_dz_xhat * _norm;
      dbias[plane] += _sum_dz * _norm;
    }
  }
}

std::vector<at::Tensor> syncbn_backward_cuda(
    const at::Tensor &dz, const at::Tensor &x, const at::Tensor &weight,
    const at::Tensor &bias, const at::Tensor &mean, const at::Tensor &var,
    const at::Tensor &sum_dz, const at::Tensor &sum_dz_xhat, bool affine,
    float eps) {
  CHECK_INPUT(dz);
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  CHECK_INPUT(mean);
  CHECK_INPUT(var);
  CHECK_INPUT(sum_dz);
  CHECK_INPUT(sum_dz_xhat);

  // Extract dimensions
  int64_t num, chn, sp;
  get_dims(x, num, chn, sp);

  // Prepare output tensors
  auto dx = at::zeros_like(dz);
  auto dweight = at::zeros_like(weight);
  auto dbias = at::zeros_like(bias);

  // Run kernel
  dim3 blocks(chn);
  dim3 threads(getNumThreads(sp));
  AT_DISPATCH_FLOATING_TYPES(
      x.type(), "syncbn_backward_cuda", ([&] {
        syncbn_backward_kernel<scalar_t><<<blocks, threads>>>(
            dz.data<scalar_t>(), x.data<scalar_t>(), weight.data<scalar_t>(),
            bias.data<scalar_t>(), mean.data<scalar_t>(), var.data<scalar_t>(),
            sum_dz.data<scalar_t>(), sum_dz_xhat.data<scalar_t>(),
            dx.data<scalar_t>(), dweight.data<scalar_t>(),
            dbias.data<scalar_t>(), affine, eps, num, chn, sp);
      }));
  return {dx, dweight, dbias};
}