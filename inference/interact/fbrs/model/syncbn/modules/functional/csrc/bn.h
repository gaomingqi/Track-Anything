/*****************************************************************************

SyncBN

*****************************************************************************/
#pragma once

#ifdef WITH_CUDA
#include "cuda/ext_lib.h"
#endif

/// SyncBN

std::vector<at::Tensor> syncbn_sum_sqsum(const at::Tensor& x) {
  if (x.is_cuda()) {
#ifdef WITH_CUDA
    return syncbn_sum_sqsum_cuda(x);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU implementation not supported");
  }
}

at::Tensor syncbn_forward(const at::Tensor& x, const at::Tensor& weight,
                          const at::Tensor& bias, const at::Tensor& mean,
                          const at::Tensor& var, bool affine, float eps) {
  if (x.is_cuda()) {
#ifdef WITH_CUDA
    return syncbn_forward_cuda(x, weight, bias, mean, var, affine, eps);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU implementation not supported");
  }
}

std::vector<at::Tensor> syncbn_backward_xhat(const at::Tensor& dz,
                                             const at::Tensor& x,
                                             const at::Tensor& mean,
                                             const at::Tensor& var, float eps) {
  if (dz.is_cuda()) {
#ifdef WITH_CUDA
    return syncbn_backward_xhat_cuda(dz, x, mean, var, eps);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU implementation not supported");
  }
}

std::vector<at::Tensor> syncbn_backward(
    const at::Tensor& dz, const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& bias, const at::Tensor& mean, const at::Tensor& var,
    const at::Tensor& sum_dz, const at::Tensor& sum_dz_xhat, bool affine,
    float eps) {
  if (dz.is_cuda()) {
#ifdef WITH_CUDA
    return syncbn_backward_cuda(dz, x, weight, bias, mean, var, sum_dz,
                                sum_dz_xhat, affine, eps);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CPU implementation not supported");
  }
}
