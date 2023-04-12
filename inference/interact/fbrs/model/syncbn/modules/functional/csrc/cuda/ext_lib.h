/*****************************************************************************

CUDA SyncBN code

*****************************************************************************/
#pragma once
#include <torch/extension.h>
#include <vector>

/// Sync-BN
std::vector<at::Tensor> syncbn_sum_sqsum_cuda(const at::Tensor& x);
at::Tensor syncbn_forward_cuda(const at::Tensor& x, const at::Tensor& weight,
                               const at::Tensor& bias, const at::Tensor& mean,
                               const at::Tensor& var, bool affine, float eps);
std::vector<at::Tensor> syncbn_backward_xhat_cuda(const at::Tensor& dz,
                                                  const at::Tensor& x,
                                                  const at::Tensor& mean,
                                                  const at::Tensor& var,
                                                  float eps);
std::vector<at::Tensor> syncbn_backward_cuda(
    const at::Tensor& dz, const at::Tensor& x, const at::Tensor& weight,
    const at::Tensor& bias, const at::Tensor& mean, const at::Tensor& var,
    const at::Tensor& sum_dz, const at::Tensor& sum_dz_xhat, bool affine,
    float eps);
