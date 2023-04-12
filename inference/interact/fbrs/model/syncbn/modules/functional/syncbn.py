"""
/*****************************************************************************/

BatchNorm2dSync with multi-gpu

code referenced from : https://github.com/mapillary/inplace_abn

/*****************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.cuda.comm as comm
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ._csrc import _backend


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class BatchNorm2dSyncFunc(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                extra, compute_stats=True, momentum=0.1, eps=1e-05):
        def _parse_extra(ctx, extra):
            ctx.is_master = extra["is_master"]
            if ctx.is_master:
                ctx.master_queue = extra["master_queue"]
                ctx.worker_queues = extra["worker_queues"]
                ctx.worker_ids = extra["worker_ids"]
            else:
                ctx.master_queue = extra["master_queue"]
                ctx.worker_queue = extra["worker_queue"]
        # Save context
        if extra is not None:
            _parse_extra(ctx, extra)
        ctx.compute_stats = compute_stats
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.affine = weight is not None and bias is not None
        if ctx.compute_stats:
            N = _count_samples(x) * (ctx.master_queue.maxsize + 1)
            assert N > 1
            # 1. compute sum(x) and sum(x^2)
            xsum, xsqsum = _backend.syncbn_sum_sqsum(x.detach())
            if ctx.is_master:
                xsums, xsqsums = [xsum], [xsqsum]
                # master : gatther all sum(x) and sum(x^2) from slaves
                for _ in range(ctx.master_queue.maxsize):
                    xsum_w, xsqsum_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    xsums.append(xsum_w)
                    xsqsums.append(xsqsum_w)
                xsum = comm.reduce_add(xsums)
                xsqsum = comm.reduce_add(xsqsums)
                mean = xsum / N
                sumvar = xsqsum - xsum * mean
                var = sumvar / N
                uvar = sumvar / (N - 1)
                # master : broadcast global mean, variance to all slaves
                tensors = comm.broadcast_coalesced(
                    (mean, uvar, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                # slave : send sum(x) and sum(x^2) to master
                ctx.master_queue.put((xsum, xsqsum))
                # slave : get global mean and variance
                mean, uvar, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * uvar)
            ctx.N = N
            ctx.save_for_backward(x, weight, bias, mean, var)
        else:
            mean, var = running_mean, running_var

        # do batch norm forward
        z = _backend.syncbn_forward(x, weight, bias, mean, var,
                                    ctx.affine, ctx.eps)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, weight, bias, mean, var = ctx.saved_tensors
        dz = dz.contiguous()

        # 1. compute \sum(\frac{dJ}{dy_i}) and \sum(\frac{dJ}{dy_i}*\hat{x_i})
        sum_dz, sum_dz_xhat = _backend.syncbn_backward_xhat(
            dz, x, mean, var, ctx.eps)
        if ctx.is_master:
            sum_dzs, sum_dz_xhats = [sum_dz], [sum_dz_xhat]
            # master : gatther from slaves
            for _ in range(ctx.master_queue.maxsize):
                sum_dz_w, sum_dz_xhat_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                sum_dzs.append(sum_dz_w)
                sum_dz_xhats.append(sum_dz_xhat_w)
            # master : compute global stats
            sum_dz = comm.reduce_add(sum_dzs)
            sum_dz_xhat = comm.reduce_add(sum_dz_xhats)
            sum_dz /= ctx.N
            sum_dz_xhat /= ctx.N
            # master : broadcast global stats
            tensors = comm.broadcast_coalesced(
                (sum_dz, sum_dz_xhat), [mean.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)
        else:
            # slave : send to master
            ctx.master_queue.put((sum_dz, sum_dz_xhat))
            # slave : get global stats
            sum_dz, sum_dz_xhat = ctx.worker_queue.get()
            ctx.worker_queue.task_done()

        # do batch norm backward
        dx, dweight, dbias = _backend.syncbn_backward(
            dz, x, weight, bias, mean, var, sum_dz, sum_dz_xhat,
            ctx.affine, ctx.eps)

        return dx, dweight, dbias, \
            None, None, None, None, None, None

batchnorm2d_sync = BatchNorm2dSyncFunc.apply

__all__ = ["batchnorm2d_sync"]
