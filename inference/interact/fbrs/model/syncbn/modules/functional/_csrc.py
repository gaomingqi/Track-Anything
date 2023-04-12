"""
/*****************************************************************************/

Extension module loader

code referenced from : https://github.com/facebookresearch/maskrcnn-benchmark

/*****************************************************************************/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path

import torch

try:
    from torch.utils.cpp_extension import load
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    raise ImportError(
        "The cpp layer extensions requires PyTorch 0.4 or higher")


def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.join(this_dir, "csrc")

    main_file = glob.glob(os.path.join(this_dir, "*.cpp"))
    sources_cpu = glob.glob(os.path.join(this_dir, "cpu", "*.cpp"))
    sources_cuda = glob.glob(os.path.join(this_dir, "cuda", "*.cu"))

    sources = main_file + sources_cpu

    extra_cflags = []
    extra_cuda_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        sources.extend(sources_cuda)
        extra_cflags = ["-O3", "-DWITH_CUDA"]
        extra_cuda_cflags = ["--expt-extended-lambda"]
    sources = [os.path.join(this_dir, s) for s in sources]
    extra_include_paths = [this_dir]
    return load(
        name="ext_lib",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        extra_cuda_cflags=extra_cuda_cflags,
    )


_backend = _load_C_extensions()
