# 项目环境搭建与 mmcv 源码编译（CUDA 12.8）

> 也许不是必须的，但是我原本使用torch 2.9.1+cu128而且不想再费事装旧版（否则可以尝试torch 2.2.x+cuda12.1）。因此尝试直接build mmcv==2.2.0

## 1. 选择并固定 Python 版本
使用 Python 3.11（项目 `pyproject.toml` 要求）。

```bash
cd /home/zoyo/Desktop/Track-Anything
uv venv -p 3.11 .venv
source .venv/bin/activate
python -V
```

## 2. 升级基础构建工具
```bash
uv pip install -U pip setuptools wheel ninja cmake
```

## 3. 安装 PyTorch CUDA 12.8 版本
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.9.1+cu128" "torchvision==0.24.1+cu128"
```

## 4. 安装项目依赖
```bash
uv pip install -r requirements.txt
```

## 5. 使用 CUDA 12.8 编译安装 mmcv
```bash
CUDA_HOME=/usr/local/cuda-12.8 \
CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
FORCE_CUDA=1 MMCV_WITH_OPS=1 \
python -m pip install -v --no-cache-dir --no-build-isolation --no-binary :all: mmcv==2.2.0
```

## 6. 验证 mmcv.ops 是否可用
```bash
python - <<'PY'
import mmcv
from mmcv.ops import ModulatedDeformConv2d
print("mmcv", mmcv.__version__)
print("mmcv.ops ok", ModulatedDeformConv2d)
PY
```

## 7. 如需改用 CUDA 13.0
把上面的 `CUDA_HOME` 和 `CUDACXX` 改成 `/usr/local/cuda-13.0` 对应路径即可。
