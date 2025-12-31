# 完全 C++ 化的 MCTS 搜索

## 概述

这个实现将整个 MCTS 搜索循环移到 C++ 中，包括模型推理，消除 Python GIL 开销。

**预期性能提升**: 50-70%（消除 GIL 91% 的影响）

## 架构对比

### 原版（Python 循环）
```
Python 主循环 (GIL 锁定)
├── for simulation in range(50):
│   ├── C++ batch_traverse()      ← 快
│   ├── Python tensor 创建        ← 慢，GIL
│   ├── Python model.forward()    ← GIL
│   ├── Python to_detach_cpu()    ← 慢，GIL
│   └── C++ batch_backpropagate() ← 快
```

### 新版（完全 C++）
```
C++ 主循环 (无 GIL)
├── for simulation in range(50):
│   ├── C++ batch_traverse()
│   ├── C++ libtorch tensor 创建   ← 快，无 GIL
│   ├── C++ model.forward()        ← 快，无 GIL
│   └── C++ batch_backpropagate()
```

## 编译步骤

### 1. 安装依赖

```bash
# 下载 libtorch (选择对应 CUDA 版本)
# CPU 版本:
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip

# CUDA 11.8 版本:
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip

unzip libtorch-*.zip

# 安装 pybind11
pip install pybind11
```

### 2. 编译

```bash
cd lzero/mcts/ctree/ctree_gumbel_muzero

mkdir build && cd build

# 设置 libtorch 路径
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
      -f ../CMakeLists_with_model.txt \
      ..

make -j8
```

### 3. 导出模型为 TorchScript

```python
import torch
from lzero.model import MuZeroModel

# 加载训练好的模型
model = MuZeroModel(...)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 导出 recurrent_inference
class RecurrentInferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, latent_state, action):
        output = self.model.recurrent_inference(latent_state, action)
        return (
            output.latent_state,
            output.reward,
            output.value,
            output.policy_logits
        )

wrapper = RecurrentInferenceWrapper(model)
scripted = torch.jit.script(wrapper)
scripted.save('model_recurrent.pt')
```

## 使用方法

### 方法 1: 直接替换 search 方法

修改 `lzero/mcts/tree_search/mcts_ctree.py`:

```python
# 在 GumbelMuZeroMCTSCtree 类中添加

def search_cpp(self, roots, model_path, latent_state_roots, to_play_batch):
    """使用完全 C++ 的搜索"""
    from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree_cpp
    
    # 将 numpy 数组传给 C++
    latent_states_np = np.asarray(latent_state_roots, dtype=np.float32)
    
    gmz_tree_cpp.search_with_model(
        roots,
        model_path,
        latent_states_np,
        list(to_play_batch),
        self._cfg.num_simulations,
        self._cfg.max_num_considered_actions,
        self._cfg.discount_factor,
        self._cfg.value_delta_max,
        True  # use_cuda
    )
```

### 方法 2: 在 Policy 中使用

修改 `lzero/policy/gumbel_muzero.py`:

```python
def _forward_collect(self, ...):
    # 检查是否有导出的 TorchScript 模型
    if hasattr(self, '_torchscript_model_path') and self._torchscript_model_path:
        # 使用 C++ 搜索
        self._mcts_collect.search_cpp(
            roots,
            self._torchscript_model_path,
            latent_state_roots,
            to_play
        )
    else:
        # 使用原版 Python 搜索
        self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)
```

## 注意事项

1. **模型导出**: 需要将 PyTorch 模型导出为 TorchScript 格式
2. **支持范围**: 需要在 C++ 中配置正确的 value/reward support 范围
3. **CUDA 版本**: libtorch 的 CUDA 版本需要和训练环境一致
4. **首次推理**: 第一次推理会慢（模型加载），后续会快

## 性能测试

```python
import time
from lzero.mcts.ctree.ctree_gumbel_muzero import gmz_tree_cpp

# 测试 C++ 搜索耗时
start = time.perf_counter()
for _ in range(100):
    gmz_tree_cpp.search_with_model(...)
cpp_time = (time.perf_counter() - start) / 100

print(f"C++ 搜索平均耗时: {cpp_time*1000:.2f} ms")
```

## 改动量评估

| 组件 | 改动量 | 说明 |
|------|--------|------|
| cnode_with_model.h | 新增 ~100 行 | C++ 头文件 |
| cnode_with_model.cpp | 新增 ~200 行 | C++ 实现 |
| bindings.cpp | 新增 ~80 行 | pybind11 绑定 |
| CMakeLists.txt | 新增 ~40 行 | 编译配置 |
| 模型导出脚本 | 新增 ~30 行 | TorchScript 导出 |
| Policy 修改 | ~10 行 | 调用 C++ 搜索 |

**总计**: ~460 行新代码 + 编译配置
