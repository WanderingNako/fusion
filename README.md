# Fusion: PyTorch 融合算子对比基准

本仓库实现并对比若干基于块分解的融合算子与朴素实现，包含梯度校验与显存/时间性能测试。

## 快速执行
- 运行所有校验与性能测试：  
  ```sh
  python3 run.py
  ```
  ```
    📌 运行设备： NVIDIA A800-SXM4-80GB
    📌 测试参数： batch_size=2, d_model=512, block_size=64
    ------------------------------------------------------------------------------------------
    🔍 开始全局梯度校验（短序列） ...
    ✅ 模型【NavieLightningAttention】梯度校验通过！
    ✅ 模型【FusedLightningAttention】梯度校验通过！
    ------------------------------------------------------------------------------------------
    📊 开始模型性能测试（显存+时间） ...
    序列长度       模型版本            峰值显存(GB)           前向平均(ms)           反向平均(ms)          
    ------------------------------------------------------------------------------------------
    1024       NavieLightningAttention 2.06               110.63             368.69            
    1024       FusedLightningAttention 0.06               2.18               6.80              
    ------------------------------------------------------------------------------------------
    2048       NavieLightningAttention 4.10               222.59             752.73            
    2048       FusedLightningAttention 0.09               3.95               12.57             
    ------------------------------------------------------------------------------------------
    4096       NavieLightningAttention 8.19               456.26             1529.21           
    4096       FusedLightningAttention 0.15               7.52               24.04             
    ------------------------------------------------------------------------------------------
    🎉 所有测试完成！
  ```

## 要求
- Python 3.12+
- PyTorch 2.8.0+

## 主要文件与符号
- 配置
  - [config.py](config.py) — 全局参数（例如 [`config.D_MODEL`](config.py), [`config.BLOCK_SIZE`](config.py), [`config.SEQ_LENS`](config.py)）。
- 入口与流程
  - [run.py](run.py) — 测试入口，模型映射在 `MODEL_MAP` 中（新增模型只需在此添加），调用 [`utils.general_grad_check`](utils/grad_check.py) 与 [`utils.measure_model_performance`](utils/performance.py)。
- 模型实现（模型类均位于 `models/` 下）
  - [`models.FusedLightningAttention`](models/FusedLightningAttention.py) — 融合版本 Linear Attention（`FusedLightningAttention`）。
  - [`models.NavieLightningAttention`](models/NavieLightningAttention.py) — 朴素版本 Linear Attention（`NavieLightningAttention`）。
  - [`models.FusedRetention`](models/FusedRetention.py) — 融合版本 Retention（`FusedRetention`）。
  - [`models.NavieRetention`](models/NavieRetention.py) — 朴素版本 Retention（`NavieRetention`）。
- 工具函数（位于 `utils/`）
  - [`utils.get_device`](utils/device.py) / [`utils.set_seed`](utils/device.py) — 设备与随机数设置。
  - [`utils.general_grad_check`](utils/grad_check.py) — 通用梯度校验（基于 `torch.autograd.gradcheck`）。
  - [`utils.measure_model_performance`](utils/performance.py) — 显存与前/后向时间测量。
- 手推公式（位于 `formula/`）

## 如何添加新模型
1. 在 `models/` 中添加实现文件并导出模型类。
2. 在 [run.py](run.py) 的 `MODEL_MAP` 中添加条目：  
   `"MyModelName": (MyModelClass, {"d_model": D_MODEL, ...})`。

## 注意事项
- 性能测试依赖 CUDA 显存统计：若无 CUDA，会提示并使用 CPU。
- 跨模型比较请保持相同输入与配置。

## 相关论文
- [LightningAttention](https://arxiv.org/abs/2405.17381)
- [Retention](https://arxiv.org/abs/2307.08621)