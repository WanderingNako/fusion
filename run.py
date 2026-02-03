# run.py 统一执行入口
import torch
from models import (
    FusedLightningAttention, 
    NavieLightningAttention,
    FusedRetention,
    NavieRetention
)
from utils import get_device, set_seed, general_grad_check, measure_model_performance
from config import *

def main():
    # 1. 初始化设备和随机种子
    device = get_device()
    set_seed(device)
    print(f"📌 运行设备：{device}")
    print(f"📌 测试参数：batch_size={BATCH_SIZE}, d_model={D_MODEL}, block_size={BLOCK_SIZE}")
    print("-" * 90)

    # 2. 定义待测试模型映射【核心：新增模型只需在这里添加键值对】
    # 格式：{模型名称: (模型类, 模型初始化参数字典)}
    MODEL_MAP = {
        #"NavieLightningAttention": (NavieLightningAttention, {"d_model": D_MODEL}),
        #"FusedLightningAttention": (FusedLightningAttention, {"d_model": D_MODEL, "block_size": BLOCK_SIZE})
        "NavieRetention": (NavieRetention, {"d_model": D_MODEL, "gamma": 0.9}),
        "FusedRetention": (FusedRetention, {"d_model": D_MODEL, "gamma": 0.9, "block_size": BLOCK_SIZE})
    }

    # 3. 通用梯度校验（短序列，支持任意模型）
    print("🔍 开始全局梯度校验（短序列）...")
    # 生成梯度校验专用输入（requires_grad=True）
    grad_x = torch.randn(
        BATCH_SIZE, GRAD_CHECK_SEQ_LEN, D_MODEL,
        device=device, requires_grad=True,
        dtype=torch.double
    )
    # 遍历所有模型做梯度校验
    for model_name, (model_cls, model_kwargs) in MODEL_MAP.items():
        model = model_cls(** model_kwargs).to(device)
        model = model.double()
        torch.cuda.synchronize()
        general_grad_check(model, grad_x)
    print("-" * 90)

    # 4. 批量性能测试（不同序列长度，支持任意模型）
    print("📊 开始模型性能测试（显存+时间）...")
    # 打印对齐的表格表头
    header = f"{'序列长度':<10} {'模型版本':<15} {'峰值显存(GB)':<18} {'前向平均(ms)':<18} {'反向平均(ms)':<18}"
    print(header)
    print("-" * 90)

    # 遍历每个序列长度测试
    for seq_len in SEQ_LENS:
        # 生成固定输入（所有模型共享，保证测试公平）
        x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device)
        # 遍历每个模型测试性能
        for model_name, (model_cls, model_kwargs) in MODEL_MAP.items():
            model = model_cls(** model_kwargs)
            try:
                # 测量性能（通用函数，支持任意模型）
                peak_mem, forward_avg, backward_avg = measure_model_performance(model, x, device)
                # 格式化输出结果（左对齐+保留2位小数）
                print(
                    f"{seq_len:<10} {model_name:<15} "
                    f"{peak_mem:<18.2f} {forward_avg:<18.2f} {backward_avg:<18.2f}"
                )
            except RuntimeError as e:
                # 捕获OOM错误，友好输出
                if "out of memory" in str(e):
                    print(f"{seq_len:<10} {model_name:<15} {'OOM':<18} {'OOM':<18} {'OOM':<18}")
                else:
                    raise e  # 其他错误正常抛出
        print("-" * 90)

    print("🎉 所有测试完成！")

if __name__ == "__main__":
    main()