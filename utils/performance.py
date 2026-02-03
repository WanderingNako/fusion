# utils/performance.py 通用性能测量（支持任意PyTorch模型）
import time
import numpy as np
import torch
import torch.optim as optim
from config import WARMUP_TIMES, TEST_TIMES

def measure_model_performance(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device
) -> tuple[float, float, float]:
    """
    通用模型性能测量：峰值显存(GB)、前向平均时间(ms)、反向平均时间(ms)
    :param model: 任意PyTorch模型实例
    :param x: 输入张量（已移至指定设备）
    :param device: 运行设备
    :return: peak_mem_gb, forward_avg_ms, backward_avg_ms
    """
    model = model.to(device)
    model.train()  # 训练模式，保证与实际训练一致
    # 简单优化器（仅用于反向传播，不影响性能测量）
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 1. CUDA预热：避免初始化/首次显存分配的耗时干扰
    for _ in range(WARMUP_TIMES):
        out = model(x)
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)  # 重置显存统计

    # 2. 多次测试，记录时间
    forward_times = []
    backward_times = []
    for _ in range(TEST_TIMES):
        optimizer.zero_grad()
        # 前向传播计时
        start = time.time()
        out = model(x)
        torch.cuda.synchronize(device)
        forward_times.append((time.time() - start) * 1000)  # 转ms

        # 反向传播计时
        start = time.time()
        loss = out.sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        backward_times.append((time.time() - start) * 1000)  # 转ms

    # 3. 统计峰值显存（GB）：核心指标，反映实际内存占用
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)

    # 4. 计算平均时间（去掉最值，避免偶然误差）
    forward_arr = np.array(forward_times)
    backward_arr = np.array(backward_times)
    forward_avg = np.mean(np.sort(forward_arr)[1:-1])
    backward_avg = np.mean(np.sort(backward_arr)[1:-1])

    return peak_mem_gb, forward_avg, backward_avg

__all__ = ["measure_model_performance"]