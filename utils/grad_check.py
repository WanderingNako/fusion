# utils/grad_check.py 通用梯度校验（支持任意PyTorch模型）
import torch
from config import GRAD_CHECK_ATOL, GRAD_CHECK_EPS
from torch.autograd.gradcheck import GradcheckError

def general_grad_check(model: torch.nn.Module, x: torch.Tensor) -> None:
    """
    通用梯度校验函数
    :param model: 任意PyTorch模型实例（已移至指定设备）
    :param x: 输入张量（requires_grad=True，已移至指定设备）
    :raise: 梯度误差过大时触发AssertionError
    """
    model.eval()  # 校验时用评估模式，避免Dropout/BatchNorm干扰
    try:
        # torch内置数值梯度校验：对比模型自动/手动梯度 与 有限差分法数值梯度
        torch.autograd.gradcheck(
            func=lambda t: model(t),  # 模型前向函数
            inputs=x,                # 输入张量
            eps=GRAD_CHECK_EPS,      # 数值梯度步长
            atol=GRAD_CHECK_ATOL,    # 绝对误差容忍度
            fast_mode=True           # 快速模式，减少计算量
        )
        print(f"✅ 模型【{model.__class__.__name__}】梯度校验通过！")
    except GradcheckError as e:
        raise RuntimeError(f"❌ 模型【{model.__class__.__name__}】梯度校验失败，误差超出阈值！")

__all__ = ["general_grad_check"]