# utils/device.py 通用设备工具
import torch
from config import SEED

def get_device() -> torch.device:
    """获取可用设备（优先CUDA）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("⚠️  未检测到CUDA，将使用CPU测试（显存优势无法体现）")
    return device

def set_seed(device: torch.device) -> None:
    """设置全局随机种子，保证实验可复现"""
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 固定算法，保证可复现

__all__ = ["get_device", "set_seed"]