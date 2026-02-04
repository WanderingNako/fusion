import torch
import torch.nn as nn
import torch.autograd as autograd

class RetentionFunction(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, gamma=0.9, block_size=64):
        B, L, D = Q.shape
        assert L % block_size == 0, "序列长度必须是块大小的整数倍！"
        ctx.gamma = gamma
        ctx.block_size = block_size
        ctx.save_for_backward(Q, K, V)

        R = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
        O = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)

        num_blocks = L // block_size
        zeta = torch.zeros(block_size, D, device=Q.device, dtype=Q.dtype)
        xi   = torch.zeros(block_size, D, device=Q.device, dtype=Q.dtype)
        D    = torch.zeros(block_size, block_size, device=Q.device, dtype=Q.dtype)
        for i in range(block_size):
            zeta[i, :] = gamma ** (block_size - i - 1)
            xi[i, :] = gamma ** (i + 1)
            for j in range(i + 1):
                D[i, j] = gamma ** (i - j)
        
        for t in range(num_blocks):
            start = t * block_size
            end = (t + 1) * block_size

            Qt = Q[:, start:end, :] # B*block_size*D
            Kt = K[:, start:end, :] # B*block_size*D
            Vt = V[:, start:end, :] # B*block_size*D
            Kt_T = Kt.transpose(1, 2) # B*D*block_size

            attn = torch.bmm(Qt, Kt_T) * D
            O_1 = torch.bmm(attn, Vt)
            O_2 = torch.bmm(Qt, R) * xi
            O[:, start:end, :] = O_1 + O_2

            attn_r = torch.bmm(Kt_T, Vt * zeta)
            R = R * (gamma ** block_size) + attn_r

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V = ctx.saved_tensors
        B, L, D = Q.shape
        gamma = ctx.gamma
        block_size = ctx.block_size
        dQ = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)
        dK = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)
        dV = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)

        R = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
        dR = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)

        num_blocks = L // block_size
        zeta = torch.zeros(block_size, D, device=Q.device, dtype=Q.dtype)
        xi   = torch.zeros(block_size, D, device=Q.device, dtype=Q.dtype)
        D    = torch.zeros(block_size, block_size, device=Q.device, dtype=Q.dtype)
        for i in range(block_size):
            zeta[i, :] = gamma ** (block_size - i - 1)
            xi[i, :] = gamma ** (i + 1)
            for j in range(i + 1):
                D[i, j] = gamma ** (i - j)
        
        # 前向重走保存块信息
        for t in range(num_blocks):
            start = t * block_size
            end = (t + 1) * block_size
            Qt = Q[:, start:end, :]
            Kt = K[:, start:end, :]
            Vt = V[:, start:end, :]
            dOt = dO[:, start:end, :]
            Kt_T = Kt.transpose(1, 2)
            dQ_1 = torch.bmm(dOt * xi, R.transpose(1, 2))
            attn = torch.bmm(dOt, Vt.transpose(1, 2)) * D
            dQ_2 = torch.bmm(attn, Kt)
            R = R * (gamma ** block_size) + torch.bmm(Kt_T, Vt * zeta)
            dQ[:, start:end, :] = dQ_1 + dQ_2

        # 反向从最后一块计算
        for t in reversed(range(num_blocks)):
            start = t * block_size
            end = (t + 1) * block_size
            Qt = Q[:, start:end, :]
            Kt = K[:, start:end, :]
            Vt = V[:, start:end, :]
            dOt = dO[:, start:end, :]

            dK_1 = torch.bmm(Vt * zeta, dR.transpose(1, 2))
            attn_k = torch.bmm(dOt, Vt.transpose(1, 2)) * D
            dK_2 = torch.bmm(attn_k.transpose(1, 2), Qt)

            dV_1 = torch.bmm(Kt, dR) * zeta
            attn_v = torch.bmm(Qt, Kt.transpose(1, 2)) * D
            dV_2 = torch.bmm(attn_v.transpose(1, 2), dOt)

            # 累积梯度
            dK[:, start:end, :] = dK_1 + dK_2
            dV[:, start:end, :] = dV_1 + dV_2

            # 更新梯度状态
            dR = dR * (gamma ** block_size) + torch.bmm(Qt.transpose(1, 2), dOt * xi)

        return dQ, dK, dV, None, None

# 融合算子封装为可直接调用的模型类
class FusedRetention(nn.Module):
    def __init__(self, d_model: int, gamma=0.9, block_size: int = 64):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.gamma = gamma
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        return RetentionFunction.apply(Q, K, V, self.gamma, self.block_size)

__all__ = ["FusedRetention"]
