# models/fused_la.py 融合算子版LinearAttention（继承nn.Module）
import torch
import torch.nn as nn
import torch.autograd as autograd

class LightningAttentionFunction(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, block_size=64):
        B, L, D = Q.shape
        assert L % block_size == 0, "序列长度必须是块大小的整数倍！"
        ctx.block_size = block_size
        ctx.save_for_backward(Q, K, V)

        S = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
        O = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)
        num_blocks = L // block_size
        M = torch.tril(torch.ones(block_size, block_size, device=Q.device, dtype=Q.dtype))

        for t in range(num_blocks):
            start = t * block_size
            end = (t + 1) * block_size

            Qt = Q[:, start:end, :] # B*block_size*D
            Kt = K[:, start:end, :] # B*block_size*D
            Vt = V[:, start:end, :] # B*block_size*D
            Kt_T = Kt.transpose(1, 2) # B*D*block_size

            O_inter = torch.bmm(Qt, S)
            attn = torch.bmm(Qt, Kt_T) * M
            O_intra = torch.bmm(attn, Vt)
            O[:, start:end, :] = O_inter + O_intra
            S = S + torch.bmm(Kt_T, Vt)

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V = ctx.saved_tensors
        B, L, D = Q.shape
        block_size = ctx.block_size
        dQ = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)
        dK = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)
        dV = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)

        S = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
        dS = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
        num_blocks = L // block_size
        M = torch.tril(torch.ones(block_size, block_size, device=Q.device, dtype=Q.dtype))

        # 前向重走保存块信息
        for t in range(num_blocks):
            start = t * block_size
            end = (t + 1) * block_size
            Qt = Q[:, start:end, :]
            Kt = K[:, start:end, :]
            Vt = V[:, start:end, :]
            dOt = dO[:, start:end, :]
            Kt_T = Kt.transpose(1, 2)
            dQ_1 = torch.bmm(dOt, S.transpose(1, 2))
            attn = torch.bmm(dOt, Vt.transpose(1, 2)) * M
            dQ_2 = torch.bmm(attn, Kt)
            S = S + torch.bmm(Kt_T, Vt)
            dQ[:, start:end, :] = dQ_1 + dQ_2

        # 反向从最后一块计算
        for t in reversed(range(num_blocks)):
            start = t * block_size
            end = (t + 1) * block_size
            Qt = Q[:, start:end, :]
            Kt = K[:, start:end, :]
            Vt = V[:, start:end, :]
            dOt = dO[:, start:end, :]

            dK_1 = torch.bmm(Vt, dS.transpose(1, 2))
            attn_k = torch.bmm(dOt, Vt.transpose(1, 2)) * M
            dK_2 = torch.bmm(attn_k.transpose(1, 2), Qt)

            dV_1 = torch.bmm(Kt, dS)
            attn_v = torch.bmm(Qt, Kt.transpose(1, 2)) * M
            dV_2 = torch.bmm(attn_v.transpose(1, 2), dOt)

            # 累积梯度
            dK[:, start:end, :] = dK_1 + dK_2
            dV[:, start:end, :] = dV_1 + dV_2

            # 更新梯度状态
            dS = dS + torch.bmm(Qt.transpose(1, 2), dOt)

        return dQ, dK, dV, None

# 融合算子封装为可直接调用的模型类
class FusedLightningAttention(nn.Module):
    def __init__(self, d_model: int, block_size: int = 64):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        return LightningAttentionFunction.apply(Q, K, V, self.block_size)

__all__ = ["FusedLightningAttention"]