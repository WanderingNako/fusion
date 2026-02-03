import torch
import torch.nn as nn

class NavieTTT(nn.Module):
    def __init__(self, d_model: int, eta=0.9):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.eta = eta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        B, L, D = Q.shape

        # 全局隐状态S
        W = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
        O = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)
        for t in range(L):
            Kt = K[:, t, :].unsqueeze(1)  # B*1*D
            Vt = V[:, t, :].unsqueeze(1)  # B*1*D
            T = torch.bmm(Kt, W) - Vt
            G = torch.bmm(Kt.transpose(1, 2), T) * 2
            W = W - self.eta * G          # B*D*D
            Qt = Q[:, t, :].unsqueeze(1)  # B*1*D
            Ot = torch.bmm(Qt, W)         # B*1*D
            O[:, t, :] = Ot.squeeze(1)    # B*D
        return O

__all__ = ["NavieTTT"]