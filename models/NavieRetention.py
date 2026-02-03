import torch
import torch.nn as nn

class NavieRetention(nn.Module):
    def __init__(self, d_model: int, gamma=0.9):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        B, L, D = Q.shape

        # 全局隐状态S
        S = torch.zeros(B, D, D, device=Q.device, dtype=Q.dtype)
        O = torch.empty(B, L, D, device=Q.device, dtype=Q.dtype)
        A = torch.ones(B, D, D, device=Q.device, dtype=Q.dtype) * self.gamma  # retention decay factor
        for t in range(L):
            Kt = K[:, t, :].unsqueeze(2)  # B*D*1
            Vt = V[:, t, :].unsqueeze(1)  # B*1*D
            S = torch.bmm(A, S) + torch.bmm(Kt, Vt)        # B*D*D
            Qt = Q[:, t, :].unsqueeze(1)  # B*1*D
            Ot = torch.bmm(Qt, S)         # B*1*D
            O[:, t, :] = Ot.squeeze(1)    # B*D
        return O

__all__ = ["NavieRetention"]