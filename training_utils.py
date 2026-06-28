from typing import Union
import torch
from torch import nn
from torch.nn import functional as F


# Paper: https://proceedings.neurips.cc/paper_files/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf
class GCELoss(nn.Module):
    """Generalized Cross-Entropy loss. Robust to noisy labels via the q-softmax trick."""

    def __init__(self, q: float = 0.7, reduction: str = "mean", *args, **kwargs) -> None:
        if reduction not in {"mean", "none", "sum"}:
            raise ValueError(
                f"Unsupported reduction '{reduction}'. Use one of ['mean', 'sum', 'none']."
            )
        super().__init__(*args, **kwargs)
        self.reduction: str = reduction
        self.q: float = q

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = outputs.double()
        softmax_probs: torch.Tensor = F.softmax(outputs, dim=1)
        gathered_preds: torch.Tensor = torch.gather(
            softmax_probs, dim=1, index=torch.unsqueeze(targets, 1)
        )
        unreduced_loss: torch.Tensor = (1 - gathered_preds ** self.q) / self.q

        match self.reduction:
            case "mean": return unreduced_loss.mean()
            case "sum":  return unreduced_loss.sum()
            case "none": return unreduced_loss

    def __repr__(self) -> str:
        return f"GCELoss(q={self.q}, reduction={self.reduction})"
