import torch as pt
from torch import Tensor, nn


class MovingAverage(nn.Module):
    def __init__(self,
                 cond_size: int,
                 pred_size: int,
    ) -> None:
        super().__init__()
        self.left_window = cond_size // 2
        self.right_window = cond_size - self.left_window
        
    def forward(self, data: Tensor):
        smoothing = []
        for step in range(data.size(1)):
            left = max(step - self.left_window, 0)
            right = min(step + self.right_window, data.size(1))
            ma = data[:, left:right].sum(dim=1).div(right-left)
            smoothing.append(ma)
        smoothing = pt.stack(smoothing, dim=1)
        residue = data - smoothing
        
        return smoothing, residue