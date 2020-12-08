from typing import Optional
import numpy as np
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F
from .covid_lin import run_aligator


class Aligator(nn.Module):
    def __init__(self, 
                 cond_size: int,
                 pred_size: int,
                 *lr: float,
    ) -> None:
        super(Aligator, self).__init__()
        self.cond_size = cond_size
        self.pred_size = pred_size
        if len(lr) > 0:
            self.lr = lr
        else:
            self.lr = [
                1e-2, 8e-2, 1e-1, 1.5e-1,
                2e-1, 2.5e-1, 1e0,
            ]
    
    def forward(self, data: Tensor):
        steps = data.size(1)
        min_err = float('inf')
        best_lr = None
        sm = []
        pr = []
        for s in range(data.size(0)):
            series = data[s].cpu().numpy()
            delta = np.max(series) - np.min(series)
            for sigma in self.lr:
                filtered = run_aligator(
                    steps, series, np.arange(steps),
                    sigma, delta, 0, -1,
                )
                residue = series - filtered
                err = np.mean(residue**2)
                if err < min_err:
                    min_err = err
                    best_lr = sigma
                    smoothed = filtered
            sm.append(smoothed)
            preds = run_aligator(
                steps, series, np.arange(steps),
                best_lr, delta, 0, self.pred_size
            )
            pr.append(preds)
        
        sm = np.stack(sm, axis=0)
        sm = data.new_tensor(sm)
        pr = np.stack(pr, axis=0)
        pr = data.new_tensor(pr)
        sm = data - sm
        return sm, pr