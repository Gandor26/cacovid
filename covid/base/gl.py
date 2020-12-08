from typing import Optional, List
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F



class GlobalLocalModel(nn.Module):
    def __init__(self,
                 n_location: int,
                 quantiles: Optional[List[float]],
                 share_params: bool = False,
    ) -> None:
        self.n_location = n_location
        self.share_params = share_params
        super(GlobalLocalModel, self).__init__()
        self.quantiles = sorted(list(set(sum(
            [[q, 1-q] for q in (quantiles or []) if q != 0.5],
            [0.5],
        ))))
        self._tradeoff = nn.Parameter(pt.zeros(1 if share_params else n_location, 1, 1, 1))
        
    @property
    def tradeoff(self) -> Tensor:
        return self._tradeoff.sigmoid()

    @property
    def n_output(self) -> int:
        return len(self.quantiles)
    
    @staticmethod
    def quantile_error(
            preds: Tensor,
            target: Tensor,
            percentage: float,
    ) -> Tensor:
        diff = target - preds
        weight = pt.where(
            condition=diff>0,
            input=diff.new_tensor(percentage),
            other=diff.new_tensor(percentage-1)
        )
        return diff*weight