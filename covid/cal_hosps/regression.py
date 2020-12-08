import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F


class CausalRegression(nn.Module):
    def __init__(self, 
                 cond_size: int,
                 pred_size: int,
                 n_location: int,
                 n_output: int,
                 s_window: int,
                 d_hidden: int,
    ) -> None:
        super(CausalRegression, self).__init__()
        self.cond_size = cond_size
        self.pred_size = pred_size
        self.n_location = n_location
        self.s_window = s_window
        self.d_hidden = d_hidden
        self.n_output = n_output
        
        self.temporal_weight = nn.Parameter(Tensor(n_output*pred_size, d_hidden, cond_size))
        self.ma_weight = nn.Parameter(Tensor(d_hidden, 1, s_window))
        self._reset_parameters()
        
    def _reset_parameters(self) -> None:
        init_weight = Tensor(self.pred_size).uniform_()
        init_weight = F.softmax(init_weight, dim=0)
        weights = []
        for day in range(self.cond_size):
            weights.append(init_weight)
            init_weight = init_weight[:-1]
            init_weight = pt.cat([
                1.0-pt.sum(init_weight, dim=0, keepdim=True), 
                init_weight,
            ], dim=0)
        weights = pt.stack(weights, dim=1)
        weights = pt.stack([weights] * self.d_hidden, dim=1)
        weights = pt.cat([weights] * self.n_output, dim=0)
        with pt.no_grad():
            self.temporal_weight.copy_(weights)
        
        init_weight = Tensor(self.d_hidden, 1, self.s_window)
        init.xavier_uniform_(init_weight)
        init_weight = F.softmax(init_weight, dim=2)
        with pt.no_grad():
            self.ma_weight.copy_(init_weight)

    def forward(self, new_cases: Tensor) -> Tensor:
        cases = new_cases.unsqueeze(dim=1)
        hidden = F.conv1d(cases, self.ma_weight)
        hidden = F.relu(hidden)
        preds = F.conv1d(hidden, self.temporal_weight)
        preds = preds.transpose(-1,-2)
        return preds