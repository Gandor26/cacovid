from typing import Optional, Tuple, List, Dict
from copy import deepcopy
import pandas as pd
import numpy as np
import torch as pt
from torch import Tensor, BoolTensor

from ..data import *


def load_data(
    start_date: str,
    end_date: str,
    device: int = -1,
    test_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    hosps = load_cal_hosps_data(
        confirmed=None,
        icu=False,
        start_date=start_date,
        end_date=end_date,
    )
    cases = load_cal_cases_data(
        death=False, 
        cumulative=False,
        start_date=start_date,
        end_date=end_date,
    ).loc[:, hosps.columns]
    
    device = pt.device('cpu') if device < 0 else pt.device(f'cuda:{device}')
    data = {
        'hosps_data': pt.tensor(hosps.values.T, dtype=pt.float, device=device),
        'cases_data': pt.tensor(cases.values.T, dtype=pt.float, device=device),
    }
    if test_size is not None:
        train_data = deepcopy(data)
        train_data['hosps_data'] = train_data['hosps_data'][:, :-test_size]
        train_data['cases_data'] = train_data['cases_data'][:, :-test_size]
        valid_data = data
    else:
        train_data = data
        valid_data = None
    return train_data, valid_data