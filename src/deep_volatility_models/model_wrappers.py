import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import torch


@dataclass
class StockModel:
    symbols: Tuple[str]
    network: torch.nn.Module
    date: dt.datetime
    epochs: int
    loss: float
    encoding: Dict[str, int] = field(default_factory=dict)
    training_data_start_date: Union[dt.datetime, None] = None
    training_data_end_date: Union[dt.datetime, None] = None
