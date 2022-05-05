import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict

import torch


@dataclass
class StockModel:
    symbols: List[str]
    network: torch.nn.Module
    date: dt.datetime
    epochs: int
    loss: float
    encoding: Dict[str, int] = field(default_factory=dict)
