import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch


@dataclass
class StockModel:
    symbols: Tuple[str]
    network: torch.nn.Module
    date: dt.datetime
    epochs: int
    loss: float
    encoding: Dict[str, int] = field(default_factory=dict)
