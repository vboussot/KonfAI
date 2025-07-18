from abc import abstractmethod
from functools import partial

import numpy as np
import torch


class Scheduler:

    def __init__(self, start_value: float) -> None:
        self.baseValue = float(start_value)
        self.it = 0

    def step(self, it: int):
        self.it = it

    @abstractmethod
    def get_value(self) -> float:
        pass


class Constant(Scheduler):

    def __init__(self, value: float = 1):
        super().__init__(value)

    def get_value(self) -> float:
        return self.baseValue


class CosineAnnealing(Scheduler):

    def __init__(self, start_value: float = 1, eta_min: float = 0.00001, t_max: int = 100):
        super().__init__(start_value)
        self.eta_min = eta_min
        self.t_max = t_max

    def get_value(self):
        return self.eta_min + (self.baseValue - self.eta_min) * (1 + np.cos(self.it * torch.pi / self.t_max)) / 2


class Warmup(torch.optim.lr_scheduler.LambdaLR):

    @staticmethod
    def warmup(warmup_steps: int, step: int) -> float:
        return min(1.0, (step + 1) / (warmup_steps + 1))

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 10,
        last_epoch=-1,
        verbose="deprecated",
    ):
        super().__init__(optimizer, partial(Warmup.warmup, warmup_steps), last_epoch, verbose)
