import torch
import numpy as np
from abc import abstractmethod
from konfai.utils.config import config

class Scheduler():

    @config("Scheduler")
    def __init__(self, start_value: float) -> None:
        self.baseValue = float(start_value)
        self.it = 0
        
    def step(self, it: int):
        self.it = it

    @abstractmethod
    def get_value(self) -> float:
        pass

class Constant(Scheduler):

    @config("Constant")
    def __init__(self, value: float = 1):
        super().__init__(value)

    def get_value(self) -> float:
        return self.baseValue

class CosineAnnealing(Scheduler):
    
    @config("CosineAnnealing")
    def __init__(self, start_value: float, eta_min: float = 0.00001, T_max: int = 100):
        super().__init__(start_value)
        self.eta_min = eta_min
        self.T_max = T_max

    def get_value(self):
        return self.eta_min + (self.baseValue - self.eta_min) *(1 + np.cos(self.it * torch.pi / self.T_max)) / 2

class CosineAnnealing(Scheduler):
    
    @config("CosineAnnealing")
    def __init__(self, start_value: float, eta_min: float = 0.00001, T_max: int = 100):
        super().__init__(start_value)
        self.eta_min = eta_min
        self.T_max = T_max

    def get_value(self):
        return self.eta_min + (self.baseValue - self.eta_min) *(1 + np.cos(self.it * torch.pi / self.T_max)) / 2