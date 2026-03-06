import torch

from konfai.data.data_manager import Attribute
from konfai.data.transform import Transform


class UnNormalize(Transform):

    def __init__(self) -> None:
        super().__init__()
        self.v_min = -1024
        self.v_max = 3071

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return (tensor + 1) / 2 * (self.v_max - self.v_min) + self.v_min

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass
