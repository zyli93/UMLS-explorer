import torch
from torch import nn, Tensor
from torch import linalg as LA

class HyperbolicL1(nn.modules.loss._Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        dist_euc = LA.norm(input - target, dim=1)
        norm_input = LA.norm(input, dim=1)
        norm_target = LA.norm(target, dim=1)
        dist_hype = torch.acosh(1 + 2 * dist_euc ** 2 / ((1 - norm_input ** 2) * (1 - norm_target ** 2)))
        loss = torch.mean(dist_hype)
        return loss