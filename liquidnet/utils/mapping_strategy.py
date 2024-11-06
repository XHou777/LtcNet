import torch
import torch.nn as nn

class MappingStrategy(nn.Module):
    """映射策略基类，允许用户定义不同的映射逻辑"""
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size


    def forward(self, *args, **kwargs):
        pass

class LinearMapping(MappingStrategy):
    """线性映射：元素乘法"""
    def __init__(self, input_size,device):
        super().__init__(input_size)
        self.w = nn.Parameter(torch.ones(input_size,device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.w

class IdentityMapping(MappingStrategy):
    """线性映射：元素乘法"""
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

class AffineMapping(MappingStrategy):
    """仿射映射：元素乘法加偏置"""
    def __init__(self, input_size,device):
        super().__init__(input_size)
        self.w = nn.Parameter(torch.ones(input_size,device=device))
        self.b = nn.Parameter(torch.zeros(input_size,device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.w + self.b
