
import torch
import torch.nn as nn

from liquidnet.rate_func.non_linearity import NonLinearity
from liquidnet.utils.mapping_strategy import LinearMapping, AffineMapping



class LiquidCell(nn.Module):

    def __init__(self, input_size,num_units,device='cpu',config=None):
        super(LiquidCell, self).__init__()

        self.device = device
        self._input_size = input_size
        self._num_units = num_units
        self.hidden_size = num_units
        self._is_built = False

        # Number of ODE solver steps in one RNN step
        self._ode_solver_unfolds = 12
        mapping_strategy = AffineMapping
        self.nonfunc = NonLinearity(input_size=input_size, num_units=num_units,
                                     config=config,ufolds=self._ode_solver_unfolds, device=device)

        #mapping
        # 初始化权重和偏置参数，只在第一次调用时创建并复用
        self.input_mapper = self._init_mapper(mapping_strategy, input_size, device=device)  # 使用默认映射

    @staticmethod
    def _init_mapper(mapping_strategy, mapping_input_size, device):
        # 初始化映射策略（如果是类则实例化）
        if mapping_strategy is None:
            input_mapper = LinearMapping(input_size=mapping_input_size, device=device)
        else:
            input_mapper = mapping_strategy(input_size=mapping_input_size, device=device)
        return input_mapper


    def forward(self, input, hx = None):
        """Forward pass through the cell"""
        inputs = input
        if not self._is_built:
            # TODO: Move this part into the build method inherited form nn.Module
            self._is_built = True
            self._input_size = int(inputs.shape[-1])
            # self._get_variables()

        elif self._input_size != int(inputs.shape[-1]):
            raise ValueError(
                "You first feed an input with {} features and now one with {} features, that is not possible".format(
                    self._input_size, int(inputs[-1])
                )
            )

        inputs = self.input_mapper(inputs)
        if hx is None:
            state = torch.zeros(inputs.shape[0], self._num_units, device=inputs.device, dtype=inputs.dtype)
        else:
            state = hx
        next_state = self.nonfunc.ode_step(inputs, state)

        state = next_state
        return next_state, state

    def get_param_constrain_op(self):
        """Constrains the parameters to the specified bounds."""
        with torch.no_grad():  # 禁止追踪梯度，直接修改参数
            self.nonfunc.get_param_constrain_op()



