import torch
from torch import nn


class Initializer:
    def __init__(self, device):
        self.device = device

    def uniform_param(self, shape, low, high, requires_grad=True):
        tensor = torch.empty(shape).to(self.device)
        nn.init.uniform_(tensor, a=low, b=high)
        return nn.Parameter(tensor,requires_grad=requires_grad)

    def binary_param(self, shape, factor):
        binary_init = 2 * torch.randint(low=0, high=2, size=shape).float() - 1 * factor
        tensor = self.constant_param(
            shape=shape,
            value=binary_init
        )
        return tensor

    def constant_param(self, shape, value,requires_grad=True):
        # 检查 value 是不是一个标量
        if isinstance(value, (int, float)):
            tensor = torch.empty(shape).to(self.device)
            nn.init.constant_(tensor, val=value)
        else:
            # 如果 value 是一个张量，则直接将其转换为 nn.Parameter
            tensor = value.to(self.device)
        return nn.Parameter(tensor,requires_grad=requires_grad)


class ParameterInitializer:
    def __init__(self, input_size, num_units, device, config):
        self.input_size = input_size
        self.num_units = num_units
        self.device = device
        self.config = config
        self.initializer = Initializer(device)
        self.nonfunc_params = {}  # 用于存储初始化好的参数
        self.sysa_params = {}  # 用于存储初始化好的参数

    def initialize_synapseactivation_parameters(self,sub_name,shape):
        w_init_min = self.config['w_init_min']
        w_init_max = self.config['w_init_max']
        init_methods = {
            f"{sub_name}mu": lambda: self.initializer.uniform_param(shape, 0.3, 0.5),
            f"{sub_name}sigma": lambda: self.initializer.uniform_param(shape, 3.0, 5.0),
            f"{sub_name}W": lambda: self.initializer.constant_param(
                shape=(self.input_size, self.num_units),
                value=torch.empty((self.input_size, self.num_units)).uniform_(w_init_min, w_init_max)
            )
        }
        params = {}
        for param_name, init_method in init_methods.items():
            if sub_name in param_name:
                param_name = param_name.replace(sub_name, "")
            params[param_name] = init_method()
        return params

    def initialize_nonfunc_parameters(self):
        erev_init_factor = self.config['erev_init_factor']
        init_methods = {
            "sensory_erev": lambda: self.initializer.binary_param(
                shape=(self.num_units, self.num_units),factor=erev_init_factor),
            "erev": lambda: self.initializer.binary_param(
                shape=(self.num_units, self.num_units),factor=erev_init_factor),
            "vleak": self.init_vleak,
            "gleak": self.init_gleak,
            "cm_t": self.init_cm_t
        }
        params = {}
        for param_name, init_method in init_methods.items():
            params[param_name] = init_method()
        return params

    def init_vleak(self):
        if self.config.get('fix_vleak') is not None:
            return self.initializer.constant_param(shape=(self.num_units,),value=self.config['fix_vleak'])
        return self.initializer.uniform_param((self.num_units,), -0.2, 0.2)

    def init_gleak(self):
        gleak_shape = (self.num_units,)
        fix_gleak = self.config['fix_gleak']
        gleak_init_max = self.config['gleak_init_max']
        gleak_init_min = self.config['gleak_init_min']
        if fix_gleak is None:
            if gleak_init_max > gleak_init_min:
                gleak = self.initializer.uniform_param(shape=gleak_shape, low=gleak_init_min,
                                                       high=gleak_init_max)
            else:
                gleak = self.initializer.constant_param(shape=gleak_shape, value=gleak_init_min)
        else:
            gleak = self.initializer.constant_param(shape=gleak_shape, value=fix_gleak)
        return gleak

    def init_cm_t(self):
        cm_t_shape = (self.num_units,)
        if self.config.get('fix_cm') is not None:
            return self.initializer.constant_param(shape=cm_t_shape,value=self.config['fix_cm'])
        if self.config['cm_init_max'] > self.config['cm_init_min']:
            return self.initializer.uniform_param(shape=cm_t_shape, low=self.config['cm_init_min'], high=self.config['cm_init_max'])
        return self.initializer.constant_param(shape=cm_t_shape,value=self.config['cm_init_min'])