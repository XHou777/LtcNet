import torch
from torch import nn

from liquidnet.utils.init_params import ParameterInitializer




class SynapseActivation(nn.Module):
    def __init__(self,param = None,w_min_value=0.00001,w_max_value=1000):
        super(SynapseActivation, self).__init__()
        self.w_min_value = w_min_value
        self.w_max_value = w_max_value
        for name, param in param.items():
            setattr(self, name, param)

    def _sigmoid(self, x):
        x = x.view(-1, x.shape[-1], 1)
        mues = x - self.mu
        x = self.sigma * mues
        return torch.sigmoid(x)

    def forward(self, v_pre):
        # Linear transformation followed by custom sigmoid activation
        w_activation = self.W * self._sigmoid(v_pre)
        # Sum across the specified dimension
        w_reduced_synapse = torch.sum(w_activation, dim=1)

        return w_activation, w_reduced_synapse

    def get_param_constrain_op(self):
        self.W.data = torch.clamp(self.W.data, self.w_min_value, self.w_max_value)


class NonLinearity(nn.Module):
    def __init__(self,input_size,num_units, config=None,ufolds=12,device='cpu'):
        super(NonLinearity, self).__init__()
        # 使用 ParameterInitializer 自动初始化参数
        self.config = config if config else {
            'w_init_min': 0.01,
            'w_init_max': 1.0,
            'erev_init_factor': 1,
            'gleak_init_min': 1,
            'gleak_init_max': 1,
            'cm_init_min': 0.5,
            'cm_init_max': 0.5,
            'fix_vleak': None,
            'fix_gleak': None,
            'fix_cm': None,
            'w_min_value': 0.00001,
            'w_max_value': 1000,
            'gleak_min_value': 0.00001,
            'gleak_max_value': 1000,
            'cm_t_min_value': 0.000001,
            'cm_t_max_value': 1000,
        }
        self.ufolds = ufolds

        param_initializer = ParameterInitializer(input_size, num_units, device, self.config)
        nonfunc_params = param_initializer.initialize_nonfunc_parameters()

        # 将初始化后的参数赋值到模型中
        for name, param in nonfunc_params.items():
            setattr(self, name, param)

        # 初始化 SynapseActivation 模块
        w_min_value = self.config['w_min_value']
        w_max_value = self.config['w_max_value']

        sensory_params = param_initializer.initialize_synapseactivation_parameters(sub_name='sensory_',shape=(input_size, num_units))
        self.sensory_activation = SynapseActivation(param = sensory_params,w_min_value=w_min_value,w_max_value=w_max_value)

        o_params = param_initializer.initialize_synapseactivation_parameters(sub_name='',shape=(num_units, num_units))
        self.synapse_activation = SynapseActivation(param = o_params,w_min_value=w_min_value,w_max_value=w_max_value)


    def _sigmoid(self, x, mu, sigma):
        x = x.view(-1, x.shape[-1], 1)
        mues = x - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def f_prime(self, inputs, state):
        v_pre = state

        # Pre-compute sensory activations 计算来自输入的突触激活
        sensory_w_activation,w_reduced_sensory = self.sensory_activation(inputs)

        # Calculate synaptic activation  计算来自当前状态的突触激活
        w_activation,w_reduced_synapse = self.synapse_activation(v_pre)

        #通过上述计算，分别得到系统对输入和状态的依赖

        # Calculate sensory and synaptic input currents 计算感觉输入电流。感觉输入电流是输入的激活值乘以逆转电位
        sensory_in = self.sensory_erev * sensory_w_activation
        # 计算突触输入电流。突触输入电流是突触激活值乘以逆转电位
        synapse_in = self.erev * w_activation

        # Calculate total input current	•	sum_in 是感觉和突触输入电流的总和，表达了输入和状态对当前电流的共同影响。
        sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
        )
        # Compute the derivative f_prime
        f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

        return f_prime

    def ode_step(self,inputs,state):
        """ODE solver step"""
        v_pre = state

        sensory_w_activation,w_denominator_sensory = self.sensory_activation(inputs)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)

        for t in range(self.ufolds):

            w_activation,w_reduced_synapse = self.synapse_activation(v_pre)
            rev_activation = w_activation * self.erev


            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = w_reduced_synapse + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / (denominator + 1e-8)

        return v_pre

    def get_param_constrain_op(self):
        """Constrains the parameters to the specified bounds."""
        self.cm_t.data = torch.clamp(self.cm_t.data, self.config['cm_t_min_value'], self.config['cm_t_max_value'])
        self.gleak.data = torch.clamp(self.gleak.data, self.config['gleak_min_value'],
                                      self.config['gleak_max_value'])
        self.synapse_activation.get_param_constrain_op()
        self.sensory_activation.get_param_constrain_op()



