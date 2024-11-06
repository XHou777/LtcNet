import torch.optim as optim
import argparse

from liquid_dataset.gesture import GestureData
from liquid_dataset.ozone_dataloader import OzoneDataLoader
from sklearn.metrics import classification_report

import os
from enum import Enum

import torch
import torch.nn as nn


class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2


class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2


class LiquidNet2(nn.Module):
    """
    Long short-term chaos cell (LiquidNet) as described in https://arxiv.org/abs/1905.12374

    Args:
        num_units: Number of units in the cell

    Attributes:
        state_size: Integer, the number of units in the cell
        output_size: Integer, the number of units in the cell

    Call arguments:
        inputs: A 2D tensor with shape [batch_size, input_size]
        state: A 2D tensor with shape [batch_size, state_size]

    Constants:
        _ode_solver_unfolds: Number of ODE solver steps in one RNN step
        _solver: ODE solver type
        _input_mapping: Input mapping type
        _erev_init_factor: Factor for the initial value of the reversal potential
        _w_init_max: Upper bound for the initial value of the synaptic weights
        _w_init_min: Lower bound for the initial value of the synaptic weights
        _cm_init_min: Lower bound for the initial value of the membrane capacitance
        _cm_init_max: Upper bound for the initial value of the membrane capacitance
        _gleak_init_min: Lower bound for the initial value of the leak conductance
        _gleak_init_max: Upper bound for the initial value of the leak conductance
        _w_min_value: Lower bound for the synaptic weights
        _w_max_value: Upper bound for the synaptic weights
        _gleak_min_value: Lower bound for the leak conductance
        _gleak_max_value: Upper bound for the leak conductance
        _cm_t_min_value: Lower bound for the membrane capacitance
        _cm_t_max_value: Upper bound for the membrane capacitance
        _fix_cm: Fix the membrane capacitance to a specific value
        _fix_gleak: Fix the leak conductance to a specific value
        _fix_vleak: Fix the leak reversal potential to a specific value

    Variables:
        sensory_mu: A 2D tensor with shape [input_size, state_size]
        sensory_sigma: A 2D tensor with shape [input_size, state_size]
        sensory_W: A 2D tensor with shape [input_size, state_size]
        sensory_erev: A 2D tensor with shape [input_size, state_size]
        mu: A 2D tensor with shape [state_size, state_size]
        sigma: A 2D tensor with shape [state_size, state_size]
        W: A 2D tensor with shape [state_size, state_size]
        erev: A 2D tensor with shape [state_size, state_size]
        vleak: A 1D tensor with shape [state_size]
        gleak: A 1D tensor with shape [state_size]
        cm_t: A 1D tensor with shape [state_size]

    Methods:
        _map_inputs: Maps the inputs to the correct dimensionality
        build: Builds the cell
        forward: Performs a forward pass through the cell
        _get_variables: Creates the torch parameters
        _ode_step: Performs a forward pass through the cell using the semi-implicit euler method
        _f_prime: Calculates the derivative of the cell
        _ode_step_runge_kutta: Performs a forward pass through the cell using the Runge-Kutta method
        _ode_step_explicit: Performs a forward pass through the cell using the explicit euler method
        _sigmoid: Calculates the sigmoid function
        get_param_constrain_op: Returns the operations to constrain the parameters to the specified bounds
        export_weights: Exports the weights of the cell to a specified directory


    Examples:
    >>> ltc_cell = LiquidNet(64)
    >>> batch_size = 4
    >>> input_size = 32
    >>> inputs = torch.randn(batch_size, input_size)
    >>> initial_state = torch.zeros(batch_size, num_units)
    >>> outputs, final_state = ltc_cell(inputs, initial_state)
    >>> print("Outputs shape:", outputs.shape)
    >>> print("Final state shape:", final_state.shape)
    Outputs shape: torch.Size([4, 64])
    Final state shape: torch.Size([4, 64])




    """

    def __init__(self, input_size,num_units,device='cpu'):
        super(LiquidNet2, self).__init__()

        self.device = device
        self._input_size = -1
        self._num_units = num_units
        self.hidden_size = num_units
        self._is_built = False

        # Number of ODE solver steps in one RNN step
        self._ode_solver_unfolds = 6
        self._solver = ODESolver.SemiImplicit
        # self._solver = ODESolver.Explicit

        self._input_mapping = MappingType.Affine

        self._erev_init_factor = 1

        self._w_init_max = 1.0
        self._w_init_min = 0.01
        self._cm_init_min = 0.5
        self._cm_init_max = 0.5
        self._gleak_init_min = 1
        self._gleak_init_max = 1

        self._w_min_value = 0.00001
        self._w_max_value = 1000
        self._gleak_min_value = 0.00001
        self._gleak_max_value = 1000
        self._cm_t_min_value = 0.000001
        self._cm_t_max_value = 1000

        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None

        #mapping
        # 初始化权重和偏置参数，只在第一次调用时创建并复用
        self.input_w = nn.Parameter(torch.ones(input_size, device=self.device))
        self.input_b = nn.Parameter(torch.zeros(input_size, device=self.device))


    @property
    def state_size(self):
        """State size of the cell."""
        return self._num_units

    @property
    def output_size(self):
        """Output size of the cell."""
        return self._num_units

    def _map_inputs(self, inputs, resuse_scope=False):
        """Maps the inputs to the correct dimensionality"""
        if self._input_mapping in (MappingType.Affine, MappingType.Linear):
            inputs = inputs * self.input_w

        if self._input_mapping == MappingType.Affine:
            inputs = inputs + self.input_b

        return inputs
    def forward(self, input, hx = None):
        """Forward pass through the cell"""
        inputs = input
        state = hx
        if not self._is_built:
            # TODO: Move this part into the build method inherited form nn.Module
            self._is_built = True
            self._input_size = int(inputs.shape[-1])
            self._get_variables()

        elif self._input_size != int(inputs.shape[-1]):
            raise ValueError(
                "You first feed an input with {} features and now one with {} features, that is not possible".format(
                    self._input_size, int(inputs[-1])
                )
            )

        inputs = self._map_inputs(inputs)
        if state is None:
            state = torch.zeros(inputs.shape[0], self._num_units, device=inputs.device, dtype=inputs.dtype)

        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(
                inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds
            )
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step(inputs, state)
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        else:
            raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))

        outputs = next_state

        return outputs, next_state
    # Create torch parameters
    def _get_variables(self):
        """Creates the torch parameters"""
        device = self.device
        self.sensory_mu = nn.Parameter(
            torch.rand(self._input_size, self._num_units) * 0.5 + 0.3
        ).to(device)
        self.sensory_sigma = nn.Parameter(
            torch.rand(self._input_size, self._num_units) * 5.0 + 3.0
        ).to(device)
        self.sensory_W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(
                    low=self._w_init_min,
                    high=self._w_init_max,
                    size=[self._input_size, self._num_units],
                )
            )
        ).to(device)
        sensory_erev_init_np = (
            2
            * np.random.randint(low=0, high=2, size=[self._input_size, self._num_units])
            - 1
        )
        sensory_erev_init = torch.tensor(sensory_erev_init_np, dtype=torch.float32, device=device)

        self.sensory_erev = nn.Parameter(
            torch.Tensor(sensory_erev_init * self._erev_init_factor).to(device)
        )

        self.mu = nn.Parameter(torch.rand(self._num_units, self._num_units) * 0.5 + 0.3).to(device)
        self.sigma = nn.Parameter(
            torch.rand(self._num_units, self._num_units) * 5.0 + 3.0
        ).to(device)
        self.W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(
                    low=self._w_init_min,
                    high=self._w_init_max,
                    size=[self._num_units, self._num_units],
                )
            )
        ).to(device)

        erev_init_np = (
            2
            * np.random.randint(low=0, high=2, size=[self._num_units, self._num_units])
            - 1
        )
        erev_init = torch.tensor(erev_init_np, dtype=torch.float32, device=device)

        self.erev = nn.Parameter(torch.Tensor(erev_init * self._erev_init_factor)).to(device)

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.rand(self._num_units) * 0.4 - 0.2).to(device)
        else:
            self.vleak = nn.Parameter(torch.Tensor(self._fix_vleak)).to(device)

        if self._fix_gleak is None:
            if self._gleak_init_max > self._gleak_init_min:
                self.gleak = nn.Parameter(
                    torch.rand(self._num_units)
                    * (self._gleak_init_max - self._gleak_init_min)
                    + self._gleak_init_min
                ).to(device)
            else:
                self.gleak = nn.Parameter(
                    torch.Tensor([self._gleak_init_min] * self._num_units)
                ).to(device)
        else:
            self.gleak = nn.Parameter(torch.Tensor(self._fix_gleak)).to(device)

        if self._fix_cm is None:
            if self._cm_init_max > self._cm_init_min:
                self.cm_t = nn.Parameter(
                    torch.rand(self._num_units)
                    * (self._cm_init_max - self._cm_init_min)
                    + self._cm_init_min
                ).to(device)
            else:
                self.cm_t = nn.Parameter(
                    torch.Tensor([self._cm_init_min] * self._num_units)
                ).to(device)
        else:
            self.cm_t = nn.Parameter(torch.Tensor(self._fix_cm)).to(device)

    # Create torch parameters
    def _get_variables1(self):
        """Creates the torch parameters with device control"""
        self.sensory_mu = nn.Parameter(
            torch.Tensor(np.random.uniform(0.3, 0.8, size=[self._input_size, self._num_units])).to(self.device)
        )
        self.sensory_sigma = nn.Parameter(
            torch.Tensor(np.random.uniform(3.0, 8.0, size=[self._input_size, self._num_units])).to(self.device)
        )
        self.sensory_W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(self._w_init_min, self._w_init_max, size=[self._input_size, self._num_units])).to(
                self.device)
        )
        sensory_erev_init = 2 * np.random.randint(0, 2, size=[self._input_size, self._num_units]) - 1
        self.sensory_erev = nn.Parameter(torch.Tensor(sensory_erev_init * self._erev_init_factor).to(self.device))

        self.mu = nn.Parameter(
            torch.Tensor(np.random.uniform(0.3, 0.8, size=[self._num_units, self._num_units])).to(self.device)
        )
        self.sigma = nn.Parameter(
            torch.Tensor(np.random.uniform(3.0, 8.0, size=[self._num_units, self._num_units])).to(self.device)
        )
        self.W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(self._w_init_min, self._w_init_max, size=[self._num_units, self._num_units])).to(
                self.device)
        )
        erev_init = 2 * np.random.randint(0, 2, size=[self._num_units, self._num_units]) - 1
        self.erev = nn.Parameter(torch.Tensor(erev_init * self._erev_init_factor).to(self.device))

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(
                torch.Tensor(np.random.uniform(-0.2, 0.2, size=[self._num_units])).to(self.device)
            )
        else:
            self.vleak = nn.Parameter(
                torch.Tensor(self._fix_vleak).to(self.device), requires_grad=False
            )

        if self._fix_gleak is None:
            if self._gleak_init_max > self._gleak_init_min:
                gleak_values = np.random.uniform(self._gleak_init_min, self._gleak_init_max, size=[self._num_units])
            else:
                gleak_values = np.full(self._num_units, self._gleak_init_min)
            self.gleak = nn.Parameter(torch.Tensor(gleak_values).to(self.device))
        else:
            self.gleak = nn.Parameter(
                torch.Tensor(self._fix_gleak).to(self.device), requires_grad=False
            )

        if self._fix_cm is None:
            if self._cm_init_max > self._cm_init_min:
                cm_values = np.random.uniform(self._cm_init_min, self._cm_init_max, size=[self._num_units])
            else:
                cm_values = np.full(self._num_units, self._cm_init_min)
            self.cm_t = nn.Parameter(torch.Tensor(cm_values).to(self.device))
        else:
            self.cm_t = nn.Parameter(
                torch.Tensor(self._fix_cm).to(self.device), requires_grad=False
            )    # Hybrid euler method

        print("sensory_mu:", self.sensory_mu.data)
        print("sensory_sigma:", self.sensory_sigma.data)
        print("sensory_W:", self.sensory_W.data)
        print("erev:", self.erev.data)
        print('------------------------------------------------')
        print("sensory_mu:", self.sensory_mu.data.cpu().numpy())
        print("sensory_sigma:", self.sensory_sigma.data.cpu().numpy())
        print("sensory_W:", self.sensory_W.data.cpu().numpy())
        print("erev:", self.erev.data.cpu().numpy())
    def _ode_step(self, inputs, state):
        """ODE solver step"""
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return v_pre

    def _f_prime(self, inputs, state):
        """Calculates the derivative of the cell"""
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
            )

            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return f_prime

    def _ode_step_runge_kutta(self, inputs, state):
        """ODE solver step"""
        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + k1 * 0.5)
            k3 = h * self._f_prime(inputs, state + k2 * 0.5)
            k4 = h * self._f_prime(inputs, state + k3)

            state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return state

    def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds):
        """ODE solver step"""
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(_ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
            )

            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre

    def _sigmoid(self, v_pre, mu, sigma):
        """Calculates the sigmoid function"""
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def get_param_constrain_op(self):
        """Returns the operations to constrain the parameters to the specified bounds"""

        cm_clipping_op = torch.clamp(
            self.cm_t, self._cm_t_min_value, self._cm_t_max_value
        )
        gleak_clipping_op = torch.clamp(
            self.gleak, self._gleak_min_value, self._gleak_max_value
        )
        w_clipping_op = torch.clamp(self.W, self._w_min_value, self._w_max_value)
        sensory_w_clipping_op = torch.clamp(
            self.sensory_W, self._w_min_value, self._w_max_value
        )

        return [cm_clipping_op, gleak_clipping_op, w_clipping_op, sensory_w_clipping_op]

    def get_param_constrain_op1(self):
        """Constrains the parameters to the specified bounds."""
        with torch.no_grad():  # 确保约束操作不会影响梯度
            # 使用 clamp 将参数限制在指定范围内，并更新参数
            self.cm_t.data = torch.clamp(self.cm_t.data, self._cm_t_min_value, self._cm_t_max_value)
            self.gleak.data = torch.clamp(self.gleak.data, self._gleak_min_value, self._gleak_max_value)
            self.W.data = torch.clamp(self.W.data, self._w_min_value, self._w_max_value)
            self.sensory_W.data = torch.clamp(self.sensory_W.data, self._w_min_value, self._w_max_value)
    def export_weights(self, dirname, output_weights=None):
        """Exports the weights of the cell to a specified directory"""
        os.makedirs(dirname, exist_ok=True)
        w, erev, mu, sigma = (
            self.W.data.cpu().numpy(),
            self.erev.data.cpu().numpy(),
            self.mu.data.cpu().numpy(),
            self.sigma.data.cpu().numpy(),
        )
        sensory_w, sensory_erev, sensory_mu, sensory_sigma = (
            self.sensory_W.data.cpu().numpy(),
            self.sensory_erev.data.cpu().numpy(),
            self.sensory_mu.data.cpu().numpy(),
            self.sensory_sigma.data.cpu().numpy(),
        )
        vleak, gleak, cm = (
            self.vleak.data.cpu().numpy(),
            self.gleak.data.cpu().numpy(),
            self.cm_t.data.cpu().numpy(),
        )

        if output_weights is not None:
            output_w, output_b = output_weights
            np.savetxt(
                os.path.join(dirname, "output_w.csv"), output_w.data.cpu().numpy()
            )
            np.savetxt(
                os.path.join(dirname, "output_b.csv"), output_b.data.cpu().numpy()
            )
        np.savetxt(os.path.join(dirname, "w.csv"), w)
        np.savetxt(os.path.join(dirname, "erev.csv"), erev)
        np.savetxt(os.path.join(dirname, "mu.csv"), mu)
        np.savetxt(os.path.join(dirname, "sigma.csv"), sigma)
        np.savetxt(os.path.join(dirname, "sensory_w.csv"), sensory_w)
        np.savetxt(os.path.join(dirname, "sensory_erev.csv"), sensory_erev)
        np.savetxt(os.path.join(dirname, "sensory_mu.csv"), sensory_mu)
        np.savetxt(os.path.join(dirname, "sensory_sigma.csv"), sensory_sigma)
        np.savetxt(os.path.join(dirname, "vleak.csv"), vleak)
        np.savetxt(os.path.join(dirname, "gleak.csv"), gleak)
        np.savetxt(os.path.join(dirname, "cm.csv"), cm)
class DynamicRNN(nn.Module):
    def __init__(self, cell, time_major=True, model_type='lstm'):
        """
        一个与 TensorFlow dynamic_rnn 类似的 PyTorch 实现。

        参数：
            cell: 一个 RNNCell 实例。
            time_major: 控制输入是否为 [max_time, batch_size, ...] 格式。
        """
        super(DynamicRNN, self).__init__()
        self.cell = cell
        self.time_major = time_major
        self.model_type = model_type
        self.hidden_size = self.cell.hidden_size

    def forward(self, inputs, hx=None):
        """
        前向传播函数。

        参数：
            inputs: RNN 的输入。
                如果 time_major == False，输入形状应为 [batch_size, max_time, ...]。
                如果 time_major == True，输入形状应为 [max_time, batch_size, ...]。
            sequence_length: （可选）形状为 [batch_size] 的张量，表示每个序列的实际长度。
            initial_state: （可选）RNN 的初始状态。
            dtype: （可选）数据类型。

        返回：
            outputs: RNN 的输出张量。
            state: 最终的状态。
        """
        if not self.time_major:
            inputs = inputs.transpose(0, 1)  # [max_time, batch_size, ...]

        max_time,batch_size= inputs.size(0), inputs.size(1)
        model_size = self.cell.hidden_size

        # 初始化 LSTM 状态

        state = self.build_state(init_state=hx, batch_size=batch_size,
                                 hidden_size=model_size, device=inputs.device)
        outputs = []
        # 遍历每个时间步
        for time_step in range(max_time):
            input_t = inputs[time_step]  # 当前时间步的输入 [batch_size, ...]
            output,new_state= self.cell(input = input_t,hx=state)

            state = new_state
            # 如果输出的维度发生变化，则需要重新创建一个全 0 张量
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        if not self.time_major:
            outputs = outputs.transpose(0, 1)  # [batch_size, max_time, ...]

        return outputs, state

    def build_state(self, init_state,batch_size, hidden_size, device):
        model_type = self.model_type
        if init_state is not None:
            if model_type == 'lstm':
                hx = (init_state[0].unsqueeze(0), init_state[1].unsqueeze(0)) if not self.time_major else init_state
                return hx
            else:
                return init_state
        if model_type == 'lstm':
            return self.build_lstm_state(batch_size, hidden_size, device)
        else:
            return self.build_ltc_state(batch_size, hidden_size, device)

    def build_lstm_state(self, batch_size, hidden_size,device):
        # 确保 LSTM 初始状态是 (h_t, c_t) 形式的元组
        h_0 = torch.zeros(batch_size, hidden_size, device=device)
        c_0 = torch.zeros(batch_size, hidden_size, device=device)
        return (h_0, c_0)  # 返回元组 (h_t, c_t)

    def build_ltc_state(self, batch_size, hidden_size,device):
        hx = torch.zeros(batch_size, hidden_size, device=device)
        return hx

class Session:
    def __init__(self, model, optimizer, criterion, model_path="best_model.pth", device="cpu"):
        """
        初始化 PyTorchSession。
        参数：
            model: 要训练的 PyTorch 模型。
            optimizer: 优化器实例。
            criterion: 损失函数实例。
            device: 使用的设备（CPU 或 GPU）。
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_path = model_path  # 模型保存路径
        self.fc = nn.Linear(model.hidden_size, 5,device=device)


    def compute_loss_and_accuracy(self, outputs, targets):
        """
        计算损失和准确率。
        参数：
            outputs: 模型输出，形状为 [batch_size * seq_len, num_classes]
            targets: 真实标签，形状为 [batch_size * seq_len]
        返回：
            loss: 计算的损失
            accuracy: 计算的准确率
        """
        loss = self.criterion(outputs, targets)
        predictions = outputs.argmax(dim=1)
        correct_predictions = (predictions == targets).float().sum().item()
        accuracy = correct_predictions / targets.size(0) * 100  # 使用正确率计算而非比率
        return loss, accuracy

    def flatten_outputs_and_labels(self, outputs, labels):
        """
        展平模型输出和标签。
        参数：
            outputs: 模型输出，形状为 [batch_size, seq_len, num_classes]
            labels: 真实标签，形状为 [batch_size, seq_len]
        返回：
            展平后的 outputs 和 labels
        """
        outputs = outputs.view(-1, outputs.size(-1))  # [total_samples, num_classes]
        labels = labels.view(-1)  # [total_samples]
        return outputs, labels

    def train_batch(self, batch_x, batch_y, state=None):
        """
        训练单个 batch，计算 loss 和 accuracy。
        """
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

        # 前向传播
        lstm_outputs, new_state = self.model(batch_x, state)  # LSTM 输出
        outputs = self.fc(lstm_outputs)  # 通过全连接层进行分类

        # 展平输出和标签
        outputs, batch_y = self.flatten_outputs_and_labels(outputs, batch_y)

        # 计算 loss 和 accuracy
        loss, accuracy = self.compute_loss_and_accuracy(outputs, batch_y)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy, new_state

    def evaluate(self, data_loader, split):
        self.model.eval()
        all_preds, all_labels = [], []

        # 获取整个数据集
        full_x = getattr(data_loader, f"{split}_x").to(self.device)
        full_y = getattr(data_loader, f"{split}_y").to(self.device)

        with torch.no_grad():
            # 前向传播
            lstm_outputs, new_state = self.model(full_x)  # LSTM 输出
            outputs = self.fc(lstm_outputs)  # 通过全连接层进行分类

            # 展平输出和标签
            outputs, batch_y = self.flatten_outputs_and_labels(outputs, full_y)

            # 计算 loss 和 accuracy
            loss, accuracy = self.compute_loss_and_accuracy(outputs, batch_y)

            if split == "test":
                all_preds = outputs.argmax(dim=1).cpu().numpy()
                all_labels = batch_y.cpu().numpy()

        # 打印分类报告（仅在测试集上）
        if split == "test":
            print(classification_report(all_labels, all_preds))

        return accuracy , loss.item()

    def save_model(self):
        """
        保存模型到指定路径。
        """
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """
        从指定路径加载模型。
        """
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print(f"Model loaded from {self.model_path}")

class BaseModel():
    def __init__(self, input_size, model_type, model_size, output_size,seq_len,lr,device):
        super(BaseModel, self).__init__()
        self.model_type = model_type
        self.model_size = model_size
        self.input_size = input_size
        self.device = device
        self.lr = lr
        self.seq_len = seq_len

        # 定义输出层
        self.fc = nn.Linear(model_size, output_size)
        self.constrain_op = None
        self.rnn = self.build_model(model_type=model_type,input_size=input_size,model_size=model_size,device=device)

        self.optimizer = optim.Adam(self.rnn.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.session = Session(model=self.rnn, optimizer=self.optimizer, criterion=self.criterion, device=device)

        # 创建保存目录
        result_file = os.path.join("results", f"{self.model_type}_{self.model_size}.csv")
        os.makedirs("results", exist_ok=True)
        self.result_file = result_file
        # 保存 CSV 文件头
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write("epoch,train_loss,train_acc,valid_loss,valid_acc\n")

    def build_model(self,model_type,input_size,model_size,device):
        # 定义 RNN 层：支持 LSTM 或 GRU（模拟 LTC）
        if model_type == "lstm":
            # rnn = nn.LSTM(input_size, model_size, batch_first=False)
            rnncell = nn.LSTMCell(input_size=input_size, hidden_size=model_size,device=device)

            rnn = DynamicRNN(cell=rnncell,model_type='lstm')

        elif model_type == "ltc":
            # rnncell = LiquidCell(
            #         num_units=model_size, input_size=input_size,device=device
            #     )
            rnncell = LiquidNet2(input_size = input_size,num_units=model_size,device=device)
            # rnncell = LTCCell2(input_size=input_size,hidden_size=model_size,device=device)
            # rnncell = nn.LSTMCell(input_size=input_size, hidden_size=model_size,device=device)
            # rnncell = nn.RNNCell(input_size=input_size, hidden_size=model_size,device=device)

            rnn = DynamicRNN(cell=rnncell,model_type='ltc')
            self.constrain_op = rnncell.get_param_constrain_op
        elif model_type == "rnn":
            rnn = nn.RNN(input_size=input_size, hidden_size=model_size,device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return rnn

    def apply_param_constraints(self):
        if hasattr(self.rnn.cell, "get_param_constrain_op"):
            self.rnn.cell.get_param_constrain_op()

    def build_state(self, batch_size):
        if self.model_type == "lstm":
            zeros = torch.zeros(1, batch_size, self.model_size, device=self.device).detach()
            state = (zeros, zeros)
        else:
            state = torch.zeros(batch_size, self.model_size, device=self.device).detach()
        return state
    # 训练函数：增加早停、日志保存和参数约束
    def fit(self, data_loader, epochs, log_period=10,batch_size=16):

        best_accuracy = 0
        best_valid_stats = (0, 0, 0, 0, 0)  # 不再记录test_loss和test_acc

        # 初始化状态
        # state = self.build_state(data_loader.batch_size)

        # 开始训练
        for epoch in range(epochs):
            self.rnn.train()
            total_loss, total_acc = 0.0, 0.0
            num_batches = 0
            losses, accs = [], []

            for batch_x, batch_y in data_loader.iterate_train(batch_size):
                loss, acc, _ = self.session.train_batch(batch_x, batch_y)
                if self.constrain_op is not None:
                    self.constrain_op()

                losses.append(loss)
                accs.append(acc)
                total_loss += loss
                total_acc += acc
                num_batches += 1

            avg_train_loss = total_loss / num_batches
            avg_train_acc = total_acc / num_batches

            if epoch % log_period == 0:
                valid_acc, valid_loss = self.session.evaluate(data_loader=data_loader, split='valid')
                print(f"Epoch {epoch:03d}, Train Loss: {avg_train_loss:.2f}, "
                      f"Train Acc: {avg_train_acc:.2f}%, Valid Loss: {valid_loss:.2f}, "
                      f"Valid Acc: {valid_acc :.2f}%")

                # 如果验证集准确率提升，则保存模型
                if valid_acc > best_accuracy:
                    best_accuracy = valid_acc
                    best_valid_stats = (epoch, avg_train_loss, avg_train_acc, valid_loss, valid_acc)
                    self.session.save_model()

                # 记录结果到 CSV 文件
                with open(self.result_file, "a") as f:
                    f.write(f"{epoch},{total_loss:.4f},{total_acc / len(data_loader.train_x):.4f},"
                            f"{valid_loss:.4f},{valid_acc:.4f}\n")
            # 终止训练条件
            if epoch > 0 and not np.isfinite(avg_train_loss):
                print("Training stopped due to non-finite loss.")
                break
            # # 在每个 epoch 结束时记录当前学习率
            # for param_group in optimizer.param_groups:
            #     print(f"Epoch {epoch}: Learning rate = {param_group['lr']}")
        # 打印最佳模型信息
        best_epoch, train_loss, train_acc, valid_loss, valid_acc = best_valid_stats
        print(f"Training completed. Best epoch {best_epoch:03d}, Train Loss: {train_loss:.2f}, "
              f"Train Acc: {train_acc:.2f}%, Valid Loss: {valid_loss:.2f}, "
              f"Valid Acc: {valid_acc:.2f}%")

        self.session.load_model()  # 加载最佳模型
        test_acc, test_loss = self.session.evaluate(data_loader=data_loader, split='test')
        print(f"test Loss: {test_loss:.2f}, test Acc: {test_acc :.2f}%")



def get_data_loader(data_type, seq_len,batch_size,data_path, device):
    if data_type == "ozone":
        return OzoneDataLoader(seq_len=seq_len,path=data_path, device=device,batch_size=batch_size)
    elif data_type == "gesture":
        ges_dataloader = GestureData(seq_len=seq_len, path=data_path,device=device)
        return ges_dataloader
        # return GestureDataLoader(seq_len=seq_len,path=data_path, device=device,batch_size=batch_size)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def summarize_gesture_dataloader(dataloader):
    def summary_stats(tensor):
        return {
            "min": float(torch.min(tensor)),
            "max": float(torch.max(tensor)),
            "mean": float(torch.mean(tensor))
        }

    train_mean_stats = summary_stats(torch.mean(dataloader.train_x, dim=(0, 1)))
    train_std_stats = summary_stats(torch.std(dataloader.train_x, dim=(0, 1)))
    valid_mean_stats = summary_stats(torch.mean(dataloader.valid_x, dim=(0, 1)))
    valid_std_stats = summary_stats(torch.std(dataloader.valid_x, dim=(0, 1)))
    test_mean_stats = summary_stats(torch.mean(dataloader.test_x, dim=(0, 1)))
    test_std_stats = summary_stats(torch.std(dataloader.test_x, dim=(0, 1)))

    summary = {
        "total_sequences": {
            "train": dataloader.train_x.size(1),
            "valid": dataloader.valid_x.size(1),
            "test": dataloader.test_x.size(1),
        },
        "feature_statistics": {
            "train": {"mean_stats": train_mean_stats, "std_stats": train_std_stats},
            "valid": {"mean_stats": valid_mean_stats, "std_stats": valid_std_stats},
            "test": {"mean_stats": test_mean_stats, "std_stats": test_std_stats},
        },
        "label_distribution": {
            "train": torch.bincount(dataloader.train_y.flatten()).tolist(),
            "valid": torch.bincount(dataloader.valid_y.flatten()).tolist(),
            "test": torch.bincount(dataloader.test_y.flatten()).tolist(),
        },
    }
    return summary
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['ozone', 'gesture'], required=True, help='选择数据类型',default='ozone')
    parser.add_argument('--data_path', required=True, help='数据路径',default='/mnt/f/dataset/liquid_data/ozone/eighthr.data')
    parser.add_argument('--model_type', choices=['lstm', 'ltc','rnn'], default='lstm', help='模型类型')
    parser.add_argument('--input_size', type=int, default=32, help='输入特征维度')
    parser.add_argument('--model_size', type=int, default=64, help='RNN 隐藏层大小')
    parser.add_argument('--output_size', type=int, default=2, help='输出类别数量')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--seq_len', type=int, default=32, help='序列长度')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    # 在脚本开头设置随机数种子
    set_seed(seed=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取数据加载器
    data_loader = get_data_loader(data_type=args.data_type,seq_len=args.seq_len,
                                  data_path= args.data_path, device=device,batch_size=args.batch_size)

    # aa = summarize_gesture_dataloader(data_loader)
    # print(aa)
    # 初始化模型
    model = BaseModel(input_size=args.input_size, model_type=args.model_type,
                      model_size=args.model_size, output_size=args.output_size,
                      device=device,seq_len=args.seq_len,lr=args.lr)

    # 训练模型
    state = model.fit(data_loader=data_loader, epochs=args.epochs,log_period=10)


    path = '/mnt/data/gesture'
