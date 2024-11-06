import torch
from torch import nn


class DynamicRNN(nn.Module):
    def __init__(self, cell, time_major=True, model_type='lstm',output_size = 5):
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
        self.fc = nn.Linear(self.hidden_size, output_size)

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

        outputs = self.fc(outputs)  # 通过全连接层进行分类

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
