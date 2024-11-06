import os
import numpy as np
import pandas as pd
import torch
from liquid_dataset.base_dataloader import BaseDataLoader
import math

def load_trace(filename):
    df = pd.read_csv(filename, header=0)

    str_y = df["Phase"].values
    convert = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}
    y = np.array([convert[val] for val in str_y], dtype=np.int32)

    x = df.values[:, :-1].astype(np.float32)

    return x, y

def cut_in_sequences(tup, seq_len, interleaved=False):
    x, y = tup

    num_sequences = x.shape[0] // seq_len
    sequences = []

    for s in range(num_sequences):
        start = seq_len * s
        end = start + seq_len
        sequences.append((x[start:end], y[start:end]))

        if interleaved and s < num_sequences - 1:
            start += seq_len // 2
            end = start + seq_len
            sequences.append((x[start:end], y[start:end]))

    return sequences

class GestureDataLoader(BaseDataLoader):
    def __init__(self, seq_len=32, path='', batch_size=16, device='cpu'):
        super().__init__(seq_len=seq_len, batch_size=batch_size, device=device)
        self.training_files = [
            "a3_va3.csv", "b1_va3.csv", "b3_va3.csv",
            "c1_va3.csv", "c3_va3.csv", "a2_va3.csv", "a1_va3.csv"
        ]
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = self.load_data(path)

    def load_data(self, path):
        train_traces = []

        interleaved_train = True
        for f in self.training_files:
            train_traces.extend(
                cut_in_sequences(load_trace(os.path.join(path, f)), self.seq_len, interleaved=interleaved_train)
            )

        train_x, train_y = list(zip(*train_traces))

        train_x = np.stack(train_x, axis=1)
        train_y = np.stack(train_y, axis=1)

        # 标准化
        flat_x = train_x.reshape([-1, train_x.shape[-1]])
        mean_x = np.mean(flat_x, axis=0)
        std_x = np.std(flat_x, axis=0)
        train_x = (train_x - mean_x) / std_x

        # 转换为 Tensor
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.long)

        # 拆分数据集
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.split_data(train_x, train_y)
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def __len__(self):
        """返回训练数据集的批次数，包括不足一个 batch 的数据。"""
        return math.ceil(self.train_x.size(1) / self.batch_size)