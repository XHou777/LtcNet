import os

import numpy as np
import pandas as pd
import torch


def load_trace(filename):
    df = pd.read_csv(filename, header=0)

    str_y = df["Phase"].values
    convert = {"D": 0, "P": 1, "S": 2, "H": 3, "R": 4}
    y = np.empty(str_y.shape[0], dtype=np.int32)
    for i in range(str_y.shape[0]):
        y[i] = convert[str_y[i]]

    x = df.values[:, :-1].astype(np.float32)

    return (x, y)


def cut_in_sequences(tup, seq_len, interleaved=False):
    x, y = tup

    num_sequences = x.shape[0] // seq_len
    sequences = []

    for s in range(num_sequences):
        start = seq_len * s
        end = start + seq_len
        sequences.append((x[start:end], y[start:end]))

        if (interleaved and s < num_sequences - 1):
            start += seq_len // 2
            end = start + seq_len
            sequences.append((x[start:end], y[start:end]))

    return sequences


training_files = [
    "a3_va3.csv",
    "b1_va3.csv",
    "b3_va3.csv",
    "c1_va3.csv",
    "c3_va3.csv",
    "a2_va3.csv",
    "a1_va3.csv",
]


class GestureData:

    def __init__(self, seq_len=32,batch_size=16,path="data/gesture",device="cpu"):
        train_traces = []
        self.batch_size = batch_size
        interleaved_train = True
        for f in training_files:
            train_traces.extend(
                cut_in_sequences(load_trace(os.path.join(path, f)), seq_len, interleaved=interleaved_train))

        train_x, train_y = list(zip(*train_traces))

        self.train_x = np.stack(train_x, axis=1)
        self.train_y = np.stack(train_y, axis=1)

        flat_x = self.train_x.reshape([-1, self.train_x.shape[-1]])
        mean_x = np.mean(flat_x, axis=0)
        std_x = np.std(flat_x, axis=0)
        self.train_x = (self.train_x - mean_x) / std_x

        # 转换为 PyTorch Tensor
        self.train_x = torch.tensor(self.train_x, dtype=torch.float32,device=device)
        self.train_y = torch.tensor(self.train_y, dtype=torch.long,device=device)

        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size:valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size:valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size:]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size:]]


    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]
            yield (batch_x, batch_y)

    def iterate_batches(self, x, y):
        total_seqs = x.shape[1]
        permutation = torch.randperm(total_seqs)
        total_batches = total_seqs // self.batch_size

        for i in range(total_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_x = x[:, permutation[start:end]]
            batch_y = y[:, permutation[start:end]]
            yield batch_x, batch_y

    def __len__(self):
        """返回训练数据集的批次数"""
        return self.train_x.shape[1]// self.batch_size
