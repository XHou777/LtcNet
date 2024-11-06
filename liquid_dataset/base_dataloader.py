import numpy as np
import torch

class BaseDataLoader:
    def __init__(self, seq_len=32, valid_ratio=0.1, test_ratio=0.15, batch_size=16, device='cpu'):
        self.seq_len = seq_len
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.device = torch.device(device)

    def normalize(self, data):
        mean = torch.mean(data, dim=0, keepdim=True)
        std = torch.std(data, dim=0, keepdim=True)
        return (data - mean) / std

    def split_data(self, x, y):
        total_seqs = x.size(1)
        print(f"Total number of sequences: {total_seqs}")
        # permutation = np.random.RandomState(23489).permutation(total_seqs)

        permutation = torch.randperm(total_seqs)
        valid_size = int(self.valid_ratio * total_seqs)
        test_size = int(self.test_ratio * total_seqs)

        valid_x = x[:, permutation[:valid_size]].to(self.device)
        valid_y = y[:, permutation[:valid_size]].to(self.device)
        test_x = x[:, permutation[valid_size:valid_size + test_size]].to(self.device)
        test_y = y[:, permutation[valid_size:valid_size + test_size]].to(self.device)
        train_x = x[:, permutation[valid_size + test_size:]].to(self.device)
        train_y = y[:, permutation[valid_size + test_size:]].to(self.device)

        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def iterate_batches(self, x, y):
        total_seqs = x.size(1)
        permutation = torch.randperm(total_seqs)
        total_batches = total_seqs // self.batch_size

        for i in range(total_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_x = x[:, permutation[start:end]].to(self.device)
            batch_y = y[:, permutation[start:end]].to(self.device)
            yield batch_x, batch_y

