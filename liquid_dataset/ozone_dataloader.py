import torch

from liquid_dataset.base_dataloader import BaseDataLoader


class OzoneDataLoader(BaseDataLoader):
    def __init__(self, path, seq_len=32, batch_size=16, device='cpu'):
        super().__init__(seq_len, batch_size=batch_size, device=device)
        x, y = self.load_trace(path)
        train_x, train_y = self.cut_in_sequences(x, y)

        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = self.split_data(train_x, train_y)

    def load_trace(self, path):
        all_x, all_y = [], []
        with open(path) as f:
            miss, total = 0, 0
            for line in f:
                if not line.strip():
                    break
                parts = line.strip().split(',')
                total += 1
                if len(parts) != 74 or "?" in parts[1:-1]:
                    miss += 1
                    continue
                label = int(float(parts[-1]))
                feats = [float(p) if p != "?" else 0 for p in parts[1:-1]]
                all_x.append(torch.tensor(feats, dtype=torch.float32))
                all_y.append(label)
        print(f"Missing features in {miss} out of {total} samples ({100 * miss / total:.2f}%)")
        print(f"Read {len(all_x)} lines")
        x = self.normalize(torch.stack(all_x, dim=0))
        y = torch.tensor(all_y, dtype=torch.int64)
        print(f"Imbalance: {100 * y.float().mean():.2f}%")
        return x, y

    def cut_in_sequences(self, x, y):
        sequences_x, sequences_y = [], []
        for s in range(0, x.size(0) - self.seq_len, 4):
            start, end = s, s + self.seq_len
            sequences_x.append(x[start:end])
            sequences_y.append(y[start:end])
        return torch.stack(sequences_x, dim=1), torch.stack(sequences_y, dim=1)

    def __len__(self):
        """返回训练数据集的批次数"""
        return self.train_x.size(1) // self.batch_size
