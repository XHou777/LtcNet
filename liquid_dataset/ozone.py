import torch
import numpy as np
from torch import nn


def to_float(v):
    if (v == "?"):
        return 0
    else:
        return float(v)


def load_trace(path):
    all_x = []
    all_y = []

    with open(path) as f:
        miss = 0
        total = 0
        while True:
            line = f.readline()
            if (line is None):
                break
            line = line[:-1]
            parts = line.split(',')

            total += 1
            for i in range(1, len(parts) - 1):
                if (parts[i] == "?"):
                    miss += 1
                    break

            if (len(parts) != 74):
                break
            label = int(float(parts[-1]))
            feats = [to_float(parts[i]) for i in range(1, len(parts) - 1)]

            all_x.append(np.array(feats))
            all_y.append(label)
    print("Missing features in {} out of {} samples ({:0.2f})".format(miss, total, 100 * miss / total))
    print("Read {} lines".format(len(all_x)))
    all_x = np.stack(all_x, axis=0)
    all_y = np.array(all_y)

    print("Imbalance: {:0.2f}%".format(100 * np.mean(all_y)))
    all_x -= np.mean(all_x)  # normalize
    all_x /= np.std(all_x)  # normalize

    return all_x, all_y


def cut_in_sequences(x, y, seq_len, inc=1):
    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


class OzoneData:

    def __init__(self,path, seq_len=32):

        x, y = load_trace(path)

        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=4)

        self.train_x = np.stack(train_x, axis=1)
        self.train_y = np.stack(train_y, axis=1)

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

