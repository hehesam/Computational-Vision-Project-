import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Custom Dataset
class PoseDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)  # (N,3,T,25,1)
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)[1][0]
        self.labels = np.array(labels)
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]                         # (3,T,25,1)
        x = np.squeeze(x, axis=-1)                 # (3,T,25)
        y = self.labels[idx]
        return torch.from_numpy(x).float(), y

# LSTM Model
class PoseLSTM(nn.Module):
    def __init__(self, in_dim=75, hidden_dim=128, num_layers=2, num_classes=60, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (N,3,T,25)
        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()      # (N,T,V,C)
        x = x.view(N, T, V * C)                    # (N,T,75)
        out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, N, hidden_dim)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)    # (N, hidden_dim*2)
        return self.fc(h)

# Utility: compute accuracy
def compute_accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()
