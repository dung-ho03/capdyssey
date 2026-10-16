"""Architecture copied from team CNN notebook; no training on import."""
import torch
from torch import nn
class Block(nn.Module):
    def __init__(self, cin, cout, k=5, p=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k//2), nn.BatchNorm1d(cout), nn.ReLU(),
                                 nn.Conv1d(cout, cout, k, padding=k//2), nn.BatchNorm1d(cout), nn.ReLU(),
                                 nn.MaxPool1d(2), nn.Dropout(p))
    def forward(self, x): return self.net(x)

class PPGQualityCNN(nn.Module):                # 입력 (B,1,300) → '좋은 품질' 로짓
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(Block(1, 16), Block(16, 32), Block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, x): return self.head(self.features(x)).squeeze(-1)
