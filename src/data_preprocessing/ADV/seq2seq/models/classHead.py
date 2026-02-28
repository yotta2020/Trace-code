import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPForSequenceClassification(nn.Module):
    def __init__(self, n_classes, hidden_size=1024):
        super(MLPForSequenceClassification, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
