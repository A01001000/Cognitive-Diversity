import torch
import torch.nn as nn

class EntangledNet(nn.Module):
    def __init__(self, in_dim=6, hidden=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)  # predicts action directly

    def forward(self, x):
        h = self.body(x)
        action_logit = self.head(h)
        return action_logit, h  # return hidden state too, for the post-hoc probe later