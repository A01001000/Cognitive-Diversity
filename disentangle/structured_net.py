import torch
import torch.nn as nn

class StructuredNet(nn.Module):
    def __init__(self, in_dim=6, hidden=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
        )
        self.belief_head = nn.Linear(hidden, 1)  # supervised toward true belief
        self.desire_head = nn.Linear(hidden, 1)  # supervised toward true desire
        self.action_head = nn.Linear(2, 1)       # combines ONLY belief and desire outputs

    def forward(self, x):
        h = self.body(x)
        belief_logit = self.belief_head(h)
        desire_logit = self.desire_head(h)
        combined = torch.cat([torch.sigmoid(belief_logit), torch.sigmoid(desire_logit)], dim=1)
        action_logit = self.action_head(combined)
        return action_logit, belief_logit, desire_logit