import torch
import torch.nn as nn
from .toy_scm import gen_data
from .entangled_net import EntangledNet
from .structured_net import StructuredNet

def to_tensors(x, belief, desire, action):
    return (torch.tensor(x), torch.tensor(belief).unsqueeze(1),
            torch.tensor(desire).unsqueeze(1), torch.tensor(action).unsqueeze(1))

def train_entangled(seed, epochs=300, n=4000):
    x, belief, desire, action = to_tensors(*gen_data(n, seed=seed))
    model = EntangledNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for e in range(epochs):
        opt.zero_grad()
        action_logit, _ = model(x)
        loss = loss_fn(action_logit, action)
        loss.backward()
        opt.step()
    return model

def train_structured(seed, epochs=300, n=4000):
    x, belief, desire, action = to_tensors(*gen_data(n, seed=seed))
    model = StructuredNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for e in range(epochs):
        opt.zero_grad()
        action_logit, belief_logit, desire_logit = model(x)
        # multi-task loss: predict action correctly AND match true belief/desire labels
        loss = (loss_fn(action_logit, action)
                + loss_fn(belief_logit, belief)
                + loss_fn(desire_logit, desire))
        loss.backward()
        opt.step()
    return model