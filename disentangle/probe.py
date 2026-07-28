import torch
import torch.nn as nn
from toy_scm import gen_data

def fit_probe(model, target_name, n=2000):
    x, belief, desire, action = gen_data(n, seed=2)
    x_t = torch.tensor(x)
    target = torch.tensor(belief if target_name == "belief" else desire).unsqueeze(1)
    with torch.no_grad():
        _, h = model(x_t)
    probe = nn.Linear(h.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad()
        pred = probe(h.detach())
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
    # the probe's weight vector is "the belief direction" in hidden space
    return probe.weight.detach()[0]  # shape (hidden_dim,)