import numpy as np

# synthetic ground truth: Belief, Desire independently generated, Action = f(Belief, Desire)
def gen_data(n):
    belief = np.random.randint(0, 2, n)
    desire = np.random.randint(0, 2, n)
    action = (belief & desire).astype(float)  # simple deterministic causal rule
    x = np.stack([belief, desire], axis=1) + np.random.randn(n, 2) * 0.1  # noisy observation
    return x, belief, desire, action