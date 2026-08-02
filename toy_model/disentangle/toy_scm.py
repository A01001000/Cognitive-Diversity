import numpy as np

# synthetic ground truth: Belief, Desire independently generated, Action = f(Belief, Desire)
def gen_data(n, seed=0):
    rng = np.random.RandomState(seed)
    belief = rng.randint(0, 2, n).astype(np.float32)
    desire = rng.randint(0, 2, n).astype(np.float32)
    # ground-truth causal rule: action = 1 only if both belief and desire are 1
    action = (belief.astype(int) & desire.astype(int)).astype(np.float32)
    # noisy observation: belief/desire aren't handed to the model directly,
    # they're mixed into a higher-dim noisy "sensory" vector x
    noise = rng.randn(n, 6) * 0.3
    # x encodes belief and desire redundantly across dimensions, plus noise dims
    x = np.stack([
        belief + noise[:, 0],
        belief * 0.8 + noise[:, 1],
        desire + noise[:, 2],
        desire * 0.8 + noise[:, 3],
        noise[:, 4],          # pure noise dim
        noise[:, 5],          # pure noise dim
    ], axis=1).astype(np.float32)
    return x, belief, desire, action