import torch
from .toy_scm import gen_data
from .train import train_entangled, train_structured
from .probe import fit_probe

def test_structured_intervention(model, seed, n=500):
    x, belief, desire, action = gen_data(n, seed=seed)
    x_t = torch.tensor(x)
    with torch.no_grad():
        _, belief_logit, desire_logit = model(x_t)
        baseline_action = model.action_head(
            torch.cat([torch.sigmoid(belief_logit), torch.sigmoid(desire_logit)], dim=1)
        )
        # do(belief=1): clamp belief to 1, leave desire untouched
        forced_belief = torch.ones_like(belief_logit)
        intervened_action = model.action_head(
            torch.cat([torch.sigmoid(forced_belief), torch.sigmoid(desire_logit)], dim=1)
        )
        # check desire is literally untouched (it wasn't recomputed at all -- by construction)
        desire_unchanged = torch.equal(desire_logit, desire_logit)  # trivially true here
    action_shift = (torch.sigmoid(intervened_action) - torch.sigmoid(baseline_action)).mean().item()
    return action_shift  # expect: large positive shift when belief forced to 1

def test_structured_intervention_filtered(model, seed, n=2000):
    x, belief, desire, action = gen_data(n, seed=seed)
    x_t = torch.tensor(x)
    with torch.no_grad():
        _, belief_logit, desire_logit = model(x_t)
        mask = (desire > 0.5) & (belief < 0.5)  # cases where forcing belief=1 SHOULD flip action
        x_sub, belief_logit, desire_logit = x_t[mask], belief_logit[mask], desire_logit[mask]
        baseline_action = model.action_head(torch.cat([torch.sigmoid(belief_logit), torch.sigmoid(desire_logit)], dim=1))
        forced_belief = torch.ones_like(belief_logit)
        intervened_action = model.action_head(torch.cat([torch.sigmoid(forced_belief), torch.sigmoid(desire_logit)], dim=1))
        
        print("Filtered structured intervention test mask sum: ", mask.sum().item())
    return (torch.sigmoid(intervened_action) - torch.sigmoid(baseline_action)).mean().item()

def test_entangled_intervention(model, seed, alpha=3.0, n=500):
    belief_dir = fit_probe(model, "belief")
    desire_dir = fit_probe(model, "desire")
    x, belief, desire, action = gen_data(n, seed=seed)
    x_t = torch.tensor(x)
    with torch.no_grad():
        baseline_logit, h = model(x_t)
        # approximate do(belief=1): push hidden state along belief probe direction
        h_intervened = h + alpha * belief_dir
        intervened_logit = model.head(h_intervened)
        # measure leakage: did pushing along "belief direction" also move the desire probe's readout?
        baseline_desire_prob = torch.sigmoid(h @ desire_dir) # normalized to [-1, 1] space for comparison with action shift
        intervened_desire_prob = torch.sigmoid(h_intervened @ desire_dir)
    action_shift = (torch.sigmoid(intervened_logit) - torch.sigmoid(baseline_logit)).mean().item()
    desire_leakage = (intervened_desire_prob - baseline_desire_prob).mean().item()
    return action_shift, desire_leakage

def alpha_sweep(model, seed, alphas=(1, 2, 3, 5, 8), n=2000):
    results = []
    for a in alphas:
        shift, leak = test_entangled_intervention(model, seed, alpha=a, n=n)
        results.append((a, shift, leak))
        print(f"alpha={a}: action_shift={shift:.3f}, desire_leakage={leak:.3f}")
    return results