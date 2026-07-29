import numpy as np
import json, os
from datetime import datetime
import matplotlib.pyplot as plt

def label_balance(scenarios):
    n_true = sum(1 for s in scenarios if s.label)
    n_false = len(scenarios) - n_true
    print(f"True: {n_true} ({n_true/len(scenarios):.1%}), False: {n_false} ({n_false/len(scenarios):.1%})")
    
def repeat(fn, n_seeds=10):
    results = []
    for seed in range(n_seeds):
        results.append(fn(seed=seed))
    return results

def save_results(results_dict, name):
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved to {path}")

def plot_alpha_sweep(sweep_results, save_path="results/alpha_sweep.png"):
    alphas = [r[0] for r in sweep_results]
    shifts = [r[1] for r in sweep_results]
    leaks = [r[2] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(alphas, shifts, marker="o", color="tab:blue", label="action shift (intended)")
    ax1.set_xlabel("intervention strength (alpha)")
    ax1.set_ylabel("action shift", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(alphas, leaks, marker="s", color="tab:red", label="desire leakage (unintended)")
    ax2.set_ylabel("desire leakage", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Entangled net: intended effect vs. unintended leakage")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")