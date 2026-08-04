import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Styling Configuration (Academic / Paper Style) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

COLOR_M1 = '#D95319'     # Pattern Model (Orange)
COLOR_M2 = '#0072BD'     # Causal Model (Blue)
COLOR_JOINT = '#7E2F8E'  # Joint Jury (Purple)

def load_training_data(filepath="training_curves.json"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Run the PyTorch training script first!")
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_accuracy_curves(data):
    epochs = [d['epoch'] for d in data]
    acc1 = [d['acc1'] for d in data]
    acc2 = [d['acc2'] for d in data]
    joint_acc = [d['joint_acc'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot individual model accuracies (dashed lines)
    ax.plot(epochs, acc1, linestyle='--', color=COLOR_M1, marker='o', 
            linewidth=2, markersize=8, label='Model 1 (Pattern Focus)')
    ax.plot(epochs, acc2, linestyle='--', color=COLOR_M2, marker='s', 
            linewidth=2, markersize=8, label='Model 2 (Causal Focus)')
    
    # Plot joint jury accuracy (solid thick line)
    ax.plot(epochs, joint_acc, linestyle='-', color=COLOR_JOINT, marker='^', 
            linewidth=3, markersize=10, label='Joint Jury (Hybridization)')

    # Fill the area between the best individual model and the joint accuracy (Hybridization Gain)
    best_individual = np.maximum(acc1, acc2)
    ax.fill_between(epochs, best_individual, joint_acc, color=COLOR_JOINT, alpha=0.15, 
                    label='Complementarity Gain')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Hybridization Gain via Joint Orthogonal Training', pad=20)
    
    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(min(min(acc1), min(acc2)) - 0.05, 1.05)
    
    # Force integer ticks on x-axis
    ax.set_xticks(epochs)

    ax.legend(loc='lower right')
    plt.tight_layout()
    
    out_file = "training_accuracy_curve.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Accuracy plot saved to {out_file}")

def plot_orthogonality_curve(data):
    epochs = [d['epoch'] for d in data]
    rho = [d['rho'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot rho over time
    ax.plot(epochs, rho, linestyle='-', color='#2CA02C', marker='D', 
            linewidth=3, markersize=9, label=r'Error Correlation ($\rho_E$)')

    # Draw the zero-correlation threshold
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    # Shade the target orthogonal zone (negative correlation)
    ax.axhspan(-1.0, 0.0, facecolor='green', alpha=0.1, label='Target Orthogonality ($\rho_E \le 0$)')
    ax.axhspan(0.0, 1.0, facecolor='red', alpha=0.05, label='Correlated Failure (Vulnerable)')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel(r'Pearson Correlation of Errors ($\rho_E$)')
    ax.set_title('Enforcing Cognitive Diversity Over Time', pad=20)
    
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(epochs)

    ax.legend(loc='upper right')
    plt.tight_layout()
    
    out_file = "training_orthogonality_curve.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Orthogonality plot saved to {out_file}")

if __name__ == "__main__":
    try:
        metrics_data = load_training_data()
        print("[*] Generating training loss curve visualizations...")
        plot_accuracy_curves(metrics_data)
        plot_orthogonality_curve(metrics_data)
        print("[🚀] All training plots successfully generated!")
    except Exception as e:
        print(f"[-] Error: {e}")