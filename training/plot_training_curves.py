import os
import json
import matplotlib.pyplot as plt
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

def load_training_data(filepath="training/training_results/training_curves.json"):
    if not os.path.exists(filepath):
        print(f"[-] Could not find {filepath}. Skipping training curves.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_accuracy_curves(data):
    if not data: return
    epochs = [d['epoch'] for d in data]
    acc1 = [d['acc1'] for d in data]
    acc2 = [d['acc2'] for d in data]
    joint_acc = [d['joint_acc'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot individual model accuracies (continuous dashed lines, no markers)
    ax.plot(epochs, acc1, linestyle='--', color=COLOR_M1, 
            linewidth=2, label='Model 1 (Pattern Focus)')
    ax.plot(epochs, acc2, linestyle='--', color=COLOR_M2, 
            linewidth=2, label='Model 2 (Causal Focus)')
    
    # Plot joint jury accuracy (continuous solid thick line, no markers)
    ax.plot(epochs, joint_acc, linestyle='-', color=COLOR_JOINT, 
            linewidth=3, label='Joint Jury (Hybridization)')

    # Fill the area between the best individual model and the joint accuracy
    best_individual = np.maximum(acc1, acc2)
    ax.fill_between(epochs, best_individual, joint_acc, color=COLOR_JOINT, alpha=0.15, 
                    label='Complementarity Gain')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Hybridization Gain via Joint Orthogonal Training', pad=20)
    
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    # Locked maximum y-axis limit to 1.0 (100%)
    ax.set_ylim(min(min(acc1), min(acc2)) - 0.05, 1.0)
    ax.set_xticks(epochs)

    # Moved legend to top right
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    out_file = "training/training_results/plots/training_accuracy_curve.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Accuracy plot saved to {out_file}")

def plot_orthogonality_curve(data):
    if not data: return
    epochs = [d['epoch'] for d in data]
    rho = [d['rho'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot rho over time (continuous line, no markers)
    ax.plot(epochs, rho, linestyle='-', color='#2CA02C', 
            linewidth=3, label=r'Error Correlation ($\rho_E$)')

    # Draw the zero-correlation threshold
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    # Shade the target orthogonal zone (Switched \le to \leq to fix Matplotlib parser error)
    ax.axhspan(-1.0, 0.0, facecolor='green', alpha=0.1, label=r'Target Orthogonality ($\rho_E \leq 0$)')
    ax.axhspan(0.0, 1.0, facecolor='red', alpha=0.05, label='Correlated Failure (Vulnerable)')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel(r'Pearson Correlation of Errors ($\rho_E$)')
    ax.set_title('Enforcing Cognitive Diversity Over Time', pad=20)
    
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(epochs)

    ax.legend(loc='upper right')
    plt.tight_layout()
    
    out_file = "training/training_results/plots/training_orthogonality_curve.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Orthogonality plot saved to {out_file}")

def plot_test_results_comparison(joint_loss_path="training/training_results/ood_test_results.json", final_ood_path="training/training_results/final_ood_results.json"):
    if not os.path.exists(joint_loss_path) or not os.path.exists(final_ood_path):
        print(f"[-] Missing one or both test result JSONs ({joint_loss_path}, {final_ood_path}). Skipping comparison plot.")
        return

    with open(joint_loss_path, 'r') as f:
        joint_data = json.load(f)
    with open(final_ood_path, 'r') as f:
        final_data = json.load(f)

    # Extract Metrics
    acc_labels = ['Model 1 Acc', 'Model 2 Acc', 'Joint Jury Acc']
    joint_accs = [joint_data['ood_acc_m1'], joint_data['ood_acc_m2'], joint_data['ood_joint_acc']]
    final_accs = [final_data['ood_acc_m1'], final_data['ood_acc_m2'], final_data['ood_joint_acc']]
    
    joint_rho = joint_data['ood_error_correlation_rho']
    final_rho = final_data['ood_error_correlation_rho']

    # Create figure with 2 subplots (Accuracies on left, Rho on right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

    x = np.arange(len(acc_labels))
    width = 0.35

    # Colors for comparison
    COLOR_JOINT_LOSS = '#E24A33' # Red/Orange tint
    COLOR_FINAL = '#348ABD'      # Blue tint

    # Left Plot: Accuracy Bars
    bars1 = ax1.bar(x - width/2, joint_accs, width, label='Unstable Joint Loss Training', color=COLOR_JOINT_LOSS)
    bars2 = ax1.bar(x + width/2, final_accs, width, label='Asymmetric Narrow Fine-Tuning', color=COLOR_FINAL)

    ax1.set_ylabel('Accuracy on Unseen OOD Set')
    ax1.set_title('Test Set Accuracy Comparison', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(acc_labels)
    
    from matplotlib.ticker import PercentFormatter
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper left')

    # Data labels for Accuracies
    for i, v in enumerate(joint_accs):
        ax1.text(i - width/2, v + 0.02, f"{v:.1%}", ha='center', va='bottom', fontweight='bold', fontsize=10)
    for i, v in enumerate(final_accs):
        ax1.text(i + width/2, v + 0.02, f"{v:.1%}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Right Plot: Rho Bars
    methods = ['Unstable\nJoint Loss', 'Narrow\nFine-Tuning']
    rhos = [joint_rho, final_rho]

    bars3 = ax2.bar(methods, rhos, color=[COLOR_JOINT_LOSS, COLOR_FINAL], width=0.5)
    ax2.set_ylabel(r'Error Correlation ($\rho_E$)')
    ax2.set_title('Cognitive Diversity (Lower = Better)', pad=15)
    
    # Adjust Y-axis depending on how low rho goes
    y_min = min(0, min(rhos))
    ax2.set_ylim(y_min, 1.1)
    ax2.axhline(0, color='black', linewidth=1.2)
    
    # Data labels for Rho
    for bar in bars3:
        yval = bar.get_height()
        offset = 0.05 if yval >= 0 else -0.05
        va = 'bottom' if yval >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.3f}", ha='center', va=va, fontweight='bold', fontsize=11)

    plt.tight_layout()
    out_file = "training/training_results/plots/test_results_comparison.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Comparison plot saved to {out_file}")

if __name__ == "__main__":
    try:
        # 1. Generate Training Curves (if JSON exists)
        metrics_data = load_training_data()
        if metrics_data:
            print("[*] Generating training loss curve visualizations...")
            plot_accuracy_curves(metrics_data)
            plot_orthogonality_curve(metrics_data)
        
        # 2. Generate Final Results Comparison
        print("[*] Generating Final OOD Test Results comparison...")
        plot_test_results_comparison()
        
        print("All available plots successfully generated!")
    except Exception as e:
        print(f"[-] Error: {e}")