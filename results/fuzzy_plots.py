import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Styling Configuration (Academic / Paper Style) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2
})

COLOR_JURY_A = '#D95319' # Highlighting failure (Red-Orange)
COLOR_JURY_B = '#0072BD' # Highlighting safety (Blue)
COLOR_JURY_C = '#7E2F8E' # Super Jury (Purple)

def load_results_data():
    """Loads exported metric data from results/fuzzy_jury_evaluation_results.json."""
    json_path = "results/fuzzy_jury_evaluation_results.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}. Run aggregate_juries.py first!")
        
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def plot_error_correlation(rho_A, rho_B):
    """Generates side-by-side heatmaps for Juror Error Correlation from real rho values."""
    corr_A = np.array([[1.0, rho_A], [rho_A, 1.0]])
    corr_B = np.array([[1.0, rho_B], [rho_B, 1.0]])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'wspace': 0.3})
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    sns.heatmap(corr_A, annot=True, fmt=".3f", cmap=cmap, vmin=-1, vmax=1, 
                ax=axes[0], cbar=False, square=True,
                xticklabels=['GPT-4o-mini', 'Mistral-Nemo'],
                yticklabels=['GPT-4o-mini', 'Mistral-Nemo'],
                annot_kws={"size": 15, "weight": "bold"})
    axes[0].set_title(f'Jury A Error Correlation\n(Model Diversity: ρ = {rho_A:.3f})', pad=15)
    
    sns.heatmap(corr_B, annot=True, fmt=".3f", cmap=cmap, vmin=-1, vmax=1, 
                ax=axes[1], cbar=True, square=True,
                xticklabels=['Gemini Pattern', 'Gemini Causal'],
                yticklabels=['Gemini Pattern', 'Gemini Causal'],
                annot_kws={"size": 15, "weight": "bold"})
    axes[1].set_title(f'Jury B Error Correlation\n(Cognitive Diversity: ρ = {rho_B:.3f})', pad=15)
    
    for ax in axes:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    out_file = "results/fuzzy_plots/error_correlation_heatmap.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Plot saved: {out_file}")

def plot_adversarial_collapse(traps, acr_A, acr_B, acr_C):
    """Generates a dynamic grouped bar chart for Adversarial Collapse Ratio (ACR)."""
    labels = [t.replace("_", " ").title() for t in traps]
    jury_a_vals = [acr_A[t] for t in traps]
    jury_b_vals = [acr_B[t] for t in traps]
    jury_c_vals = [acr_C[t] for t in traps]
    
    x = np.arange(len(labels))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, jury_a_vals, width, label='Jury A (Model Div)', color=COLOR_JURY_A, edgecolor='black')
    rects2 = ax.bar(x, jury_b_vals, width, label='Jury B (Cognitive Div)', color=COLOR_JURY_B, edgecolor='black')
    rects3 = ax.bar(x + width, jury_c_vals, width, label='Jury C (Super Jury)', color=COLOR_JURY_C, edgecolor='black')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 1.0: 
                ax.annotate(f'{height:.1f}x',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, weight='bold')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_ylabel('Adversarial Collapse Ratio (ACR)\n(Multiplier of Baseline Error)')
    ax.set_title('Systemic Failure Amplification Across Attack Vectors', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15 if len(labels) > 4 else 0)
    ax.legend()
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.yscale('log')
    max_val = max(max(jury_a_vals), max(jury_b_vals), max(jury_c_vals), 10.0)
    ax.set_ylim(0.5, max_val * 2) 
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}x'.format(y)))

    plt.tight_layout()
    out_file = "results/fuzzy_plots/adversarial_collapse_bar.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Plot saved: {out_file}")

def plot_maximum_shared_bias(traps, msb_A, msb_B, msb_C):
    """Generates a line plot showing Maximum Shared Bias (Joint False Positive Rate)."""
    labels = [t.replace("_", " ").title() for t in traps]
    
    df = pd.DataFrame({
        'Trap': labels,
        'Jury A (Model Div)': [msb_A[t] for t in traps],
        'Jury B (Cognitive Div)': [msb_B[t] for t in traps],
        'Jury C (Super Jury)': [msb_C[t] for t in traps]
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['Trap'], df['Jury A (Model Div)'], marker='o', markersize=10, 
            linewidth=2, label='Jury A (Model Div)', color=COLOR_JURY_A)
    ax.plot(df['Trap'], df['Jury B (Cognitive Div)'], marker='s', markersize=10, 
            linewidth=2, label='Jury B (Cognitive Div)', color=COLOR_JURY_B)
    ax.plot(df['Trap'], df['Jury C (Super Jury)'], marker='^', markersize=10, 
            linewidth=2, label='Jury C (Super Jury)', color=COLOR_JURY_C, linestyle='-.')

    ax.axhspan(0.5, 1.05, facecolor='red', alpha=0.1, label='Critical Failure Zone (>50% MSB)')
    ax.set_ylabel('Maximum Shared Bias (Joint False Positive Rate)')
    ax.set_title('Worst-Case Joint Failure Probability by Attack Vector', pad=20)
    
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='center left')

    plt.tight_layout()
    out_file = "results/fuzzy_plots/max_shared_bias_plot.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Plot saved: {out_file}")

def generate_all_plots():
    data = load_results_data()
    print("[*] Generating visualizations from results/fuzzy_jury_evaluation_results.json...")
    
    plot_error_correlation(data["rho_A"], data["rho_B"])
    plot_adversarial_collapse(data["traps"], data["acr_A"], data["acr_B"], data["acr_C"])
    plot_maximum_shared_bias(data["traps"], data["msb_A"], data["msb_B"], data["msb_C"])
    print("All plots successfully generated and saved to ./results/")

if __name__ == "__main__":
    generate_all_plots()