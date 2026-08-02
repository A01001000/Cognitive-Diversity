import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- 1. Styling Configuration (Academic/Paper Style) ---
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

# Custom colors for distinct Juries
COLOR_JURY_A = '#D95319' # Highlighting failure/correlation (Red-Orange)
COLOR_JURY_B = '#0072BD' # Highlighting safety/orthogonality (Blue)

# --- Mock Data (Replace with your actual aggregate_juries.py outputs) ---
# Correlation Matrices (Pearson's rho)
corr_jury_A = np.array([[1.0, 0.78], [0.78, 1.0]])
corr_jury_B = np.array([[1.0, -0.12], [-0.12, 1.0]])

# Adversarial Collapse Ratio (ACR) Data
# Format: { 'Trap': [Baseline, Jury A, Jury B] }
acr_data = {
    'Baseline': [1.0, 1.0],
    'Semantic Trap': [5.0, 1.0],
    'Referential Trap': [42.5, 1.0],
    'Combined Trap': [37.5, 1.0]
}

# Maximum Shared Bias (False Positive Rates) Data
msb_data = {
    'Trap': ['Baseline', 'Semantic Trap', 'Referential Trap', 'Combined Trap'],
    'Jury A (Model Div)': [0.02, 0.10, 0.85, 0.75],
    'Jury B (Cognitive Div)': [0.00, 0.00, 0.00, 0.00]
}


def plot_error_correlation(corr_A, corr_B):
    """Generates side-by-side heatmaps for Juror Error Correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'wspace': 0.3})
    
    # Custom diverging colormap (Blue for negative, White for 0, Red for positive correlation)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Jury A Heatmap
    sns.heatmap(corr_A, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, 
                ax=axes[0], cbar=False, square=True,
                xticklabels=['GPT-4o-mini', 'Claude Haiku'],
                yticklabels=['GPT-4o-mini', 'Claude Haiku'],
                annot_kws={"size": 16, "weight": "bold"})
    axes[0].set_title('Jury A Error Correlation\n(Model Diversity)', pad=15)
    
    # Jury B Heatmap
    sns.heatmap(corr_B, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, 
                ax=axes[1], cbar=True, square=True,
                xticklabels=['Gemini Pattern', 'Gemini Causal'],
                yticklabels=['Gemini Pattern', 'Gemini Causal'],
                annot_kws={"size": 16, "weight": "bold"})
    axes[1].set_title('Jury B Error Correlation\n(Cognitive Diversity)', pad=15)
    
    # Ensure tick labels are not rotated
    for ax in axes:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.savefig('error_correlation_heatmap.png', bbox_inches='tight')
    plt.show()

def plot_adversarial_collapse(acr_dict):
    """Generates a grouped bar chart for Adversarial Collapse Ratio (ACR)."""
    labels = list(acr_dict.keys())
    jury_a_acr = [val[0] for val in acr_dict.values()]
    jury_b_acr = [val[1] for val in acr_dict.values()]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, jury_a_acr, width, label='Jury A (Model Div)', color=COLOR_JURY_A, edgecolor='black')
    rects2 = ax.bar(x + width/2, jury_b_acr, width, label='Jury B (Cognitive Div)', color=COLOR_JURY_B, edgecolor='black')

    # Add text labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 1:
                ax.annotate(f'{height:.1f}x',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, weight='bold')
            else:
                 ax.annotate(f'{height:.1f}x',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    ax.set_ylabel('Adversarial Collapse Ratio (ACR)\n(Multiplier of Baseline Error)')
    ax.set_title('Systemic Failure Amplification under Adversarial Attack', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add a horizontal line at 1.0x (No collapse)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.yscale('log') # Log scale is often better for multipliers
    ax.set_ylim(0.5, 100) # Adjust based on your max ACR
    
    # Custom y-axis formatting for log scale
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}x'.format(y)))

    plt.tight_layout()
    plt.savefig('adversarial_collapse_bar.png', bbox_inches='tight')
    plt.show()

def plot_maximum_shared_bias(msb_dict):
    """Generates a line/scatter plot showing Maximum Shared Bias (False Positive rate)."""
    df = pd.DataFrame(msb_dict)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting lines with markers
    ax.plot(df['Trap'], df['Jury A (Model Div)'], marker='o', markersize=10, 
            linewidth=2, label='Jury A (Model Div)', color=COLOR_JURY_A)
    ax.plot(df['Trap'], df['Jury B (Cognitive Div)'], marker='s', markersize=10, 
            linewidth=2, label='Jury B (Cognitive Div)', color=COLOR_JURY_B)

    # Shaded region indicating danger zone (high false positive rate)
    ax.axhspan(0.5, 1.0, facecolor='red', alpha=0.1, label='Critical Failure Zone (>50% MSB)')

    ax.set_ylabel('Maximum Shared Bias (False Positive Rate)')
    ax.set_title('Worst-Case Joint Failure Probability by Attack Vector', pad=20)
    
    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('max_shared_bias_plot.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating Error Correlation Heatmap...")
    plot_error_correlation(corr_jury_A, corr_jury_B)
    
    print("Generating Adversarial Collapse Bar Chart...")
    plot_adversarial_collapse(acr_data)
    
    print("Generating Maximum Shared Bias Plot...")
    plot_maximum_shared_bias(msb_data)
    
    print("All plots saved successfully in the current directory.")