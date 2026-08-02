import os
import glob
from inspect_ai.log import read_eval_log
from .metrics import compute_advanced_metrics

def load_latest_logs():
    """Finds and categorizes evaluation log files from ./logs based on model and system prompt signatures."""
    log_files = glob.glob("./logs/*.eval")
    if not log_files:
        raise FileNotFoundError("No .eval files found in ./logs directory. Run eval_judges.py first!")

    logs = {}
    for filepath in log_files:
        log_data = read_eval_log(filepath)
        model_name = log_data.eval.model
        sample_sys = log_data.samples[0].messages[0].content

        if "gpt-4o-mini" in model_name:
            logs["juryA_gpt"] = log_data
        elif "haiku" in model_name:
            logs["juryA_haiku"] = log_data
        elif "gemini" in model_name:
            if "severe cognitive bias" in sample_sys:
                logs["juryB_pattern"] = log_data
            elif "strict causal logician" in sample_sys:
                logs["juryB_causal"] = log_data
            elif "objective evaluator" in sample_sys:
                logs["juryA_gemini_std"] = log_data

    return logs

def evaluate_juries():
    logs = load_latest_logs()
    
    num_samples = len(logs["juryB_pattern"].samples)

    y_true = np.zeros(num_samples)
    y_gpt = np.zeros(num_samples)
    y_gemini_std = np.zeros(num_samples)
    y_pattern = np.zeros(num_samples)
    y_causal = np.zeros(num_samples)
    trap_types = []

    for i in range(num_samples):
        sample_pattern = logs["juryB_pattern"].samples[i]
        trap_types.append(sample_pattern.metadata.get("trap_type", "baseline"))
        
        # Convert "True"/"False" strings to 1/0
        y_true[i] = 1 if sample_pattern.target == "True" else 0
        
        y_gpt[i] = 1 if logs["juryA_gpt"].samples[i].scores["verdict_scorer"].answer == "True" else 0
        y_gemini_std[i] = 1 if logs["juryA_gemini_std"].samples[i].scores["verdict_scorer"].answer == "True" else 0
        y_pattern[i] = 1 if sample_pattern.scores["verdict_scorer"].answer == "True" else 0
        y_causal[i] = 1 if logs["juryB_causal"].samples[i].scores["verdict_scorer"].answer == "True" else 0

    # Calculate metrics for Jury A (Matched Vendors) and Jury B (Cognitive Diversity)
    metrics_A = compute_advanced_metrics(y_true, y_gemini_std, y_gpt, trap_types)
    metrics_B = compute_advanced_metrics(y_true, y_pattern, y_causal, trap_types)

    print("\n" + "="*80)
    print("      JURY ROBUSTNESS: ADVANCED METRICS EVALUATION")
    print("="*80)
    
    print(f"\n1. Error Correlation (Pearson's rho)")
    print(f"   Jury A (Model Diversity):     {metrics_A['error_correlation_rho']:.3f}")
    print(f"   Jury B (Cognitive Diversity): {metrics_B['error_correlation_rho']:.3f}")
    
    print(f"\n2. Maximum Shared Bias (False Positive)")
    print(f"   Jury A: {metrics_A['max_shared_bias_fp']:.1%} (Worst trap: {metrics_A['worst_case_leaf']})")
    print(f"   Jury B: {metrics_B['max_shared_bias_fp']:.1%} (Worst trap: {metrics_B['worst_case_leaf']})")

    print("\n3. Adversarial Collapse Ratio (ACR)")
    print(f"{'Trap Vector':<20} | {'Jury A ACR':<15} | {'Jury B ACR':<15}")
    print("-" * 55)
    
    # Ensure baseline prints first
    print(f"{'baseline':<20} | {metrics_A['collapse_ratios'].get('baseline', 1.0):.1f}x            | {metrics_B['collapse_ratios'].get('baseline', 1.0):.1f}x")
    
    for trap in set(trap_types):
        if trap != "baseline":
            acr_a = metrics_A['collapse_ratios'].get(trap, 0)
            acr_b = metrics_B['collapse_ratios'].get(trap, 0)
            print(f"{trap:<20} | {acr_a:<4.1f}x           | {acr_b:<4.1f}x")

    print("="*80 + "\n")

if __name__ == "__main__":
    evaluate_juries()