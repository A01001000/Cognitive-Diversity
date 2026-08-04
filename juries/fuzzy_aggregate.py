import os
import glob
import json
import numpy as np
from inspect_ai.log import read_eval_log
from metrics import compute_advanced_metrics

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
        elif "llama3" in model_name:
            logs["juryA_llama3_70b"] = log_data
        elif "mistral" in model_name or "nemo" in model_name:
            logs["juryA_mistral_nemo"] = log_data
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
    os.makedirs("results", exist_ok=True)
    
    num_samples = len(logs["juryB_pattern"].samples)

    # Initialize arrays
    y_true = np.zeros(num_samples)
    y_gpt = np.zeros(num_samples)
    y_mistral = np.zeros(num_samples)
    y_pattern = np.zeros(num_samples)
    y_causal = np.zeros(num_samples)
    trap_types = []

    for i in range(num_samples):
        sample_pattern = logs["juryB_pattern"].samples[i]
        
        # Capture trap type dynamically (supports baseline, semantic, referential, fuzzy, etc.)
        trap = sample_pattern.metadata.get("trap_type", "baseline")
        trap_types.append(trap)
        
        # Convert "True"/"False" target strings to binary 1/0
        y_true[i] = 1 if sample_pattern.target == "True" else 0
        
        # Parsed predictions
        y_gpt[i] = 1 if logs["juryA_gpt"].samples[i].scores["verdict_scorer"].answer == "True" else 0
        y_mistral[i] = 1 if logs["juryA_mistral_nemo"].samples[i].scores["verdict_scorer"].answer == "True" else 0
        y_pattern[i] = 1 if sample_pattern.scores["verdict_scorer"].answer == "True" else 0
        y_causal[i] = 1 if logs["juryB_causal"].samples[i].scores["verdict_scorer"].answer == "True" else 0

    # Calculate metrics for Jury A (GPT + Mistral) and Jury B (Pattern + Causal)
    metrics_A = compute_advanced_metrics(y_true, y_gpt, y_mistral, trap_types)
    metrics_B = compute_advanced_metrics(y_true, y_pattern, y_causal, trap_types)

    # --- CALCULATE JURY C (Super Jury: GPT + Mistral + Pattern + Causal) ---
    msb_fp_C = {}
    unique_traps = list(set(trap_types))
    
    # Sort trap types so 'baseline' is always evaluated first
    if "baseline" in unique_traps:
        unique_traps.remove("baseline")
        unique_traps = ["baseline"] + sorted(unique_traps)
    else:
        unique_traps = sorted(unique_traps)

    for trap in unique_traps:
        indices = [idx for idx, t in enumerate(trap_types) if t == trap]
        sub_y = y_true[indices]
        
        # False Positive occurs when ground truth is False (0), but ALL 4 judges output True (1)
        false_claims = (sub_y == 0)
        if np.sum(false_claims) > 0:
            joint_fa = np.sum((y_gpt[indices][false_claims] == 1) & 
                              (y_mistral[indices][false_claims] == 1) & 
                              (y_pattern[indices][false_claims] == 1) & 
                              (y_causal[indices][false_claims] == 1)) / np.sum(false_claims)
        else:
            joint_fa = 0.0
        msb_fp_C[trap] = joint_fa
        
    worst_case_trap_C = max(msb_fp_C, key=msb_fp_C.get)
    max_shared_bias_C = msb_fp_C[worst_case_trap_C]
    
    baseline_asr_C = msb_fp_C.get("baseline", 0.001)
    acr_C = {trap: msb_fp_C[trap] / max(baseline_asr_C, 0.001) for trap in unique_traps}

    # --- FORMAT AND SAVE TEXT REPORT ---
    report = []
    report.append("="*85)
    report.append("      JURY ROBUSTNESS: ADVANCED METRICS EVALUATION")
    report.append("="*85)
    
    report.append("\n1. Error Correlation (Pearson's rho)")
    report.append(f"   Jury A (Model Diversity):     {metrics_A['error_correlation_rho']:.3f}")
    report.append(f"   Jury B (Cognitive Diversity): {metrics_B['error_correlation_rho']:.3f}")
    
    report.append("\n2. Maximum Shared Bias (Worst-Case False Positive Rate)")
    report.append(f"   Jury A: {metrics_A['max_shared_bias_fp']:.1%} (Worst trap: {metrics_A['worst_case_leaf']})")
    report.append(f"   Jury B: {metrics_B['max_shared_bias_fp']:.1%} (Worst trap: {metrics_B['worst_case_leaf']})")
    report.append(f"   Jury C: {max_shared_bias_C:.1%} (Worst trap: {worst_case_trap_C})")

    report.append("\n3. Adversarial Collapse Ratio (ACR)")
    report.append(f"{'Trap Vector':<22} | {'Jury A ACR':<12} | {'Jury B ACR':<12} | {'Jury C ACR':<12}")
    report.append("-" * 68)
    
    for trap in unique_traps:
        val_a = metrics_A['collapse_ratios'].get(trap, 0)
        val_b = metrics_B['collapse_ratios'].get(trap, 0)
        val_c = acr_C.get(trap, 0)
        report.append(f"{trap:<22} | {val_a:<11.1f}x | {val_b:<11.1f}x | {val_c:<11.1f}x")

    report.append("="*85 + "\n")

    final_output = "\n".join(report)
    print("\n" + final_output)

    # Save readable text report
    txt_filename = "results/fuzzy_jury_evaluation_results.txt"
    with open(txt_filename, "w") as f:
        f.write(final_output)
    print(f"[+] Text report saved to {txt_filename}")

    # --- EXPORT STRUCTURED JSON DATA FOR PLOTS.PY ---
    json_data = {
        "rho_A": float(metrics_A['error_correlation_rho']),
        "rho_B": float(metrics_B['error_correlation_rho']),
        "traps": unique_traps,
        "acr_A": {trap: float(metrics_A['collapse_ratios'].get(trap, 1.0)) for trap in unique_traps},
        "acr_B": {trap: float(metrics_B['collapse_ratios'].get(trap, 1.0)) for trap in unique_traps},
        "acr_C": {trap: float(acr_C.get(trap, 1.0)) for trap in unique_traps},
        "msb_A": {trap: float(metrics_A['msb_fp_dict'].get(trap, 0.0)) for trap in unique_traps},
        "msb_B": {trap: float(metrics_B['msb_fp_dict'].get(trap, 0.0)) for trap in unique_traps},
        "msb_C": {trap: float(msb_fp_C.get(trap, 0.0)) for trap in unique_traps}
    }
    
    json_filename = "results/jury_evaluation_results.json"
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"[+] JSON metric data saved to {json_filename}")

def debug_raw_predictions():
    logs = load_latest_logs()
    num_samples = len(logs["juryB_pattern"].samples)
    
    print("\n" + "="*80)
    print(" RAW DATA EXTRACTION CHECK (First 5 Samples)")
    print("="*80)
    
    for i in range(min(5, num_samples)):
        sample_pattern = logs["juryB_pattern"].samples[i]
        trap_type = sample_pattern.metadata.get("trap_type", "baseline")
        ground_truth = sample_pattern.target
        
        ans_gpt = logs["juryA_gpt"].samples[i].scores["verdict_scorer"].answer
        ans_mistral = logs["juryA_mistral_nemo"].samples[i].scores["verdict_scorer"].answer
        ans_pattern = sample_pattern.scores["verdict_scorer"].answer
        ans_causal = logs["juryB_causal"].samples[i].scores["verdict_scorer"].answer
        
        print(f"Sample {i+1} [{trap_type}] | Ground Truth: {ground_truth}")
        print(f"  -> GPT-4o-mini:    {ans_gpt}")
        print(f"  -> Mistral-Nemo:   {ans_mistral}")
        print(f"  -> Gemini Pattern: {ans_pattern}")
        print(f"  -> Gemini Causal:  {ans_causal}")
        print("-" * 40)

if __name__ == "__main__":
    debug_raw_predictions()
    evaluate_juries()