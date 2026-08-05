import os
import glob
import json
import numpy as np
from scipy.stats import pearsonr
from inspect_ai.log import read_eval_log
from metrics import compute_advanced_metrics

def load_latest_logs():
    log_files = glob.glob("./logs2/*.eval")
    if not log_files:
        raise FileNotFoundError("No .eval files found in ./logs2 directory. Run eval_judges.py first!")

    log_files.sort(key=os.path.getmtime, reverse=True)

    logs = {}
    for filepath in log_files:
        log_data = read_eval_log(filepath)
        model_name = log_data.eval.model
        sample_sys = log_data.samples[0].messages[0].content if log_data.samples[0].messages else ""

        key = None
        if "gpt-4o-mini" in model_name:
            key = "juryA_gpt"
        elif "mistral" in model_name or "nemo" in model_name:
            key = "juryA_mistral_nemo"
        elif "gemini" in model_name:
            if "severe cognitive bias" in sample_sys:
                key = "juryB_pattern"
            elif "strict causal logician" in sample_sys:
                key = "juryB_causal"
            elif "objective evaluator" in sample_sys:
                key = "juryA_gemini_std"

        if key and key not in logs:
            logs[key] = log_data
            print(f"[+] Loaded latest log for '{key}': {os.path.basename(filepath)}")

    return logs

def evaluate_juries():
    logs = load_latest_logs()
    os.makedirs("results", exist_ok=True)
    
    num_samples = len(logs["juryB_pattern"].samples)

    y_true = np.zeros(num_samples)
    y_gpt = np.zeros(num_samples)
    y_mistral = np.zeros(num_samples)
    y_pattern = np.zeros(num_samples)
    y_causal = np.zeros(num_samples)
    trap_types = []

    for i in range(num_samples):
        sample_pattern = logs["juryB_pattern"].samples[i]
        trap = sample_pattern.metadata.get("trap_type", "baseline")
        trap_types.append(trap)
        
        y_true[i] = 1 if sample_pattern.target == "True" else 0
        y_gpt[i] = 1 if logs["juryA_gpt"].samples[i].scores["verdict_scorer"].answer == "True" else 0
        y_mistral[i] = 1 if logs["juryA_mistral_nemo"].samples[i].scores["verdict_scorer"].answer == "True" else 0
        y_pattern[i] = 1 if sample_pattern.scores["verdict_scorer"].answer == "True" else 0
        y_causal[i] = 1 if logs["juryB_causal"].samples[i].scores["verdict_scorer"].answer == "True" else 0

    unique_traps = list(set(trap_types))
    if "baseline" in unique_traps:
        unique_traps.remove("baseline")
        unique_traps = ["baseline"] + sorted(unique_traps)
    else:
        unique_traps = sorted(unique_traps)
        
    # --- CALCULATE JOINT ERROR RATE (JER) INSTEAD OF MSB ---
    jer_A = {}
    jer_B = {}
    jer_C = {}

    # --- TRACK INDIVIDUAL ERRORS & JOINT ERRORS ---
    indiv_gpt, indiv_mistral = {}, {}
    indiv_pattern, indiv_causal = {}, {}
    jer_A, jer_B, jer_C = {}, {}, {}
    
    for trap in unique_traps:
        indices = [idx for idx, t in enumerate(trap_types) if t == trap]
        sub_y = y_true[indices]
        
        # Joint Error = ALL judges in the jury got it WRONG
        err_A = (y_gpt[indices] != sub_y) & (y_mistral[indices] != sub_y)
        err_B = (y_pattern[indices] != sub_y) & (y_causal[indices] != sub_y)
        err_C = (y_gpt[indices] != sub_y) & (y_mistral[indices] != sub_y) & (y_pattern[indices] != sub_y) & (y_causal[indices] != sub_y)
        
        jer_A[trap] = np.sum(err_A) / len(sub_y) if len(sub_y) > 0 else 0.0
        jer_B[trap] = np.sum(err_B) / len(sub_y) if len(sub_y) > 0 else 0.0
        jer_C[trap] = np.sum(err_C) / len(sub_y) if len(sub_y) > 0 else 0.0
        
        # Individual Errors
        err_gpt = (y_gpt[indices] != sub_y)
        err_mistral = (y_mistral[indices] != sub_y)
        err_pattern = (y_pattern[indices] != sub_y)
        err_causal = (y_causal[indices] != sub_y)
        
        indiv_gpt[trap] = np.sum(err_gpt) / len(sub_y) if len(sub_y) > 0 else 0.0
        indiv_mistral[trap] = np.sum(err_mistral) / len(sub_y) if len(sub_y) > 0 else 0.0
        indiv_pattern[trap] = np.sum(err_pattern) / len(sub_y) if len(sub_y) > 0 else 0.0
        indiv_causal[trap] = np.sum(err_causal) / len(sub_y) if len(sub_y) > 0 else 0.0
        
        # Joint Errors
        jer_A[trap] = np.sum(err_gpt & err_mistral) / len(sub_y) if len(sub_y) > 0 else 0.0
        jer_B[trap] = np.sum(err_pattern & err_causal) / len(sub_y) if len(sub_y) > 0 else 0.0
        jer_C[trap] = np.sum(err_gpt & err_mistral & err_pattern & err_causal) / len(sub_y) if len(sub_y) > 0 else 0.0

    worst_trap_A = max(jer_A, key=jer_A.get)
    worst_trap_B = max(jer_B, key=jer_B.get)
    worst_trap_C = max(jer_C, key=jer_C.get)
    
    # Calculate Adversarial Collapse Ratio (ACR) relative to Baseline Joint Error
    baseline_A = max(jer_A.get("baseline", 0.001), 0.001)
    baseline_B = max(jer_B.get("baseline", 0.001), 0.001)
    baseline_C = max(jer_C.get("baseline", 0.001), 0.001)
    
    acr_A = {trap: jer_A[trap] / baseline_A for trap in unique_traps}
    acr_B = {trap: jer_B[trap] / baseline_B for trap in unique_traps}
    acr_C = {trap: jer_C[trap] / baseline_C for trap in unique_traps}
    
    # --- FORMAT REPORT ---
    report = []
    
    report.append("="*85)
    report.append("      JURY ROBUSTNESS: ADVANCED METRICS EVALUATION (JOINT ERROR RATE)")
    report.append("="*85)
    
    report.append("\n1. Maximum Joint Error Rate (Worst-Case Total Failure)")
    report.append(f"   Jury A (Model Div):     {jer_A[worst_trap_A]:.1%} (Worst trap: {worst_trap_A})")
    report.append(f"   Jury B (Cognitive Div): {jer_B[worst_trap_B]:.1%} (Worst trap: {worst_trap_B})")
    report.append(f"   Jury C (Super Jury):    {jer_C[worst_trap_C]:.1%} (Worst trap: {worst_trap_C})")

    report.append("\n2. Adversarial Collapse Ratio (ACR) based on Joint Error")
    report.append(f"{'Trap Vector':<22} | {'Jury A ACR':<12} | {'Jury B ACR':<12} | {'Jury C ACR':<12}")
    report.append("-" * 68)
    
    for trap in unique_traps:
        report.append(f"{trap:<22} | {acr_A[trap]:<11.1f}x | {acr_B[trap]:<11.1f}x | {acr_C[trap]:<11.1f}x")

    report.append("="*85 + "\n")
    
    report.append("="*85)
    report.append("      INDIVIDUAL VS JURY ERROR RATES")
    report.append("="*85)
    
    report.append(f"{'Trap Vector':<20} | {'GPT-4o':<8} | {'Mistral':<8} | {'Jury A':<10} || {'Pattern':<8} | {'Causal':<8} | {'Jury B':<10}")
    report.append("-" * 85)
    
    for trap in unique_traps:
        report.append(f"{trap:<20} | {indiv_gpt[trap]:<8.1%} | {indiv_mistral[trap]:<8.1%} | {jer_A[trap]:<10.1%} || "
                      f"{indiv_pattern[trap]:<8.1%} | {indiv_causal[trap]:<8.1%} | {jer_B[trap]:<10.1%}")

    report.append("="*85 + "\n")
    final_output = "\n".join(report)
    print("\n" + final_output)

    with open("results/individual_vs_jury_results.txt", "w") as f:
        f.write(final_output)

    # Export JSON data for plots.py
    rho_A, _ = pearsonr(y_gpt - y_true, y_mistral - y_true)
    rho_B, _ = pearsonr(y_pattern - y_true, y_causal - y_true)
    
    # Export JSON data for plotting
    json_data = {
        "rho_A": float(rho_A),
        "rho_B": float(rho_B),
        "traps": unique_traps,
        "acr_A": {trap: float(acr_A[trap]) for trap in unique_traps},
        "acr_B": {trap: float(acr_B[trap]) for trap in unique_traps},
        "acr_C": {trap: float(acr_C[trap]) for trap in unique_traps},
        "msb_A": {trap: float(jer_A[trap]) for trap in unique_traps},
        "msb_B": {trap: float(jer_B[trap]) for trap in unique_traps},
        "msb_C": {trap: float(jer_C[trap]) for trap in unique_traps},
        "indiv_gpt": {trap: float(indiv_gpt[trap]) for trap in unique_traps},
        "indiv_mistral": {trap: float(indiv_mistral[trap]) for trap in unique_traps},
        "indiv_pattern": {trap: float(indiv_pattern[trap]) for trap in unique_traps},
        "indiv_causal": {trap: float(indiv_causal[trap]) for trap in unique_traps},
        "jer_A": {trap: float(jer_A[trap]) for trap in unique_traps},
        "jer_B": {trap: float(jer_B[trap]) for trap in unique_traps},
        "jer_C": {trap: float(jer_C[trap]) for trap in unique_traps}
    }
    
    with open("results/individual_vs_jury_results.json", "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"[+] Text report and JSON saved to ./results/")

if __name__ == "__main__":
    evaluate_juries()