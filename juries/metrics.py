import numpy as np

def compute_advanced_metrics(y_true, y_pred1, y_pred2, trap_types):
    """Computes Error Correlation, MSB, and ACR."""
    # 1. Error vectors
    e1 = (y_pred1 != y_true).astype(int)
    e2 = (y_pred2 != y_true).astype(int)
    
    # 2. Pearson Correlation of Errors (rho_E)
    if np.std(e1) == 0 or np.std(e2) == 0:
        rho_e = 0.0
    else:
        rho_e = np.corrcoef(e1, e2)[0, 1]
        
    # 3. Maximum Shared Bias (MSB) per trap type
    msb_fp = {}
    unique_traps = set(trap_types)
    
    for trap in unique_traps:
        indices = [idx for idx, t in enumerate(trap_types) if t == trap]
        sub_y = y_true[indices]
        sub_p1 = y_pred1[indices]
        sub_p2 = y_pred2[indices]
        
        # Joint False Accept Rate (Both predict 1 when target is 0)
        # Assuming target 0 means ground truth is False (the trap should be rejected)
        false_claims = (sub_y == 0)
        if np.sum(false_claims) > 0:
            joint_fa = np.sum((sub_p1[false_claims] == 1) & (sub_p2[false_claims] == 1)) / np.sum(false_claims)
        else:
            joint_fa = 0.0
            
        msb_fp[trap] = joint_fa
        
    worst_case_trap = max(msb_fp, key=msb_fp.get)
    max_shared_bias = msb_fp[worst_case_trap]
    
    # 4. Adversarial Collapse Ratio (ACR)
    baseline_asr = msb_fp.get("baseline", 0.001) # Avoid division by zero
    acr = {trap: msb_fp[trap] / max(baseline_asr, 0.001) for trap in unique_traps}
    
    return {
        "error_correlation_rho": rho_e,
        "max_shared_bias_fp": max_shared_bias,
        "worst_case_leaf": worst_case_trap,
        "collapse_ratios": acr,
        "msb_fp_dict": msb_fp
    }