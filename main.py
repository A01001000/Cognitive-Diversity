import random

from disentangle.train import train_entangled, train_structured
from disentangle.intervention_test import test_structured_intervention, test_structured_intervention_filtered, test_entangled_intervention, alpha_sweep
from disentangle.probe import probe_accuracy

from judges.pattern_judge_mlp import PatternJudgeMLP
from judges.pattern_judge import PatternJudge
from judges.causal_judge import CausalJudge

from jury.evaluate import hack_rate
from jury.aggregate import jury_hack_rate, jury_hack_rate_3

from adversary.search import adversarial_search
from adversary.blind_spot_analysis import overlap_report
from scenarios.generate import make_scenario
from misc import label_balance, save_results, repeat, plot_alpha_sweep

def run_jury_comparison(test_set, pj, cj):
    homog_pattern = jury_hack_rate(test_set, [(pj, True), (pj, True)])
    homog_causal  = jury_hack_rate(test_set, [(cj, True), (cj, True)])
    mixed         = jury_hack_rate(test_set, [(pj, True), (cj, True)])
    print(f"Homogeneous pattern jury: {homog_pattern:.3f}")
    print(f"Homogeneous causal jury:  {homog_causal:.3f}")
    print(f"Mixed jury:               {mixed:.3f}")
    
def run_judge_experiment(seed=3, n_scenarios=3000):
    rng = random.Random(seed)
    scenarios = [make_scenario(rng) for _ in range(n_scenarios)]
    train_set, test_set = scenarios[:2000], scenarios[2000:]
    
    label_balance(train_set)

    pj_lr = PatternJudge()
    pj_lr.fit([s.claim_text for s in train_set], [s.label for s in train_set])
    print("LR Pattern judge hack rate:", hack_rate(test_set, pj_lr, use_text=True))
    pj_lr.fit_full_scenarios(train_set)
    print("LR Pattern judge (full scenario) hack rate:", hack_rate(test_set, pj_lr, use_text=True))  
    
    rng2 = random.Random(seed + 1000)
    scenarios_v2 = [make_scenario(rng2) for _ in range(n_scenarios)]
    train_set_v2 = scenarios_v2[:2000]
    pj_lr_v2 = PatternJudge()
    pj_lr_v2.fit([s.claim_text for s in train_set_v2], [s.label for s in train_set_v2])
    pj_lr_v2.fit_full_scenarios(train_set_v2)

    pj_mlp = PatternJudgeMLP()
    pj_mlp.fit([s.claim_text for s in train_set], [s.label for s in train_set])
    print("MLP Pattern judge hack rate:", hack_rate(test_set, pj_mlp, use_text=True))
    pj_mlp.fit_full_scenarios(train_set)
    print("MLP Pattern judge (full scenario) hack rate:", hack_rate(test_set, pj_mlp, use_text=True))  
    
    cj = CausalJudge()
    print("Causal judge hack rate:", hack_rate(test_set, cj, use_text=True))
    
    print("--- Blind spot overlap: LR vs causal ---")
    overlap_report(test_set, pj_lr, cj)

    print("--- Blind spot overlap: LR vs MLP ---")
    overlap_report(test_set, pj_lr, pj_mlp, use_text_b=True)
    
    print("Homogeneous LR jury:", jury_hack_rate(test_set, [(pj_lr, True), (pj_lr, True)]))
    print("Homogeneous MLP jury:", jury_hack_rate(test_set, [(pj_mlp, True), (pj_mlp, True)]))
    print("Two-pattern-type jury (LR+MLP):", jury_hack_rate(test_set, [(pj_lr, True), (pj_mlp, True)]))
    print("Mixed jury (LR+causal):", jury_hack_rate(test_set, [(pj_lr, True), (cj, True)]))
    print("Three-pattern-type jury (2 LR + 1 MLP):", jury_hack_rate_3(test_set, [(pj_lr, True), (pj_mlp, True), (pj_lr_v2, True)]))
    print("Mixed: Two pattern + 1 causal jury (1 LR + 1 MLP + 1 causal):", jury_hack_rate_3(test_set, [(pj_lr, True), (pj_mlp, True), (cj, True)]))

def run_disentangle_experiment(seed=3):
    alpha_sweep_results = []
    
    ent_model = train_entangled(seed)
    struct_model = train_structured(seed)
    
    belief_acc = probe_accuracy(ent_model, "belief")
    desire_acc = probe_accuracy(ent_model, "desire")
    print(f"Entangled net: belief probe acc={belief_acc:.3f}, desire probe acc={desire_acc:.3f}")

    shift_struct = test_structured_intervention_filtered(struct_model, seed)
    shift_ent, leak_ent = test_entangled_intervention(ent_model, seed)
    alpha_sweep_results = alpha_sweep(ent_model, seed)

    print(f"Structured net: action_shift={shift_struct:.3f}, desire_leakage=0.000 (by construction)")
    print(f"Entangled net:  action_shift={shift_ent:.3f}, desire_leakage={leak_ent:.3f}")
    print(f"Alpha Sweep: {alpha_sweep_results}")
    
    return {
        "structured_action_shift": shift_struct,
        "entangled_action_shift": shift_ent,
        "entangled_desire_leakage": leak_ent,
    }

if __name__ == "__main__":
    run_judge_experiment()
    run_disentangle_experiment()
    # repeat(run_disentangle_experiment()) 
    
    # SAVE RESULTS with save_results(results_dict, name) 