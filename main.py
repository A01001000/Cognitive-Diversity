from judges.pattern_judge import PatternJudge
from judges.causal_judge import CausalJudge
from adversary.search import adversarial_search
from jury.evaluate import hack_rate
from scenarios.generate import make_scenario
import random

from disentangle.train import train_entangled, train_structured
from disentangle.intervention_test import test_structured_intervention, test_entangled_intervention

def run_judge_experiment(n_scenarios=3000):
    rng = random.Random(0)
    scenarios = [make_scenario(rng) for _ in range(n_scenarios)]
    train_set, test_set = scenarios[:2000], scenarios[2000:]

    pj = PatternJudge()
    pj.fit([s.claim_text for s in train_set], [s.label for s in train_set])
    cj = CausalJudge()

    print("Pattern judge hack rate:", hack_rate(test_set, pj, use_text=True))
    print("Causal judge hack rate:", hack_rate(test_set, cj, use_text=False))
    # add your jury/mixed-verdict version here once both are working

def run_disentangle_experiment():
    ent_model = train_entangled()
    struct_model = train_structured()

    shift_struct = test_structured_intervention(struct_model)
    shift_ent, leak_ent = test_entangled_intervention(ent_model)

    print(f"Structured net: action_shift={shift_struct:.3f}, desire_leakage=0.000 (by construction)")
    print(f"Entangled net:  action_shift={shift_ent:.3f}, desire_leakage={leak_ent:.3f}")

if __name__ == "__main__":
    run_judge_experiment()
    run_disentangle_experiment()