from adversary.search import adversarial_search
def baseline_false_positive_rate(scenarios, judge, use_text=True, full_scenario=True):
    false_scenarios = [s for s in scenarios if not s.label]
    fp = 0
    for s in false_scenarios:
        if use_text:
            eval_text = judge.scenario_to_full_text(s) if full_scenario and hasattr(judge, "scenario_to_full_text") else s.claim_text
            score = judge.verdict(eval_text)
        else:
            score = judge.verdict(s)
        if score > 0.5:
            fp += 1
    return fp / len(false_scenarios)

def hack_rate(scenarios, judge, use_text=True):
    hacks = 0
    
    baseline = baseline_false_positive_rate(scenarios, judge, use_text)
    
    print(f"Baseline false positive rate: {baseline:.2%}")
    
    for s in scenarios:
        if s.label: continue  # only care about false claims being wrongly accepted
        if use_text:
            _, score = adversarial_search(s, judge)
        else:
            score = judge.verdict(s)  # causal judge ignores framing
        if score > 0.5:
            hacks += 1
    return hacks / sum(1 for s in scenarios if not s.label)