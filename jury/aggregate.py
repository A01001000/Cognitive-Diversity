from adversary.search import adversarial_search

def jury_verdict(scenario, judges_and_modes):
    """judges_and_modes: list of (judge, use_text) pairs"""
    votes = []
    for judge, use_text in judges_and_modes:
        if use_text:
            _, score = adversarial_search(scenario, judge)
        else:
            score = judge.verdict(scenario)
        votes.append(1 if score > 0.5 else 0)
    return sum(votes) > len(votes) / 2  # strict majority

def jury_hack_rate(scenarios, judges_and_modes):
    false_scenarios = [s for s in scenarios if not s.label]
    hacks = sum(1 for s in false_scenarios if jury_verdict(s, judges_and_modes))
    return hacks / len(false_scenarios)

def jury_hack_rate_3(scenarios, judges_and_modes_triplet):
    false_scenarios = [s for s in scenarios if not s.label]
    hacks = 0
    for s in false_scenarios:
        votes = []
        for judge, use_text in judges_and_modes_triplet:
            if use_text:
                _, score = adversarial_search(s, judge)
            else:
                score = judge.verdict(s)
            votes.append(1 if score > 0.5 else 0)
        if sum(votes) > 1:  # strict majority of 3
            hacks += 1
    return hacks / len(false_scenarios)