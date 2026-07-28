from search import adversarial_search

def hack_rate(scenarios, judge, use_text=True):
    hacks = 0
    for s in scenarios:
        if s.label: continue  # only care about false claims being wrongly accepted
        if use_text:
            _, score = adversarial_search(s, judge)
        else:
            score = judge.verdict(s)  # causal judge ignores framing
        if score > 0.5:
            hacks += 1
    return hacks / sum(1 for s in scenarios if not s.label)