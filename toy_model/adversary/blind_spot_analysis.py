from .framing import FRAMINGS

def per_scenario_hacking_framing(scenarios, judge, framings=FRAMINGS, use_text=True):
    """For each false scenario, return which framing (if any) got accepted."""
    results = {}
    for s in scenarios:
        if s.label:
            continue
        hacked_by = None
        for f in framings:
            framed_claim = f(s.claim_text)
            if use_text and hasattr(judge, "scenario_to_full_text"):
                eval_text = judge.scenario_to_full_text(s, claim_override=framed_claim)
                score = judge.verdict(eval_text)
            else:
                score = judge.verdict(s)
            if score > 0.5:
                hacked_by = f.__name__
                break
        results[id(s)] = hacked_by
    return results

def overlap_report(scenarios, judge_a, judge_b, framings=FRAMINGS, use_text_a=True, use_text_b=False):
    hacks_a = per_scenario_hacking_framing(scenarios, judge_a, framings, use_text=use_text_a)
    hacks_b = per_scenario_hacking_framing(scenarios, judge_b, framings, use_text=use_text_b)
    both = sum(1 for k in hacks_a if hacks_a[k] and hacks_b.get(k))
    only_a = sum(1 for k in hacks_a if hacks_a[k] and not hacks_b.get(k))
    only_b = sum(1 for k in hacks_b if hacks_b[k] and not hacks_a.get(k))
    print(f"Hacked by both: {both}, only A: {only_a}, only B: {only_b}")
    return both, only_a, only_b