from .framing import FRAMINGS

def adversarial_search(scenario, judge, framings=FRAMINGS, full_scenario=True):
    best_score, best_text = -1, scenario.claim_text
    for f in framings:
        framed_claim = f(scenario.claim_text)
        if full_scenario and hasattr(judge, "scenario_to_full_text"):
            eval_text = judge.scenario_to_full_text(scenario, claim_override=framed_claim)
        else:
            eval_text = framed_claim
        score = judge.verdict(eval_text)
        if score > best_score:
            best_score, best_text = score, eval_text
    return best_text, best_score