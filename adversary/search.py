from framing import FRAMINGS

def adversarial_search(scenario, judge, framings=FRAMINGS):
    best_score, best_text = -1, scenario.claim_text
    for f in framings:
        framed = f(scenario.claim_text)
        score = judge.verdict(framed)
        if score > best_score:
            best_score, best_text = score, framed
    return best_text, best_score