from .belief_parser import BeliefParser
import numpy as np
class CausalJudge:

    def __init__(self):
        self.parser = BeliefParser()

    def verdict(self, text):
        graph = self.parser.parse(text)
        claim = text.split("Claim:")[-1]
        return float(
            graph["belief_location"] in claim
        )
    
    def fit(self, scenarios):
        # causal judge isn't trained, just reasons
        pass
    
    def predict(self, scenarios):
        preds = []

        for s in scenarios:
            text = self.scenario_to_full_text(s)
            preds.append(self.verdict(text))

        return np.array(preds)