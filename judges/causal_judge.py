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

    