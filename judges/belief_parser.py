import re
import random

class BeliefParser:

    def parse(self, text):
        observed = ("saw this happen" in text)
        
        if random.random() < 0.05: # TODO: adding for noise BUT CHANGE to ambiguity in actual generated text data input
            observed = not observed

        original = re.search(r"put the .* in (.*?).", text).group(1)
        moved = re.search(r"moved to (.*?)(?: without| and)", text)

        if moved:
            moved = moved.group(1)
        else:
            moved = original

        belief = moved if observed else original

        return {
            "belief_location": belief,
            "observed": observed,
            "original": original,
            "true": moved
        }