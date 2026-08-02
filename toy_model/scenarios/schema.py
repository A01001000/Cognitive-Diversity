from dataclasses import dataclass

@dataclass
class Scenario:
    agent: str
    object: str
    original_location: str
    true_location: str         # may differ from original if moved
    agent_observed_move: bool  # did the agent see the object move?
    claim_text: str            # the natural-language claim to be judged
    claim_location: str        # what location the claim asserts
    label: bool                # is claim_text actually true?