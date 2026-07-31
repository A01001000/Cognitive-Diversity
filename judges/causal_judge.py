from .belief_parser import BeliefParser

class CausalJudge:
    def __init__(self):
        self.parser = BeliefParser()

    def verdict(self, text):
        # We don't unpack Scenario objects here. The pipeline must provide text.
        if not isinstance(text, str):
            # If a Scenario object slipped through, convert it to a string, 
            # though ideally your evaluation loop passes the actual narrative string.
            if hasattr(text, 'text'):
                text = text.text
            else:
                text = str(text)
                
        graph = self.parser.parse(text)
        
        # If the parser is confused by adversarial framing, it abstains.
        if graph is None:
            return 0.5
        
        # Extract the claim from the text string
        if "Claim:" in text:
            claim_str = text.split("Claim:")[-1].strip().lower()
        else:
            claim_str = text.lower()
            
        belief_location = graph.get("belief_location", "").lower()
        
        if belief_location and belief_location in claim_str:
            return 1.0
        else:
            return 0.0
    