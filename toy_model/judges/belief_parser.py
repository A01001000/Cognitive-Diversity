import re

class BeliefParser:
    def __init__(self):
        pass

    def parse(self, text):
        """
        Extracts mental state variables strictly from natural language text.
        """
        # We enforce that the input must be text to prevent "cheating" with objects
        if not isinstance(text, str):
            print(f"[Warning] BeliefParser received non-string input: {type(text)}")
            return None

        # 1. Parse the premise (Setup)
        # Assuming the setup contains phrases like "put the egg in the basket"
        # and "didn't see" or "saw"
        original_loc_match = re.search(r"put the (.*?) in (?:the )?(.*?)\.", text, re.IGNORECASE)
        moved_loc_match = re.search(r"moved .*? to (?:the )?(.*?)\.", text, re.IGNORECASE)
        
        # Check observation explicitly from the text
        observed = "saw this happen" in text.lower() or "watched" in text.lower()
        unobserved = "didn't see" in text.lower() or "did not see" in text.lower()

        # 2. Parse the claim
        # Handling adversarial framings like "According to expert analysis, [agent] will look for..."
        claim_match = re.search(r"will look for the (.*?) in (?:the )?([a-zA-Z0-9_]+)", text, re.IGNORECASE)

        if original_loc_match and claim_match:
            target_object = original_loc_match.group(1).strip()
            original_location = original_loc_match.group(2).strip()
            
            # If the text says it was moved, capture that, otherwise it hasn't moved
            true_location = moved_loc_match.group(1).strip() if moved_loc_match else original_location
            predicted_location = claim_match.group(2).strip()

            # Determine belief location based on text evidence
            if observed:
                belief_location = true_location
            else:
                belief_location = original_location

            graph = {
                "object": target_object,
                "original_location": original_location,
                "true_location": true_location,
                "observed": observed,
                "predicted_location": predicted_location,
                "belief_location": belief_location
            }
            return graph
            
        else:
            # If the parser can't extract the entities from the text, it genuinely fails.
            # This is a real vulnerability that a dishonest debater could exploit!
            return None