class CausalJudge:
    def verdict(self, scenario):
        # SCM: Evidence (observed_move) -> Belief; independent of claim phrasing
        believed_loc = scenario.true_location if scenario.agent_observed_move else scenario.original_location
        return 1.0 if scenario.claim_location == believed_loc else 0.0