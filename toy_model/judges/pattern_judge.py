from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class PatternJudge:
    def __init__(self):
        self.vec = TfidfVectorizer()
        self.clf = LogisticRegression(class_weight='balanced', max_iter=1000)

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)

    def verdict(self, text):
        X = self.vec.transform([text])
        return self.clf.predict_proba(X)[0, 1]  # P(accept)
    
    def scenario_to_full_text(self, s, claim_override=None):
        claim = claim_override if claim_override is not None else s.claim_text
        if s.true_location != s.original_location:
            move_clause = f"It was later moved to {s.true_location}"
            obs_clause = " and the agent saw this happen." if s.agent_observed_move else " without the agent noticing."
        else:
            move_clause = "It was never moved"
            obs_clause = "."
        return (f"{s.agent} put the {s.object} in {s.original_location}. "
                f"{move_clause}{obs_clause} "
                f"Claim: {claim}")
        
    def fit_full_scenarios(self, scenarios):
        texts = [self.scenario_to_full_text(s) for s in scenarios]
        labels = [s.label for s in scenarios]
        self.fit(texts, labels)
        
    