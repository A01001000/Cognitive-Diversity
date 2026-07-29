from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import numpy as np

class PatternJudgeMLP:
    def __init__(self):
        self.vec = TfidfVectorizer()
        self.clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1500)

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        
        # oversample minority class to 1:1 so it doesn't get drowned out by the majority class 
        labels = np.array(labels)
        idx_true = np.where(labels)[0]
        idx_false = np.where(~labels)[0]
        n_target = len(idx_true)
        idx_false_up = np.random.choice(idx_false, size=n_target, replace=True)
        idx_all = np.concatenate([idx_true, idx_false_up])
        self.clf.fit(X[idx_all], labels[idx_all])

    def verdict(self, text):
        X = self.vec.transform([text])
        return self.clf.predict_proba(X)[0, 1]
    
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