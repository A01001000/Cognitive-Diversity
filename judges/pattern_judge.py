from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class PatternJudge:
    def __init__(self):
        self.vec = TfidfVectorizer()
        self.clf = LogisticRegression()

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)

    def verdict(self, text):
        X = self.vec.transform([text])
        return self.clf.predict_proba(X)[0, 1]  # P(accept)