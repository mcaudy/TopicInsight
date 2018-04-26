from sklearn import base
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class PreprocessTokensTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, vectorizer=CountVectorizer(), tfidf=TfidfTransformer()):
        self.vectorizer = vectorizer
        self.tfidf = tfidf
        
        self.vectorizer.lowercase = False
        self.vectorizer.preprocessor = lambda x: x
        self.vectorizer.tokenizer = lambda x: x

    def fit(self, X, y=None):
        self.tfidf.fit(self.vectorizer.fit_transform(X))
        return self

    def transform(self, X):
        return self.tfidf.transform(self.vectorizer.transform(X))