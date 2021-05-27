"""Logistic regression classifier for borrowings in Spanish."""


import features
import sklearn.feature_extraction  # type: ignore
import sklearn.linear_model  # type: ignore
import sklearn.naive_bayes  # type: ignore


class BorrowingsClassifier:
    """This class stores code for extracting features,
    training, and predicting, along with associated model
    and vectorization data."""

    def __init__(self, model: str):
        self.vectorizer = sklearn.feature_extraction.DictVectorizer()
        if model == "logreg":
            self.classifier = sklearn.linear_model.LogisticRegression(
                penalty="l1", C=10, solver="liblinear", max_iter=100
            )
        elif model == "bayes":
            self.classifier = sklearn.naive_bayes.BernoulliNB()

    def _get_file_features(self, path: str):
        features_dict = []
        labels = []
        with open(path, "r") as f:
            sentence = []
            sent_labels = []
            for line in f:
                line = line.rstrip("\n")
                if line:
                    if "\t" in line:
                        word, label = line.split("\t")
                        sent_labels.append(label)
                    else:
                        word = line
                    sentence.append(word)
                else:
                    features_dict += features.extract_sent_feats(sentence)
                    labels += sent_labels
                    sentence = []
                    sent_labels = []
        return features_dict, labels

    def train(self, path: str):
        x, y = self._get_file_features(path)
        xx = self.vectorizer.fit_transform(x)
        self.classifier.fit(xx, y)

    def predict(self, path: str):
        features, gold = self._get_file_features(path)
        x = self.vectorizer.transform(features)
        return list(self.classifier.predict(x)), gold
