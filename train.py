#!/usr/bin/env python
"""Borrowing detection training."""

from typing import Dict, List, Tuple

import sklearn.feature_extraction  # type: ignore
import sklearn.linear_model  # type: ignore
import sklearn.metrics  # type: ignore


def extract_sent_feats(sent: List[str]):
    feats = []
    for i in range(len(sent)):
        features = dict()
        append_surrounding_tokens(features, sent, i)
        features["cap(t)"] = get_casing(sent[i])
#         if get_accent(sent[i]):
#             features["acc"] = get_accent(sent[i])
        feats.append(features)
    return feats
        

def append_surrounding_tokens(feats: Dict[str, str], sent:List[str], index: int):
    for i in range(-2, 3):
        adjacent_t = index + i
        if adjacent_t > -1 and adjacent_t < len(sent):
            name = "t" + str(["-" if i < 0 else "+"][0]) + str(abs(i))
            feats[name] = ("[NUMERIC]" if sent[adjacent_t].isnumeric() else sent[adjacent_t].casefold())
            if i == -2:
                name = "t-2^t-1"
                feats[name] = get_concatenation(sent, adjacent_t, 1)
            if i == -1 and index + 1 < len(sent):
                name = "t-1^t+1"
                feats[name] = get_concatenation(sent, adjacent_t, 2)
            if i == 1 and index + 2 < len(sent):
                name = "t+1^t+2"
                feats[name] = get_concatenation(sent, adjacent_t, 1)


def get_concatenation(tokens: List[str], index: int, offset: int) -> str:
    concatenation = ("[NUMERIC]" if tokens[index].isnumeric() else tokens[index].casefold())
    concatenation += "^"
    concatenation += ("[NUMERIC]" if tokens[index + offset].isnumeric() else tokens[index + offset].casefold())
    return concatenation


def get_casing(word: bytes) -> str:
    if word.islower():
        casing = "lower"
    elif word.isupper():
        casing = "upper"
    elif word.istitle():
        casing = "title"
    else:
        casing = "mixed"
    return casing


def get_accent(word: bytes) -> str:
    for char in word:
        if char in ['á', 'é', 'í', 'ó', 'ú']:
            return char
        else:
            return False


def get_file_features(path: str):
    features = []
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
                features += extract_sent_feats(sentence)
                labels += sent_labels
                sentence = []
                sent_labels = []
    return features, labels


def main() -> None:
    train_features, train_labels = get_file_features("training.conll")
    vectorizer = sklearn.feature_extraction.DictVectorizer()
    train_feats_vect = vectorizer.fit_transform(train_features)
    model = sklearn.linear_model.LogisticRegression(
        penalty="l1", C=10, solver="liblinear", max_iter=200
    )
    model.fit(train_feats_vect, train_labels)
    test_features, test_labels = get_file_features("dev.conll")
    test_feats_vect = vectorizer.transform(test_features)
    predictions = model.predict(test_feats_vect)
    num_correct = sum(
        [1 for gold, hyp in zip(test_labels, predictions) if gold == hyp]
    )
    print(f"Correct: {num_correct}")
    print(f"Total: {len(test_labels)}")
    print(f"{num_correct/len(test_labels)*100}%")
    f1 = sklearn.metrics.f1_score(test_labels, predictions, average="micro")
    print(f"F1: {f1}")
    print("word\t\tgold\thyp")
    for feats, gold, hyp in zip(test_features, test_labels, predictions):
        if gold != hyp:
            print(feats["t+0"] + "\t" + "\t" + gold + "\t" + hyp)
#     pred_features, _ = get_file_features("test.conll")
#     pred_feats_vect = vectorizer.transform(pred_features)
#     final_pred = model.predict(pred_feats_vect)
#     with open("test.conll", "r") as source, open("results.txt", "w") as sink:
#         for word, label in zip(source, final_pred):
#             word = word.rstrip("\n")
#             if word:
#                 print(word + "\t" + label, file=sink)
#             else:
#                 print("", file=sink)

main()
                
        