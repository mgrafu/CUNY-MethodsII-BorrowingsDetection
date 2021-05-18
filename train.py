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
        endings = get_word_endings(sent[i])
        if endings:
            for end in endings:
                features['last' + str(len(end))] = end
        unique_envs = get_unique_env(sent[i])
        if unique_envs:
            for letter, env in unique_envs:
                features[letter] = 'y'
                for name, value in env:
                    features[name] = value
        if get_accent(sent[i]):
            features["acc"] = get_accent(sent[i])
        non_alphabet_letters = get_non_alphabet(sent[i])
        if non_alphabet_letters:
            for letter in non_alphabet_letters:
                features[letter] = 'y'
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


def get_word_endings(word: bytes) -> List[str]:
    endings = []
    for i in range(4, 0, -1):
        if len(word) > i:
            endings.append(word[-i:])
    return endings


def get_unique_env(word: bytes) -> List[str]:
    letters = ["y", 'q', 'h', 'k']
    letters_present = []
    for i in range(len(word)):
        if word[i] in letters and len(word) > 1:
            environment = []
            if i > 0:
                environment.append(('left', word[i - 1]))
            if i > 1:
                environment.append(('left2', word[i - 2:i - 1]))
            if i + 1 < len(word):
                environment.append(('right', word[i + 1]))
            if i + 2 < len(word):
                environment.append(('right2', word[i + 1:i + 2]))
            if environment:
                letters_present.append((word[i], environment))
    return letters_present


def get_accent(word: bytes) -> str:
    for char in word:
        if char in ['á', 'é', 'í', 'ó', 'ú']:
            return char
        else:
            return False

        
def get_non_alphabet(word: bytes) -> List[str]:
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ']
    letters_present = []
    if len(word) > 1:
        for letter in word:
            if letter not in alphabet:
                letters_present.append(letter)
    return letters_present


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
    
    pred_features, _ = get_file_features("test.conll")
    pred_feats_vect = vectorizer.transform(pred_features)
    final_pred = model.predict(pred_feats_vect)
    
    with open("test.conll", "r") as source, open("results.txt", "w") as sink:
        i = 0
        for word in source:
            word = word.rstrip("\n")
            if word:
                print(word + "\t" + final_pred[i], file=sink)
                i += 1
            else:
                print("", file=sink)


main()
