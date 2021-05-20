#!/usr/bin/env python3
"""Borrowing detection training and prediction."""


import argparse
import pickle

import model
import util
import sklearn.feature_extraction  # type: ignore
import sklearn.linear_model  # type: ignore
import sklearn.metrics  # type: ignore


def main(args: argparse.Namespace) -> None:
    classifier = model.BorrowingsClassifier()
    model_path = [args.model if args.model else "model"][0]
    if args.train:
        classifier.train(args.train)
        with open(model_path, "wb") as sink:
            pickle.dump(classifier, sink)
    else:
        with open(model_path, "rb") as source:
            classifier = pickle.load(source)
    if args.dev or args.test:
        eval_path = [args.dev if args.dev else args.test][0]
        predictions, gold = classifier.predict(eval_path)
    if args.dev:
        util.evaluate(gold, predictions)
    if args.test:
        util.write_file(predictions, args.test)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", help="path to input training TSV")
    parser.add_argument("--model", help="path to output model")
    parser.add_argument("--dev", help="path to input dev TSV")
    parser.add_argument("--test", help="path to input test file")
    main(parser.parse_args())
