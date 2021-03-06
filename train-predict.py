#!/usr/bin/env python3.8
"""Borrowing detection training and prediction."""


import argparse
import pickle

import model
import util


def main(args: argparse.Namespace) -> None:
    classifier = model.BorrowingsClassifier(args.modeltype)
    model_path = args.modelpath if args.modelpath else "model"
    if args.train:
        classifier.train(args.train)
        with open(model_path, "wb") as sink:
            pickle.dump(classifier, sink)
    else:
        with open(model_path, "rb") as source:
            classifier = pickle.load(source)
    if args.dev or args.test:
        eval_path = args.dev if args.dev else args.test
        predictions, gold = classifier.predict(eval_path)
    if args.dev:
        util.evaluate(gold, predictions)
    if args.test:
        util.write_file(predictions, args.test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modeltype", help="model type (logreg or bayes)")
    parser.add_argument("--train", help="path to input training TSV")
    parser.add_argument("--modelpath", help="path to output model")
    parser.add_argument("--dev", help="path to input dev TSV")
    parser.add_argument("--test", help="path to input test file")
    main(parser.parse_args())
