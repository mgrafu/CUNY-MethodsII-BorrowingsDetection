"""Helper utilities."""


from typing import List, Dict


def write_file(labels: List[str], words: str):
    with open(words, "r") as source, open("results.txt", "w") as sink:
        i = 0
        for word in source:
            word = word.rstrip("\n").split("\t")[0]
            if word:
                print(word + "\t" + labels[i], file=sink)
                i += 1
            else:
                print("", file=sink)


def evaluate(gold_labels: List[str], pred_labels: List[str]):
    num_correct = sum(
        [1 for gold, hyp in zip(gold_labels, pred_labels) if gold == hyp]
    )
    size = len(gold_labels)
    print(f"Correct: {num_correct}")
    print(f"Total: {size}")
    print(f"{num_correct/size*100:.2f}%")


