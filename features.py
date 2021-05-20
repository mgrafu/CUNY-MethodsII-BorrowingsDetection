"""Borrowing detection features."""


from typing import Dict, List, Tuple


def append_surrounding_tokens(
    feats: Dict[str, str], sent: List[str], index: int
):
    for i in range(-2, 3):
        adjacent_t = index + i
        if adjacent_t > -1 and adjacent_t < len(sent):
            name = "t" + str(["-" if i < 0 else "+"][0]) + str(abs(i))
            feats[name] = (
                "[NUMERIC]"
                if sent[adjacent_t].isnumeric()
                else sent[adjacent_t].casefold()
            )
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
    concatenation = (
        "[NUMERIC]" if tokens[index].isnumeric() else tokens[index].casefold()
    )
    concatenation += "^"
    concatenation += (
        "[NUMERIC]"
        if tokens[index + offset].isnumeric()
        else tokens[index + offset].casefold()
    )
    return concatenation


def get_casing(word: str) -> str:
    if word.islower():
        casing = "lower"
    elif word.isupper():
        casing = "upper"
    elif word.istitle():
        casing = "title"
    else:
        casing = "mixed"
    return casing


def get_word_endings(word: str) -> List[str]:
    endings = []
    for i in range(4, 0, -1):
        if len(word) > i:
            endings.append(word[-i:])
    return endings


def get_unique_env(word: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    letters = ["y", "q", "h", "k"]
    letters_present = []
    for i in range(len(word)):
        if word[i] in letters and len(word) > 1:
            environment = []
            if i > 0:
                environment.append(("left", word[i - 1]))
            if i > 1:
                environment.append(("left2", word[i - 2 : i - 1]))
            if i + 1 < len(word):
                environment.append(("right", word[i + 1]))
            if i + 2 < len(word):
                environment.append(("right2", word[i + 1 : i + 2]))
            if environment:
                letters_present.append((word[i], environment))
    return letters_present


def get_accent(word: str) -> str:
    accent = ""
    for char in word:
        if char in ["á", "é", "í", "ó", "ú"]:
            accent = char
    return accent


def get_non_alphabet(word: str) -> List[str]:
    alphabet = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "á",
        "é",
        "í",
        "ó",
        "ú",
        "ü",
        "ñ",
    ]
    letters_present = []
    if len(word) > 1:
        for letter in word:
            if letter not in alphabet:
                letters_present.append(letter)
    return letters_present


def extract_sent_feats(sent: List[str]):
    feats = []
    for i in range(len(sent)):
        sent_feats: Dict[str, str] = dict()
        append_surrounding_tokens(sent_feats, sent, i)
        sent_feats["cap(t)"] = get_casing(sent[i])
        endings = get_word_endings(sent[i])
        if endings:
            for end in endings:
                sent_feats["last" + str(len(end))] = end
        unique_envs = get_unique_env(sent[i])
        if unique_envs:
            for letter, env in unique_envs:
                sent_feats[letter] = "y"
                for name, value in env:
                    sent_feats[name] = value
        if get_accent(sent[i]):
            sent_feats["acc"] = get_accent(sent[i])
        non_alphabet_letters = get_non_alphabet(sent[i])
        if non_alphabet_letters:
            for letter in non_alphabet_letters:
                sent_feats[letter] = "y"
        feats.append(sent_feats)
    return feats
