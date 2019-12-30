from typing import List

import pandas as pd


def tagmap(word: str) -> str:
    if word in [".", ";", "!"]:
        return "PERIOD"
    if word == "?":
        return "QUESTION"
    if word in [",", ":", '-']:
        return "COMMA"
    else:
        return "O"


def tagging(sentence: str) -> str:
    words = sentence.replace(".", " .").replace("?", " ?").replace(",",
                                                                   " ,").replace("!", " !").replace(";", " ;").replace(
                                                                       ":", " :").replace('-', ' -').split(" ")
    sent = []

    for i in range(len(words) - 1):
        w = words[i].replace(".", "").replace("?", "").replace("!", "").replace(",", "")
        if w == "":
            continue

        sent.append(f"{w}###{tagmap(words[i+1])}")
    return " ".join(sent)


def reconstruct(seq: List[str], labels: List[str]) -> str:
    res = []
    for s, l in zip(seq, labels):
        if l == "PERIOD":
            res.append(s + ".")
        elif l == "QUESTION":
            res.append(s + "?")
        elif l == "COMMA":
            res.append(s + ",")
        else:
            res.append(s)
    return " ".join(res)


def make_data(dataset: List[str], lang: str, filename: str) -> None:
    pd.DataFrame([tagging(d[lang]) for d in dataset]).to_csv(filename, index=False)
