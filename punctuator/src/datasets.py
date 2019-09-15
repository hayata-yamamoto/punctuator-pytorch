from typing import List, Optional

import pandas as pd
from nltk import tokenize
from tqdm import tqdm

from punctuator.src.core.path_manager import PathManager


def ted_data() -> pd.DataFrame:
    return pd.read_csv(PathManager.RAW / 'transcripts.csv')


def write_txt(contents: List[str], filename: str) -> None:
    with open(filename, 'w') as f:
        [f.write(s + '\n') for s in contents]


def make_records(sentence: str, res: Optional[List[str]] = None) -> List[str]:
    words = tokenize.word_tokenize(sentence)

    if res is None:
        res = []

    for i in range(len(words)-1):
        if words[i] == '.':
            continue
        s = words[i].lower()
        if words[i+1] == '.':
            res.append(f'{s} PERIOD')
            continue
        if words[i+1] == ',':
            res.append(f"{s} COMMA")
            continue
        if words[i+1] == '?':
            res.append(f'{s} QUESTION')
            continue
        res.append(f'{s} NONE')

    res.append('')
    return res


def make_dataset(df: pd.Series):
    res = None
    for i, sent in tqdm(df.iteritems(), total=df.shape[0]):
        res = make_records(sent, res)
    return res
