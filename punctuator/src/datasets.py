from pathlib import Path
from typing import List, Optional, Dict, Iterable, Callable

import pandas as pd
from allennlp.data import Instance, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from nltk import tokenize
from tqdm import tqdm

from punctuator.src.core.path_manager import PathManager


class TedDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        tokens, tags = [], []
        with open(file_path, 'r') as f:
            for line in f:
                if line == '\n':
                    yield self.text_to_instance(tokens=tokens, tags=tags)
                    tokens, tags = [], []
                else:
                    pairs = line.strip().split()
                    tokens.append(Token(pairs[0]))
                    tags.append(pairs[1])


def ted_data() -> pd.DataFrame:
    return pd.read_csv(PathManager.RAW / 'transcripts.csv')


def write_txt(contents: List[str], filename: Path) -> None:
    with open(filename, 'w') as f:
        [f.write(s + '\n') for s in contents]


def make_records(sentence: str, res: Optional[List[str]] = None) -> List[str]:
    words = tokenize.word_tokenize(sentence)

    if res is None:
        res = []

    for i in range(len(words)-1):
        if words[i] in ['.', '?', ',']:
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
        res.append(f'{s} O')

    res.append('')
    return res


def make_dataset(df: pd.Series):
    res = None
    for i, sent in tqdm(df.iteritems(), total=df.shape[0]):
        res = make_records(sent, res)
    return res
