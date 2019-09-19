from pathlib import Path
from typing import List, Optional, Dict, Iterable, Callable, Tuple

import pandas as pd
import json
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
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame.from_records(data)

        for i, row in df.iterrows():
            tokens = [Token(token) for token in row['tokens']]
            yield self.text_to_instance(tokens=tokens, tags=row['tags'])


def ted_data() -> pd.DataFrame:
    return pd.read_csv(PathManager.RAW / 'transcripts.csv')


def write_txt(contents: List[str], filename: Path) -> None:
    with open(filename, 'w') as f:
        [f.write(s + '\n') for s in contents]


def tagmap(word: str) -> str:
    if word == ".":
        return "PERIOD"
    if word == "?":
        return "QUESTION"
    if word == ",":
        return "COMMA"
    else:
        return "O"


def tagging(sentence: str) -> Tuple[List[str], List[str]]:
    words = tokenize.word_tokenize(sentence)
    tokens, tags = [], []

    for i in range(len(words)-1):
        if words[i] in ['.', '?', ',']:
            continue

        tokens.append(words[i].lower())
        tags.append(tagmap(words[i+1]))
    return tokens, tags


def make_records(df: pd.Series) -> List[Dict[str, str]]:
    records = []
    for i, sent in tqdm(df.iteritems(), total=df.shape[0]):
        tokens, tags = tagging(sentence=sent)
        records.append({
            "tokens": tokens,
            "tags": tags
        })
    return records
